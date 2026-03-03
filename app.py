import os
import re
import json
import time
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "priority_model.joblib")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "priority_vectorizer.joblib")
RETRIEVAL_PATH = os.path.join(ARTIFACTS_DIR, "retrieval_bundle.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")
AUDIT_PATH = os.path.join(LOGS_DIR, "triage_runs.jsonl")

ALLOWED_PRIORITIES = ["low", "medium", "high", "urgent"]
PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2, "urgent": 3}


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
LONG_NUM_RE = re.compile(r"\b\d{4,}\b")


def mask_pii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = IP_RE.sub("[IP]", t)
    t = URL_RE.sub("[URL]", t)
    t = LONG_NUM_RE.sub("[NUMBER]", t)
    return t


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

SECURITY_PATTERNS = [
    r"\bstolen\b", r"\btheft\b", r"\bfraud\b", r"\bscam\b", r"\bhacked\b",
    r"\bbreach\b", r"\bcompromised\b", r"\bunauthorized\b", r"\bsuspicious login\b",
    r"\bphishing\b", r"\bmalware\b", r"\bransomware\b",
]

OUTAGE_PATTERNS = [
    r"\bdown\b", r"\boutage\b", r"\bunavailable\b", r"\boffline\b",
    r"\b503\b", r"\b500\b", r"\bservice unavailable\b", r"\bprod\b", r"\bproduction\b",
]

URGENT_WORDS = [r"\burgent\b", r"\basap\b", r"\bimmediately\b", r"\bсрочно\b", r"\bнемедленно\b"]

MONEY_PATTERNS = [
    r"\$\s*\d[\d,.\s]*", r"\b\d[\d,.\s]*\s*(usd|dollars|eur|€|руб|rub|рублей|₽)\b",
    r"\b\d+\s*(k|тыс)\b",
]


def _has_any(patterns: List[str], text_lower: str) -> bool:
    return any(re.search(p, text_lower) for p in patterns)


def policy_min_priority(text: str) -> Optional[str]:
    """
    Возвращает минимально допустимый приоритет по правилам.
    None означает "policy не вмешивается".
    """
    t = (text or "").lower()

    if _has_any(SECURITY_PATTERNS, t):
        if _has_any(MONEY_PATTERNS, t):
            return "urgent"
        return "urgent"

    if _has_any(OUTAGE_PATTERNS, t):
        return "high"

    if _has_any(URGENT_WORDS, t):
        return "high"

    return None


def apply_policy_floor(ml_priority: str, floor: Optional[str]) -> str:
    if floor is None:
        return ml_priority
    return floor if PRIORITY_RANK[floor] > PRIORITY_RANK[ml_priority] else ml_priority


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH) or not os.path.exists(RETRIEVAL_PATH):
        raise FileNotFoundError(
            "Не найдены артефакты модели. "
            "Открой notebooks/it_ticket_triage_full.ipynb и выполни секцию 'Save artifacts', "
            "или используй артефакты из репозитория."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    retrieval = joblib.load(RETRIEVAL_PATH)

    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return model, vectorizer, retrieval, metrics


def ml_predict_priority(model, vectorizer, text_masked: str) -> Tuple[str, float, Dict[str, float]]:
    X = vectorizer.transform([text_masked])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        proba_map = {c: float(p) for c, p in zip(classes, proba)}
        pred = classes[int(np.argmax(proba))]
        conf = float(np.max(proba))
        return pred, conf, proba_map
    pred = model.predict(X)[0]
    return str(pred), 0.0, {}


def retrieve_similar(retrieval_bundle: Dict[str, Any], vectorizer, query_masked: str, top_k: int = 5) -> pd.DataFrame:
    templates_df = retrieval_bundle["templates_df"].copy()
    template_matrix = retrieval_bundle["template_matrix"]

    qv = vectorizer.transform([query_masked])
    sims = cosine_similarity(qv, template_matrix).ravel()
    top_idx = np.argsort(-sims)[:top_k]

    rows = []
    for idx in top_idx:
        row = templates_df.iloc[int(idx)].to_dict()
        row["similarity"] = float(sims[int(idx)])
        dist = row.get("priority_dist", {})
        row["priority_dist_str"] = ", ".join([f"{k}:{dist.get(k,0):.2f}" for k in ALLOWED_PRIORITIES if k in dist])
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["similarity", "count", "priority_mode", "priority_dist_str", "issue_type_mode", "product_area_mode", "template_text"]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols]


def explain_linear(model, vectorizer, text_masked: str, target_class: str, top_n: int = 12) -> List[Tuple[str, float]]:
    if not hasattr(model, "coef_"):
        return []

    X = vectorizer.transform([text_masked])
    if X.nnz == 0:
        return []

    classes = list(model.classes_)
    if target_class not in classes:
        return []

    k = classes.index(target_class)
    coef = model.coef_[k] 

    idx = X.indices
    vals = X.data * coef[idx]
    if len(vals) == 0:
        return []

    order = np.argsort(-vals)
    feats = vectorizer.get_feature_names_out()
    out = []
    for j in order[: top_n * 2]:
        if vals[j] <= 0:
            continue
        out.append((str(feats[idx[j]]), float(vals[j])))
        if len(out) >= top_n:
            break
    return out


@st.cache_resource
def load_llm():
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Не найден OPENAI_API_KEY. Добавь его в .env (см. .env.example).")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

    return ChatOpenAI(model=model_name, temperature=temperature)


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — AI ассистент для triage IT тикетов. "
     "Твоя задача: объяснить приоритет и дать шаги. "
     "НЕЛЬЗЯ раскрывать персональные данные. "
     "Выводи ТОЛЬКО валидный JSON без лишнего текста."),
    ("human",
     """Ticket (masked):
{ticket}

ML prediction:
- ml_priority: {ml_priority}
- ml_confidence: {ml_confidence}
- ml_proba: {ml_proba}

Policy floor (hard rules):
- policy_floor: {policy_floor}
- final_priority: {final_priority}

Similar historical templates (top):
{similar}

Return JSON with schema:
{{
  "summary": string,
  "why_this_priority": [string, ...],
  "immediate_actions": [string, ...],
  "clarifying_questions": [string, ...],
  "risks": [string, ...],
  "suggested_priority": "low"|"medium"|"high"|"urgent"
}}

Notes:
- suggested_priority может отличаться от final_priority, но final_priority уже зафиксирован policy/ML.
- Если тикет явно про кражу денег/фрод/взлом/утечку — suggested_priority должен быть urgent.
""")
])


def safe_parse_json(text: str) -> Dict[str, Any]:

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"summary": "GPT returned non-JSON", "raw": text}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"summary": "GPT returned invalid JSON", "raw": text}


def gpt_explain(llm, ticket_masked: str, ml_priority: str, ml_conf: float, ml_proba: Dict[str, float],
                policy_floor: Optional[str], final_priority: str, similar_df: pd.DataFrame) -> Dict[str, Any]:
    similar_short = []
    for _, r in similar_df.head(5).iterrows():
        similar_short.append({
            "similarity": float(r["similarity"]),
            "count": int(r["count"]),
            "priority_mode": str(r["priority_mode"]),
            "priority_dist": str(r["priority_dist_str"]),
            "template_text": str(r["template_text"])[:200],
        })

    msg = PROMPT.format_messages(
        ticket=ticket_masked,
        ml_priority=ml_priority,
        ml_confidence=f"{ml_conf:.3f}",
        ml_proba=json.dumps(ml_proba, ensure_ascii=False),
        policy_floor=policy_floor or "none",
        final_priority=final_priority,
        similar=json.dumps(similar_short, ensure_ascii=False, indent=2),
    )
    resp = llm.invoke(msg)
    return safe_parse_json(getattr(resp, "content", str(resp)))


def normalize_priority(p: Any) -> Optional[str]:
    if not isinstance(p, str):
        return None
    p = p.strip().lower()
    return p if p in ALLOWED_PRIORITIES else None


def decide_final_priority(ml_priority: str, ml_conf: float, policy_floor: Optional[str], gpt_suggested: Optional[str]) -> Tuple[str, str]:
    """
    Возвращает (final_priority, decision_reason).
    Правило:
    - сначала policy floor (hard rules)
    - если ml_conf низкий -> можно взять gpt_suggested (но всё равно не ниже policy floor)
    """
    floor_applied = apply_policy_floor(ml_priority, policy_floor)

    if ml_conf < 0.45 and gpt_suggested:
        combined = apply_policy_floor(gpt_suggested, policy_floor)
        return combined, "gpt_fallback_due_to_low_ml_confidence"
    return floor_applied, "ml_plus_policy_floor"


def audit_log(payload: Dict[str, Any]) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_feedback(ticket_hash: str, ticket_masked: str, predicted: str, corrected: str, note: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticket_hash": ticket_hash,
        "ticket_masked": ticket_masked,
        "predicted_priority": predicted,
        "corrected_priority": corrected,
        "note": note,
    }
    if os.path.exists(FEEDBACK_PATH):
        df = pd.read_csv(FEEDBACK_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(FEEDBACK_PATH, index=False)


st.set_page_config(page_title="IT Ticket Triage", layout="wide")

st.title("IT Ticket Triage")

with st.expander("Как все работает?", expanded=False):
    st.markdown(
        """
- **ML модель (локально)** предсказывает priority по тексту.
- **Policy слой** поднимает приоритет для security/fraud/outage/.
- **GPT (Transformer)** всегда генерирует объяснение + next steps, и может быть fallback при низкой уверенности ML.
- **Similar tickets** показываются.
        """
    )

try:
    model, vectorizer, retrieval_bundle, metrics = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    ticket = st.text_area("Вставь текст тикета", height=160, placeholder="Например: My account was hacked and money was stolen...")
    triage_btn = st.button("Run triage", type="primary")


if triage_btn:
    if not ticket.strip():
        st.warning("Введите текст тикета.")
        st.stop()

    ticket_hash = sha256_text(ticket)
    ticket_masked = mask_pii(ticket)

    ml_priority, ml_conf, ml_proba = ml_predict_priority(model, vectorizer, ticket_masked)

    similar_df = retrieve_similar(retrieval_bundle, vectorizer, ticket_masked, top_k=5)

    floor = policy_min_priority(ticket_masked)
    priority_after_floor = apply_policy_floor(ml_priority, floor)

    try:
        llm = load_llm()
        gpt_json = gpt_explain(
            llm=llm,
            ticket_masked=ticket_masked,
            ml_priority=ml_priority,
            ml_conf=ml_conf,
            ml_proba=ml_proba,
            policy_floor=floor,
            final_priority=priority_after_floor,
            similar_df=similar_df,
        )
    except Exception as e:
        gpt_json = {"summary": "GPT error", "error": str(e)}
    gpt_suggested = normalize_priority(gpt_json.get("suggested_priority"))

    final_priority, decision_reason = decide_final_priority(
        ml_priority=ml_priority,
        ml_conf=ml_conf,
        policy_floor=floor,
        gpt_suggested=gpt_suggested,
    )

    top_feats = explain_linear(model, vectorizer, ticket_masked, target_class=ml_priority, top_n=12)

    audit_payload = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "ticket_hash": ticket_hash,
        "ticket_masked": ticket_masked,
        "ml_priority": ml_priority,
        "ml_confidence": ml_conf,
        "ml_proba": ml_proba,
        "policy_floor": floor,
        "priority_after_policy": priority_after_floor,
        "gpt_suggested_priority": gpt_suggested,
        "final_priority": final_priority,
        "decision_reason": decision_reason,
        "similar_templates": [
            {
                "similarity": float(r["similarity"]),
                "count": int(r["count"]),
                "priority_mode": str(r["priority_mode"]),
                "priority_dist": str(r["priority_dist_str"]),
                "template_text": str(r["template_text"])[:200],
            }
            for _, r in similar_df.iterrows()
        ],
    }
    audit_log(audit_payload)

    st.divider()
    out1, out2 = st.columns([1, 1])

    with out1:
        st.subheader("Priority result")
        st.markdown(f"**Final priority:** `{final_priority.upper()}`")
        st.markdown(f"- ML priority: `{ml_priority}` (confidence: `{ml_conf:.3f}`)")
        st.markdown(f"- Policy floor: `{floor or 'none'}`")
        st.markdown(f"- GPT suggested: `{gpt_suggested or 'none'}`")
        st.markdown(f"- Decision: `{decision_reason}`")

        if top_feats:
            st.markdown("**Почему ML так решил (топ фичи):**")
            st.table(pd.DataFrame(top_feats, columns=["feature", "contribution"]))
        else:
            st.info("Нет локальных фич для объяснения (возможно, все слова OOV).")

    with out2:
        st.subheader("GPT explanation")
        st.json(gpt_json)

    st.subheader("Similar historical ticket templates (deduped)")
    st.dataframe(similar_df, use_container_width=True)

    st.divider()
    st.subheader("Feedback (для улучшения модели)")
    fb_col1, fb_col2, fb_col3 = st.columns([1, 2, 2])
    with fb_col1:
        corrected = st.selectbox("Правильный priority", options=ALLOWED_PRIORITIES, index=ALLOWED_PRIORITIES.index(final_priority))
    with fb_col2:
        note = st.text_input("Комментарий (почему так)")
    with fb_col3:
        if st.button("Save feedback"):
            save_feedback(ticket_hash, ticket_masked, final_priority, corrected, note)
            st.success("Feedback сохранён в data/feedback.csv")
