# IT Ticket Triage (ML + Policy + GPT)

This project is an end-to-end **ticket triage** system: a user submits a ticket text and the system predicts the ticket **priority** (`low / medium / high / urgent`), shows relevant context, and produces a structured explanation of the decision.

## What it does
- Predicts priority for incoming tickets.
- Applies **hard policy rules** to guarantee minimum severity for critical signals (e.g., fraud/security/outage).
- Generates an **auditable JSON** explanation: summary, reasons, immediate actions, clarifying questions, risks.
- Shows **similar historical tickets** (deduplicated) to improve transparency and debugging.
- Masks PII before any model/LLM processing.

## Core approach (architecture)
**ML → Policy → GPT**
- **ML (fast & cheap):** a local scikit-learn classifier provides `ml_priority` + confidence.
- **Policy (reliability):** deterministic rules set a minimum floor priority for critical cases.
- **GPT (audit + support):** used after ML+policy to generate structured JSON output and can **only raise** priority when ML confidence is low (never lower it, never bypass policy).

## Technologies used
- **Jupyter Notebook**: EDA, data quality checks, training, metrics, artifact export.
- **scikit-learn**: TF-IDF vectorization (char n-grams) + classification (e.g., Logistic Regression / SGD), evaluation metrics.
- **LangChain + OpenAI**: GPT-based structured reasoning/explanation in **strict JSON** format.
- **Streamlit**: simple UI for demo/“prod-like” interaction.
- **joblib**: saving/loading model, vectorizer, retrieval bundle, metrics.
- **Python**: glue code, policy rules, PII masking, logging.

## Outputs (artifacts)
After training, the notebook exports model artifacts used by the Streamlit app:
- `priority_vectorizer.joblib`
- `priority_model.joblib`
- `retrieval_bundle.joblib`
- `metrics.json`

## Notes on robustness
- The project includes sanity checks to avoid deploying a “blind” vectorizer (e.g., zero feature matches / NNZ≈0).
- Deduplication is applied to the retrieval layer to prevent showing the same ticket multiple times.
- Policy rules ensure critical scenarios never get under-prioritized even if ML is uncertain.
