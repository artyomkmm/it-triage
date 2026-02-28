# IT Ticket Triage (Notebook-first) + Streamlit

Это проект для triage IT-тикетов: пользователь вводит текст тикета → система возвращает:
- `final_priority` (low / medium / high / urgent)
- объяснение и next steps от GPT (Transformer)
- топ похожих исторических шаблонов тикетов (без дублей)
- audit-лог (без утечек PII)
- feedback → можно улучшать модель

## 1) Быстрый старт (чтобы всё заработало)

### 1.1 Установка зависимостей
```bash
python -m venv .venv
# mac/linux:
source .venv/bin/activate
# windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 1.2 Ключ OpenAI
Скопируй `.env.example` → `.env` и вставь ключ:
```bash
cp .env.example .env
```

### 1.3 Запуск Streamlit
```bash
streamlit run app.py
```

Откроется UI. Вставляй тикет → получишь результат.

## 2) Где обучение, EDA, метрики, объяснимость?

Основной файл проекта — ноутбук:

- `notebooks/it_ticket_triage_full.ipynb`

Там:
- большой EDA с графиками
- анализ дублей/шума в разметке
- обучение модели (train/test split)
- метрики, confusion matrix
- “какие слова важны” (feature importance)
- RAG-подобный retrieval похожих тикетов (уникальные шаблоны)
- GPT (Transformer) для объяснения и для fallback, если ML не уверен
- сохранение артефактов в `artifacts/`

## 3) Артефакты модели
Streamlit использует:
- `artifacts/priority_vectorizer.joblib`
- `artifacts/priority_model.joblib`
- `artifacts/retrieval_bundle.joblib`
- `artifacts/metrics.json`

Ты можешь переобучить всё из ноутбука (в конце есть секция “Save artifacts”),
после чего просто перезапустить Streamlit.

## 4) Важно про качество
Этот датасет `synthetic_it_support_tickets.csv` **сильно шаблонный**:
- всего ~96 уникальных `initial_message`
- приоритеты внутри одинаковых текстов часто разные

В ноутбуке есть отдельный раздел, который это доказывает графиками.
Из-за этого чистый ML часто “не может выучить” настоящие закономерности.

Поэтому в продовой логике triage:
- критичные кейсы (security/fraud/outage, украли деньги и т.п.) поднимаются policy-слоем
- GPT используется как объяснялка и fallback для “нестандартных” тикетов

Это даёт:
- предсказуемость (policy)
- интерпретируемость (важные слова/фичи + GPT)
- обучаемость (feedback)

