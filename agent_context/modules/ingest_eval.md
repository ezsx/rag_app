# Ingest & Evaluation Module (Phase 1)

## Telegram Ingest Pipeline

### Ключевые файлы

- `scripts/ingest_telegram.py` — основной скрипт ingestion
- `src/services/ingest_service.py` — сервис фоновых задач ingestion (API endpoint)
- `Dockerfile.ingest` — контейнер для ingestion
- `sessions/` — Telethon session files (не коммитить)

### Текущий статус

> **SPEC-RAG-06 (pending)**: ingest_telegram.py ещё Phase 0 (ChromaDB + SentenceTransformer).
> После реализации SPEC-RAG-06 будет: Qdrant + TEI HTTP + fastembed sparse.

### Запуск ingestion

```bash
# Через Docker Compose (рекомендуется)
docker compose -f deploy/compose/compose.dev.yml run --rm ingest \
  --channel @channel_name \
  --since 2024-01-01 \
  --until 2024-01-31

# Collection по умолчанию: QDRANT_COLLECTION env или "news"
# Env переменные для Telethon:
# TG_API_ID, TG_API_HASH, TG_SESSION=/app/sessions/telegram.session
```

### Целевой формат документов в Qdrant (Phase 1, после SPEC-RAG-06)

```
Point:
  id:      "{channel_name}:{message_id}"     — детерминированный, idempotent upsert
  vector:  {
    "dense_vector": list[float]              — TEI embed (multilingual-e5-large, 1024-dim)
    "sparse_vector": SparseVector            — fastembed (Qdrant/bm25, language="russian")
  }
  payload: {
    "text": str,
    "channel": str,                          — без "@"
    "channel_id": int,
    "message_id": int,
    "date": "YYYY-MM-DDTHH:MM:SS",
    "author": str | null,
    "url": "https://t.me/{channel}/{msg_id}"
  }
```

## Evaluation Tooling (MVP)

### Ключевые файлы

- `scripts/evaluate_agent.py` — CLI evaluation скрипт
- `datasets/eval_dataset.json` — датасет оценки
- `docs/ai/planning/agent_evaluation_spec.md` — спецификация evaluation
- `results/raw/` — per-query JSON результаты
- `results/reports/` — агрегированные Markdown отчёты

### Формат eval_dataset.json

```json
[
  {
    "id": "q001",
    "query": "Вопрос для агента",
    "category": "factual|analytical|temporal",
    "expected_documents": ["channel:msg_id"],
    "answerable": true,
    "notes": "опциональный комментарий"
  }
]
```

### Запуск evaluation

```bash
# Требует запущенного API (docker compose --profile api up)
python scripts/evaluate_agent.py \
  --collection news \
  --limit 10 \
  --dry-run          # только валидация датасета

python scripts/evaluate_agent.py \
  --collection news \
  --skip-markdown    # без Markdown отчёта
  --api-key $API_KEY
```

### Метрики MVP

| Метрика | Описание |
|--------|---------|
| `agent_latency_sec` | latency p50/p95/max для агента |
| `baseline_latency_sec` | latency для baseline `/v1/qa` |
| `agent_coverage` | coverage из compose_context (naive → composite после SPEC-RAG-07) |
| `recall@5` | попадание expected_documents в top-5 hits |
| `refinements` | сколько раз агент делал refinement (max 2, DEC-0019) |
| `verification` | confidence из verify tool |

### Phase 2 (запланировано)

- LLM-judge для correctness/faithfulness (Qwen3-8B, русские промпты)
- Citation Precision
- Conciseness score
- Автоматизация регрессии при изменении промптов/настроек

## Инварианты

- Telethon session хранится в `sessions/` — volume в Docker, не в git.
- Point ID в Qdrant: `"{channel_name}:{message_id}"` — уникальность, idempotent upsert.
- Evaluation скрипт НЕ изменяет коллекцию — только читает.
- `datasets/eval_dataset.json` — основной датасет, не удалять без замены.
