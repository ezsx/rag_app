# Ingest & Evaluation Module

## Telegram Ingest Pipeline

### Ключевые файлы

- `scripts/ingest_telegram.py` — основной скрипт ingestion
- `scripts/payload_enrichment.py` — NER, URL/arxiv extraction, temporal fields (SPEC-RAG-12)
- `scripts/migrate_collection.py` — создание news_colbert_v2 с indexes
- `scripts/add_colbert_vectors.py` — пакетное добавление ColBERT vectors
- `src/services/ingest_service.py` — сервис фоновых задач ingestion (API endpoint)
- `sessions/` — Telethon session files (не коммитить)

### Текущий статус

> Коллекция: `news_colbert_v2` (dense 1024 + sparse BM25 + ColBERT 128-dim MaxSim).
> 13088 points из 36 каналов (июль 2025 — март 2026).
> Enriched payload: entities, arxiv_ids, urls, url_domains, lang, year_week + 16 payload indexes.

### Запуск ingestion

```bash
# Через Docker Compose
docker compose -f deploy/compose/compose.dev.yml run --rm ingest \
  --channel @channel_name \
  --since 2024-01-01 \
  --until 2024-01-31

# Collection: QDRANT_COLLECTION env (default: news_colbert_v2)
```

### Формат документов в Qdrant (после SPEC-RAG-12)

```
Point:
  id:      UUID5("{channel_id}:{message_id}")  — стабильный, idempotent upsert
  vector:  {
    "dense_vector": list[float]     — Qwen3-Embedding-0.6B, 1024-dim
    "sparse_vector": SparseVector   — fastembed BM25 (russian)
    "colbert_vector": MultiVector   — jina-colbert-v2, 128-dim per-token MaxSim
  }
  payload: {
    text, channel, channel_id, message_id, date, author, url,
    entities[], entity_orgs[], entity_models[],
    arxiv_ids[], urls[], url_domains[], github_repos[], hashtags[],
    lang, year_week, year_month, text_length,
    is_forward, forwarded_from_id, forwarded_from_name, reply_to_msg_id,
    media_types[], root_message_id, has_arxiv, has_links
  }
```

## Evaluation Pipeline V2 (SPEC-RAG-14, Phase 3.3)

### Ключевые файлы

- `scripts/evaluate_agent.py` — eval runner: tool tracking, failure attribution, LLM judge, reports
- `datasets/eval_golden_v1.json` — golden dataset (25 Qs, 6 categories)
- `datasets/eval_dataset_quick.json` (v1, 10 Qs), `eval_dataset_quick_v2.json` (v2, 10 Qs), `eval_dataset_v3.json` (v3, 30 Qs)
- `docs/specifications/active/SPEC-RAG-14-evaluation-pipeline.md` — спецификация
- `docs/research/reports/R18-deep-evaluation-methodology-dataset.md` — целевой eval blueprint
- `results/raw/` — per-question JSON результаты (unified format)
- `results/reports/` — агрегированные JSON + Markdown отчёты

### Golden dataset формат

```json
{
  "id": "golden_q01",
  "query": "Вопрос",
  "expected_answer": "Ожидаемый ответ",
  "category": "broad_search|constrained_search|compare_summarize|navigation|negative_refusal|future_baseline",
  "key_tools": ["search"],
  "forbidden_tools": ["list_channels"],
  "acceptable_alternatives": [],
  "answerable": true,
  "expected_refusal": false,
  "future_tool_flag": false,
  "calibration": false
}
```

### Запуск evaluation

```bash
# Без judge (быстрый — ~15 мин)
python scripts/evaluate_agent.py \
  --dataset datasets/eval_golden_v1.json \
  --skip-judge \
  --api-key $TOKEN

# С Claude judge
EVAL_JUDGE_API_KEY=sk-ant-... python scripts/evaluate_agent.py \
  --dataset datasets/eval_golden_v1.json \
  --judge claude \
  --api-key $TOKEN

# Legacy datasets (backward compatible)
python scripts/evaluate_agent.py \
  --dataset datasets/eval_dataset_v3.json \
  --skip-judge
```

### Метрики

| Метрика | Описание | Источник |
|---------|---------|----------|
| `recall_at_5` | Fuzzy match expected_documents vs citation_hits (±5/±50) | Программный |
| `key_tool_accuracy` | Binary: agent вызвал key_tool ∪ alternatives, не вызвал forbidden | Программный (SSE tool_invoked) |
| `factual_correctness` | 0.0/0.5/1.0 — фактическая корректность vs expected_answer | LLM judge (Claude API) |
| `usefulness` | 0/1/2 — полезность ответа для пользователя | LLM judge (Claude API) |
| `failure_type` | tool_hidden/tool_wrong/tool_failed/retrieval_empty/generation_wrong/refusal_wrong/judge_uncertain | Программный |

### Текущие результаты (golden_v1, 2026-03-24)

| Метрика | Значение | Примечание |
|---------|----------|------------|
| Key Tool Accuracy | **0.955** | Agent routing работает |
| Strict Recall@5 | ~0.43 | Занижен: dataset strictness + alternative evidence |
| Manual judge factual | **0.52** | Консенсус Claude + Codex |
| Manual judge useful | **1.14/2** | Консенсус Claude + Codex |
| Coverage | ~0.66 | |
| Navigation | key_tool=1.0, latency=7s | Fixed (P0) |
| Refusal | 2/3 wrong (stochastic) | Partially fixed (P1) |

### Инварианты

- Telethon session хранится в `sessions/` — volume в Docker, не в git.
- Point ID: UUID5 от `channel_id:message_id` (стабильный, не зависит от CLI hint).
- Evaluation скрипт НЕ изменяет коллекцию — только читает.
- Legacy datasets (v1/v2/v3) auto-detect при загрузке.
- `results/` — gitignored, large per_question.json хранятся локально.
