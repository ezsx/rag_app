## Data Model

### Qdrant Collection Schema (Phase 1 — целевое)

**Коллекция**: `news` (по умолчанию, задаётся `QDRANT_COLLECTION`)

```
Named vectors:
  dense_vector:   multilingual-e5-large (1024-dim, cosine distance)
                  prefix при индексировании: "passage: " + text
  sparse_vector:  Qdrant/bm25 (language="russian", Snowball stemmer)
                  при поиске: "query: " + query

Payload (per point):
  text:       str    — полный текст сообщения
  channel:    str    — "@channel_name"
  channel_id: int    — Telegram channel ID
  message_id: int    — Telegram message ID
  date:       str    — "YYYY-MM-DDTHH:MM:SS" (ISO 8601)
  author:     str    — имя автора/канала
  url:        str    — ссылка на сообщение (опционально)

Point ID: uuid (Qdrant-generated) или str "{channel_name}:{message_id}" (upsert key)
```

**Hybrid search запрос:**
```python
client.query_points(
    collection_name=QDRANT_COLLECTION,
    prefetch=[
        Prefetch(query=dense_vector, using="dense_vector", limit=20),
        Prefetch(query=sparse_vector, using="sparse_vector", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
    with_vectors=True,  # ОБЯЗАТЕЛЬНО для composite coverage computation
)
```

**Docker**: Qdrant storage — **только named volume** `qdrant_data`. Bind mounts → silent data corruption на Windows.

---

### Candidate (HybridRetriever output — Phase 1)

```
ScoredPoint (из Qdrant):
  id:          str    — point id
  score:       float  — RRF score (для ранжирования, НЕ для coverage)
  payload:     dict   — {text, channel, date, author, ...}
  vector:      dict   — {dense_vector: [...]} (при with_vectors=True)

После извлечения:
  cosine_sim:  float  — dot(query_vec, doc_vec) [оба L2-нормированы]
                        используется для composite coverage computation
```

После RRF fusion → после MMR (`rescore=MmrQuery(lambda_mult=0.5)`) → после BGE reranker → топ-N.

---

### ChromaDB Schema (Phase 0 — устаревшее, для справки)

> **Deprecated**: заменяется Qdrant (DEC-0015). Оставлено для понимания migration path.

```
ChromaDB collection: `news_demo4`
  id:        str    — "{channel_name}:{message_id}"
  document:  str    — текст сообщения
  embedding: float[] — multilingual-e5-large (1024 dims)
  metadata:  {channel, date, message_id, author, url, type}
```

---

### Eval Dataset Schema (v1.0)

**Файл**: `datasets/eval_dataset.json`
**Генератор**: `scripts/generate_eval_dataset.py` (из Qdrant)

```json
{
  "version": "1.0",
  "created_at": "YYYY-MM-DD",
  "generation_model": "Qwen3-8B",
  "statistics": {
    "total": 200,
    "by_type": {"factual": 70, "temporal": 40, "aggregation": 40, "negative": 30, "comparative": 20}
  },
  "examples": [
    {
      "id":                   "eval_0001",
      "question":             "str — вопрос на русском",
      "expected_answer":      "str — ожидаемый ответ",
      "contexts":             ["str — текст документа"],
      "expected_document_ids": ["str — qdrant point id"],
      "question_type":        "factual|temporal|aggregation|negative|comparative",
      "answerable":           true,
      "metadata": {
        "qdrant_point_id": "str",
        "source":          "str — channel name",
        "collection":      "str"
      }
    }
  ]
}
```

**Распределение типов (обязательно!):**
- `factual` 35% — конкретный факт из контекста
- `temporal` 20% — даты, сроки, хронология
- `aggregation` 20% — объединение информации из нескольких источников
- `negative` 15% — вопросы, на которые нельзя ответить из контекста (проверка hallucination)
- `comparative` 10% — сравнение двух сущностей

> Без принудительного распределения → 95% factual → завышенные метрики (R05, "Know Your RAG" arXiv 2411.19710).

**Минимальный размер**: 200 примеров (margin of error ±5.5% при 95% CI).

---

### JWT Payload

```json
{
  "sub": "admin",
  "roles": ["admin", "write", "read"],
  "exp": <unix_timestamp>
}
```

Algorithm: HS256, secret: `JWT_SECRET_KEY` из `.env`.

---

### Settings Key Fields (актуально Phase 1)

| Переменная | Default (Phase 1) | Описание |
|-----------|------------------|---------|
| `LLM_MODEL_KEY` | `qwen3-8b` | Имя модели для llama-server (имя GGUF файла) |
| `LLM_BASE_URL` | `http://host.docker.internal:8080/v1` | URL llama-server (или vLLM после Proxmox) |
| `EMBEDDING_MODEL_KEY` | `multilingual-e5-large` | Ключ embedding |
| `QDRANT_URL` | `http://qdrant:6333` | URL Qdrant в Docker compose |
| `QDRANT_COLLECTION` | `news` | Имя коллекции Qdrant |
| `COVERAGE_THRESHOLD` | `0.65` | Порог для refinement (был 0.80) |
| `MAX_REFINEMENTS` | `2` | Макс. refinement раундов (было 1) |
| `HYBRID_ENABLED` | `true` | Включить HybridRetriever |
| `ENABLE_RERANKER` | `true` | Включить BGE reranker |
| `AGENT_MAX_STEPS` | `15` | Максимум шагов ReAct |
| `LLM_CONTEXT_SIZE` | `8192` | Context window (было 10000; 8192 экономит VRAM) |
