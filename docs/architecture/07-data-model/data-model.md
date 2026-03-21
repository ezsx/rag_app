## Data Model

### Qdrant Collection Schema

**Коллекция**: `news_colbert` (по умолчанию, задаётся `QDRANT_COLLECTION`)

```
Named vectors:
  dense_vector:    Qwen3-Embedding-0.6B (1024-dim, cosine distance)
  sparse_vector:   Qdrant/bm25 (language="russian", Snowball stemmer)
  colbert_vector:  jina-colbert-v2 (128-dim per token, MaxSim multi-vector)

Payload (per point):
  text:       str    — полный текст сообщения
  channel:    str    — "@channel_name"
  channel_id: int    — Telegram channel ID
  message_id: int    — Telegram message ID
  date:       str    — "YYYY-MM-DDTHH:MM:SS" (ISO 8601)
  author:     str    — имя автора/канала
  url:        str    — ссылка на сообщение (опционально)

Point ID: UUID5 deterministic — uuid5(NAMESPACE_URL, f"{channel}:{message_id}")
```

**Hybrid search запрос (текущий):**
```python
client.query_points(
    collection_name=QDRANT_COLLECTION,
    prefetch=[
        Prefetch(query=sparse_vector, using="sparse_vector", limit=max(k*10, 100)),
        Prefetch(query=dense_vector, using="dense_vector", limit=max(k*2, 20)),
    ],
    query=models.RrfQuery(rrf=models.Rrf(weights=[1.0, 3.0])),  # BM25 weight=3, dense weight=1
    limit=rrf_limit,
    with_vectors=True,  # ОБЯЗАТЕЛЬНО для composite coverage computation
)
```

**ColBERT MaxSim rerank (post-RRF):**
```python
# Если colbert_vector есть в коллекции:
query_tokens = gpu_server.colbert_encode(query)  # per-token 128-dim vectors
for candidate in rrf_results:
    doc_tokens = candidate.vector["colbert_vector"]  # stored per-token vectors
    candidate.colbert_score = maxsim(query_tokens, doc_tokens)
candidates.sort(key=lambda c: c.colbert_score, reverse=True)
```

**Channel dedup (post-rerank):**
```python
# max 2 docs from same channel → diversity
_channel_dedup(candidates, max_per_channel=2)
```

**Docker**: Qdrant storage — **только named volume** `qdrant_data`. Bind mounts → silent data corruption на Windows.

---

### Candidate (HybridRetriever output)

```
Candidate (внутренняя структура):
  id:            str    — point id (uuid5)
  text:          str    — текст документа
  channel:       str    — канал
  message_id:    int    — Telegram message ID
  date:          str    — ISO date
  dense_score:   float  — cosine similarity (для coverage)
  rrf_score:     float  — RRF score (для ранжирования, НЕ для coverage)
  colbert_score: float  — MaxSim score (для reranking, если ColBERT available)
  rerank_score:  float  — cross-encoder score (после bge-reranker-v2-m3)
```

Pipeline: BM25+Dense → weighted RRF 3:1 → ColBERT MaxSim rerank → cross-encoder rerank → channel dedup → top-N.

---

### Eval Dataset Schema (актуально 2026-03-20)

**Файлы**:
- `datasets/eval_dataset_quick.json` — Golden dataset v1 (10 вопросов)
- `datasets/eval_dataset_quick_v2.json` — Golden dataset v2 (10 вопросов, сложные)
- `datasets/eval_retrieval_100.json` — Retrieval eval (100 auto-generated queries)

```json
{
  "version": "1.0",
  "questions": [
    {
      "id": "q1",
      "question": "str — вопрос на русском",
      "expected_answer": "str — ожидаемый ответ",
      "expected_sources": [
        {"channel": "@channel_name", "message_id": 1234}
      ],
      "category": "factual|temporal|channel|comparative|multi_hop|entity|product|negative",
      "answerable": true
    }
  ]
}
```

**Распределение типов:**
- v1: factual ×3, temporal ×2, channel ×2, comparative ×1, multi_hop ×1, negative ×1
- v2: entity ×1, product ×3, fact_check ×1, cross_channel ×1, recency ×1, numeric ×1, long_tail ×1, negative ×1

**Eval matching**: fuzzy ±5 message_id для factual, ±50 для temporal/multi_hop.

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

### Settings Key Fields (актуально 2026-03-20)

| Переменная | Default | Описание |
|-----------|---------|---------|
| `LLM_MODEL_KEY` | `qwen3-30b-a3b` | Имя модели для llama-server |
| `LLM_BASE_URL` | `http://host.docker.internal:8080/v1` | URL llama-server |
| `EMBEDDING_MODEL_KEY` | `qwen3-embedding-0.6b` | Ключ embedding модели |
| `EMBEDDING_TEI_URL` | `http://host.docker.internal:8082` | URL gpu_server.py (embedding) |
| `RERANKER_TEI_URL` | `http://host.docker.internal:8082` | URL gpu_server.py (reranker) — тот же порт |
| `QDRANT_URL` | `http://qdrant:6333` | URL Qdrant в Docker compose |
| `QDRANT_COLLECTION` | `news_colbert` | Имя коллекции Qdrant |
| `COVERAGE_THRESHOLD` | `0.65` | Порог для refinement |
| `MAX_REFINEMENTS` | `2` | Макс. refinement раундов |
| `HYBRID_ENABLED` | `true` | Включить HybridRetriever |
| `ENABLE_RERANKER` | `true` | Включить BGE reranker |
| `AGENT_MAX_STEPS` | `15` | Максимум шагов ReAct |
