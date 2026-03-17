# Retrieval Module — RAG Pipeline (Phase 1)

## Ключевые файлы

- `src/adapters/search/hybrid_retriever.py` — `HybridRetriever` (Qdrant prefetch + FusionQuery(RRF) + MmrQuery)
- `src/adapters/qdrant/store.py` — `QdrantStore` (dense+sparse, named vectors)
- `src/adapters/tei/embedding_client.py` — `TEIEmbeddingClient` (async httpx → TEI :8082)
- `src/adapters/tei/reranker_client.py` — `TEIRerankerClient` (async httpx → TEI :8083)
- `src/services/query_planner_service.py` — GBNF-based планировщик подзапросов
- `src/services/reranker_service.py` — sync bridge над async TEIRerankerClient
- `src/utils/ranking.py` — `rrf_merge`, `mmr_select` (используются в qa_service для legacy ветки)
- `src/core/settings.py` — все гиперпараметры поиска
- `src/core/deps.py` — DI: `get_hybrid_retriever`, `get_reranker`, `get_query_planner`

## Pipeline (поиск в агенте)

```
query
  → QueryPlanner (Qwen3-30B-A3B, llama-server, GBNF grammar)
      → нормализованные подзапросы + MetadataFilters
  → HybridRetriever.search_with_plan()
      → TEI embed (Qwen3-Embedding-0.6B, instruction prefix для query)
      → fastembed sparse (Qdrant/bm25, language="russian")
      → Qdrant prefetch:
          ├─ dense_vector: top-100
          └─ sparse_vector: top-100
      → FusionQuery(RRF) → top-N кандидатов
      → with_vectors=True (для coverage cosine_sim)
  → Reranker (TEI HTTP → Qwen3-Reranker-0.6B-seq-cls, sigmoid scores)
  → compose_context → coverage check (0.65, DEC-0019)
```

## Ключевые настройки

```python
# QueryPlanner
use_gbnf_planner      = True
planner_timeout       = 15.0
max_plan_subqueries   = 5

# Hybrid search (Qdrant)
hybrid_enabled        = True
hybrid_top_dense      = 100    # prefetch limit для dense
hybrid_top_sparse     = 100    # prefetch limit для sparse
k_fusion              = 60     # RRF k-параметр

# MMR (нативно через Qdrant MmrQuery)
enable_mmr            = True
mmr_lambda            = 0.7    # релевантность vs разнообразие
mmr_top_n             = 120
mmr_output_k          = 60

# Reranker (TEI HTTP)
enable_reranker       = True
reranker_tei_url      = "http://host.docker.internal:8083"
reranker_top_n        = 80
reranker_batch_size   = 16     # ignored — TEI управляет батчингом
```

## Embedding

- **Qwen3-Embedding-0.6B** — текущая модель (1024-dim, cosine, long context)
- Через TEI HTTP (WSL2 native, RTX 5060 Ti, порт 8082)
- Query format: `Instruct: Given a user question about ML, AI, LLM or tech news, retrieve relevant Telegram channel posts\nQuery: ...`
- Documents идут без prefix

## Qdrant

- URL: `QDRANT_URL` (default: `http://qdrant:6333` в Docker)
- Named vectors: `dense_vector` (cosine, 1024-dim) + `sparse_vector` (Qdrant/bm25, russian)
- Коллекция: `settings.qdrant_collection` (hot-swap через `settings.update_collection()`)
- Default: `news`
- Storage: named volume `qdrant_data` (bind mounts → silent corruption на Windows)

## Sparse Encoding

- fastembed `SparseTextEmbedding` (модель: `Qdrant/bm25`, language="russian", CPU)
- Используется в HybridRetriever для sparse ветки prefetch
- `embed()` для индексации, `query_embed()` для поиска (разные BM25 веса)

## Reranker

- **Qwen3-Reranker-0.6B-seq-cls** — текущая TEI-compatible модель
- Основана на `Qwen3-Reranker-0.6B`, но обёрнута в seq-cls формат для TEI
- Используется после search и перед compose_context

## Chunking при ingest

- Посты `<1500` символов индексируются одним point
- Посты `>1500` символов режутся через `_smart_chunk()`
- Recursive split по `["\n\n", "\n", ". ", " "]`
- Target size чанка: `1200` символов
- Без overlap
- Point ID при chunking: `{channel}:{message_id}:{chunk_idx}`

## Hardware

- TEI embedding: RTX 5060 Ti (WSL2 native, порт 8082)
- TEI reranker: RTX 5060 Ti (WSL2 native, порт 8083)
- Qdrant: Docker CPU (порт 6333)
- GPU недоступна в Docker Desktop (DEC-0024: V100 TCC блокирует NVML)

## Типичные задачи

- Добавить фильтр по дате/каналу: `MetadataFilters` в `schemas/search.py`
- Сменить коллекцию: `settings.update_collection("new_name")`
- Проверить healthcheck TEI: `GET http://host.docker.internal:8082/health`
