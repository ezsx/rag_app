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
  → QueryPlanner (Qwen3-8B, llama-server, GBNF grammar)
      → нормализованные подзапросы + MetadataFilters
  → HybridRetriever.search_with_plan()
      → TEI embed (multilingual-e5-large, prefix "query: ")
      → fastembed sparse (Qdrant/bm25, language="russian")
      → Qdrant prefetch:
          ├─ dense_vector: top-100
          └─ sparse_vector: top-100
      → FusionQuery(RRF) → top-N кандидатов
      → with_vectors=True (для coverage cosine_sim)
  → Reranker (TEI HTTP → bge-reranker-v2-m3, sigmoid scores)
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

- **multilingual-e5-large** — текущая модель (1024-dim, cosine)
- Через TEI HTTP (WSL2 native, RTX 5060 Ti, порт 8082)
- Prefix: `"query: "` для запросов, `"passage: "` для документов (добавляется TEIEmbeddingClient)
- **Будущее**: Qwen3-Embedding-0.6B (DEC-0026, approved, не реализовано)

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

## Hardware

- TEI embedding: RTX 5060 Ti (WSL2 native, порт 8082)
- TEI reranker: RTX 5060 Ti (WSL2 native, порт 8083)
- Qdrant: Docker CPU (порт 6333)
- GPU недоступна в Docker Desktop (DEC-0024: V100 TCC блокирует NVML)

## Типичные задачи

- Добавить фильтр по дате/каналу: `MetadataFilters` в `schemas/search.py`
- Сменить коллекцию: `settings.update_collection("new_name")`
- Проверить healthcheck TEI: `GET http://host.docker.internal:8082/health`
