### Модуль: `src/core/deps.py`

Назначение: DI-фабрики Phase 1. Все компоненты через `@lru_cache` — синглтоны на процесс.
Смена настроек требует `cache_clear()` через `settings.update_*()`.

#### Ключевые фабрики
- `get_llm()` → `LlamaServerClient` (HTTP к llama-server.exe, Qwen3-8B).
- `get_planner_llm()` → `LlamaServerClient` для QueryPlanner (отдельный endpoint или fallback на основной).
- `get_query_planner()` → `QueryPlannerService`.
- `get_tei_embedding_client()` → `TEIEmbeddingClient` (async httpx → TEI :8082).
- `get_tei_reranker_client()` → `TEIRerankerClient` (async httpx → TEI :8083).
- `get_qdrant_store()` → `QdrantStore` (dense+sparse, named vectors).
- `get_sparse_encoder()` → `SparseTextEmbedding` (fastembed, Qdrant/bm25, language="russian", CPU).
- `get_hybrid_retriever()` → `HybridRetriever` (Qdrant prefetch+FusionQuery(RRF)+MmrQuery).
- `get_retriever()` → backward-compatible алиас для `get_hybrid_retriever()`.
- `get_reranker()` → `RerankerService` (sync bridge over TEIRerankerClient).
- `get_qa_service()` → `QAService`.
- `get_redis_client()` → Redis (опционально).
- `get_agent_service()` → `AgentService` с полным набором 8 инструментов.

#### Удалено в Phase 1
- `get_chroma_client()` — ChromaDB удалён.
- `get_bm25_index_manager()` — BM25IndexManager удалён.
- `release_llm_vram_temporarily()` — llama-server.exe управляет VRAM сам.

#### Внешние зависимости
- `qdrant-client`, `httpx`, `fastembed`, `redis` (опц.), FastAPI `Depends`.
