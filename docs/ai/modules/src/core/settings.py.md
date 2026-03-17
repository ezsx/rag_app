## src/core/settings.py — конфигурация приложения (Phase 1)

### Назначение
Централизованные настройки Phase 1: LLM (llama-server HTTP), embedding (TEI HTTP),
reranker (TEI HTTP), Qdrant, кеши, гибрид/MMR/ререйк, параметры планировщика и агента.

### Ключевые поля
- **LLM**: `current_llm_key`, `llm_base_url`, `llm_model_name`, `llm_request_timeout`.
- **Planner**: `planner_llm_base_url`, `planner_llm_key`.
- **Embedding (TEI HTTP)**: `current_embedding_key`, `embedding_tei_url`.
- **Reranker (TEI HTTP)**: `reranker_tei_url`, `enable_reranker`, `reranker_top_n`, `reranker_batch_size`.
- **Qdrant**: `qdrant_url`, `qdrant_collection`, `current_collection` (алиас).
- **Redis**: `redis_enabled`, `redis_host/port/password`, `cache_ttl`.
- **Query Planner/Fusion**: `enable_query_planner`, `fusion_strategy`, `k_fusion`, `search_k_per_query_default`, `max_plan_subqueries`, `planner_timeout`, GBNF параметры.
- **Hybrid Retriever**: `hybrid_enabled`, `hybrid_top_dense`, `hybrid_top_sparse`.
- **MMR**: `enable_mmr`, `mmr_lambda`, `mmr_top_n`, `mmr_output_k`.
- **ReAct Agent**: `enable_agent`, `agent_max_steps`, `agent_default_steps`, `agent_tool_timeout`, `agent_token_budget`, декодинг для tool-шагов и финального ответа.
- **Coverage (DEC-0019)**: `coverage_threshold=0.65`, `max_refinements=2`, `enable_verify_step`.

### Удалено в Phase 1
- `chroma_host/port/path`, `bm25_index_root`, `reranker_model_key`, `models_dir`, `cache_dir`.

### Динамические обновления
- `update_llm_model()`, `update_embedding_model()`, `update_collection()` — горячая смена с очисткой `lru_cache` фабрик.
