## src/core/settings.py — конфигурация приложения

### Назначение
- Централизованные настройки: модели (LLM/embedding/reranker), Chroma, кеши, гибрид/ммр/ререйк, BM25, параметры планировщика.

### Ключевые поля
- `current_llm_key`, `planner_llm_key`, `planner_llm_device` — выбор и устройство LLM планировщика.
- `current_embedding_key`, `current_collection` — активная embedding‑модель/коллекция.
- Redis: `redis_enabled`, `redis_host/port/password`, `cache_ttl`.
- Chroma: `chroma_host/port/path`.
- Директории моделей/кэша: `models_dir`, `cache_dir`.
- Query Planner/Fusion: `enable_query_planner`, `fusion_strategy`, `k_fusion`, `search_k_per_query_default`, `max_plan_subqueries`, `enable_cache`.
- BM25/Hybrid: `bm25_index_root`, `hybrid_enabled`, `hybrid_top_*`, `bm25_default_top_k`, `bm25_reload_min_interval_sec`.
- Reranker: `enable_reranker`, `reranker_model_key`, `reranker_top_n`, `reranker_batch_size`.

### Динамические обновления
- `update_llm_model`, `update_embedding_model`, `update_collection` — горячая смена с очисткой `lru_cache` фабрик.


