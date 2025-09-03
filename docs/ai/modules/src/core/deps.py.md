### Модуль: `src/core/deps.py`

Назначение: фабрики зависимостей и ресурсоёмких компонентов с ленивой инициализацией (`@lru_cache`). Управление клиентами ChromaDB, LLM (`llama_cpp`), Query Planner, BM25, гибридным ретривером, ререйкером и Redis. Есть контекст‑менеджер для временного освобождения VRAM.

#### Ключевые фабрики
- `get_chroma_client()` → `chromadb.HttpClient` c fallback на `PersistentClient`.
- `get_retriever()` → `adapters.chroma.Retriever` с автоскачиванием embedding‑модели при необходимости.
- `get_llm()` → `llama_cpp.Llama` (автоскачивание GGUF через `utils.model_downloader.auto_download_models`).
- `release_llm_vram_temporarily()` → async contextmanager, очищает KV‑кеш и освобождает ресурсы.
- `get_planner_llm()` → CPU‑LLM для планировщика (обычно Qwen2.5‑3B, n_gpu_layers=0).
- `get_query_planner()` → `QueryPlannerService` (использует planner LLM или fallback на основную LLM).
- `get_bm25_index_manager()` → `BM25IndexManager`.
- `get_qa_service()` → `QAService` (ленивая LLM‑фабрика, опционально planner/reranker/hybrid).
- `get_hybrid_retriever()` → опциональный `HybridRetriever` (BM25 + dense), если включено.
- `get_reranker()` → `RerankerService` (CPU CrossEncoder), с автоскачиванием.
- `get_redis_client()` → Redis при включённом кешировании.

#### Конфигурация и инварианты
- Все фабрики мемоизируются `@lru_cache` и пересоздаются при изменении `Settings` (сброс кешей в `settings.update_*`).
- Подробное логирование путей и параметров LLM, GGUF‑хедера, размера файла.
- Chroma: попытка HTTP клиента, при ошибке — локальный `PersistentClient`.

#### Внешние зависимости
- `chromadb`, `llama_cpp`, `redis` (опц.), `numpy`, FastAPI `Depends` для `get_settings`.





