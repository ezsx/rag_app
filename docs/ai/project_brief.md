## Обзор проекта

`rag_app` — FastAPI-платформа Retrieval-Augmented Generation с агентским ReAct-пайплайном.
Поисковик/агрегатор новостей из Telegram-каналов: ingest → Qdrant (dense+sparse) → Hybrid Retrieval → ReAct Agent → SSE-ответ.
Каноническая архитектурная схема: `docs/architecture/04-system/overview.md`.

### Архитектура (Phase 1)
1. **Vector Store**: Qdrant — dense 1024-dim cosine + sparse BM25 (russian), named vectors, native RRF+MMR.
2. **LLM**: Qwen3-8B GGUF через llama-server.exe (Windows Host, V100 SXM2 32GB, порт 8080). Thinking mode отключён (`/no_think`, DEC-0022).
3. **Embedding**: multilingual-e5-large через TEI HTTP (WSL2 native, RTX 5060 Ti, порт 8082). Будущее: Qwen3-Embedding-0.6B (DEC-0026, approved, не реализовано).
4. **Reranker**: bge-reranker-v2-m3 через TEI HTTP (WSL2 native, RTX 5060 Ti, порт 8083).
5. **Sparse**: fastembed SparseTextEmbedding (Qdrant/bm25, language="russian", CPU).
6. **Docker**: api + qdrant — CPU only. GPU недоступна в Docker Desktop (DEC-0024: V100 TCC блокирует NVML).

### Сервисы
- `qa_service` — QA через HybridRetriever (Qdrant) + LLM.
- `agent_service` — ReAct-цикл, оркестрация 7+1 инструментов, coverage/refinements.
- `reranker_service` — синхронный мост над async TEIRerankerClient.
- `query_planner_service` — GBNF-грамматика, декомпозиция подзапросов.

### Адаптеры
- `src/adapters/qdrant/store.py` — QdrantStore (dense+sparse upsert, query, RRF).
- `src/adapters/search/hybrid_retriever.py` — Qdrant prefetch + FusionQuery(RRF) + MmrQuery.
- `src/adapters/tei/embedding_client.py` — TEIEmbeddingClient (async httpx).
- `src/adapters/tei/reranker_client.py` — TEIRerankerClient (async httpx).
- `src/adapters/llm/llama_server_client.py` — LlamaServerClient.

### DI и конфигурация
- Все фабрики через `@lru_cache` в `src/core/deps.py`. Hot-swap: `settings.update_*()` + `cache_clear()`.
- API стартует при недоступных зависимостях (ошибки логируются, warmup опционален).

### ReAct Agent
- **7+1 инструментов**: router_select, query_plan, search, rerank, compose_context, verify, final_answer (+ fetch_docs).
- Coverage threshold: **0.65** (DEC-0019, composite 5-signal metric в SPEC-RAG-07).
- Max refinements: **2** (DEC-0019).
- SSE-события: `thought`, `tool_invoked`, `observation`, `citations`, `final`.
- `compose_context` считает citation coverage, предотвращает lost-in-the-middle.
- `verify` возвращает confidence и evidence-ссылку.

### API
- REST `/v1/**`: QA, поиск, ReAct-агент.
- SSE `/v1/agent/stream` — долгие соединения (таймаут 60 с).
- `/v1/qa` — baseline без ReAct.

### Безопасность
- JWT + роль админа, `RateLimitMiddleware` с экспоненциальным бэкоффом.
- `SecurityManager`, `sanitize_for_logging`, фильтр prompt-injection, блокировка PII.
- CORS, проверка прав на SSE-эндпоинтах.

### Evaluation
- `scripts/evaluate_agent.py` — CLI, вызывает `/v1/agent/stream` и `/v1/qa`.
- `datasets/eval_dataset.json` — тестовый датасет (id/query/category/expected_documents/answerable).
- Метрики: latency, coverage, recall@5, agent_steps.
- Флаги: `--dry-run`, `--limit`, `--collection`, `--skip-markdown`, `--api-key`.
- Отчёты: per-query JSON в `results/raw`, агрегаты + Markdown в `results/reports`.

### Незавершённые работы Phase 1
- **SPEC-RAG-06**: миграция ingest pipeline (ingest_telegram.py ещё Phase 0).
- **SPEC-RAG-07**: composite coverage metric (сейчас naive doc count ratio, цель — 5-signal weighted).
- `collections.py` endpoint отключён (Phase 0, ChromaDB-код).
- Phase 2 planned: LLM-judge correctness/faithfulness, Citation Precision, conciseness.
