# API Module — FastAPI Layer (Phase 1)

## Ключевые файлы

- `src/main.py` — точка входа FastAPI приложения
- `src/api/v1/router.py` — главный роутер v1
- `src/api/v1/endpoints/` — эндпоинты
- `src/core/deps.py` — DI фабрики (lru_cache)
- `src/core/auth.py` — JWT аутентификация
- `src/core/rate_limit.py` — rate limiting с exponential backoff
- `src/core/security.py` — SecurityManager, sanitize_for_logging
- `src/schemas/` — Pydantic схемы

## Эндпоинты

| Путь | Файл | Описание |
|------|------|---------|
| `POST /v1/agent/stream` | `endpoints/agent.py` | SSE ReAct стриминг |
| `POST /v1/qa` | `endpoints/qa.py` | Baseline QA (без агента) |
| `POST /v1/search` | `endpoints/search.py` | Прямой поиск |
| `POST /v1/ingest` | `endpoints/ingest.py` | Загрузка документов |
| `GET /v1/models` | `endpoints/models.py` | Список/смена моделей |
| `GET /v1/health` | `endpoints/system.py` | Healthcheck, статус |
| `POST /v1/auth/token` | `endpoints/auth.py` | Получение JWT токена |

> **Примечание**: `collections.py` отключён (Phase 0 ChromaDB код, ожидает SPEC-RAG-06).

## DI (deps.py) — lru_cache фабрики

```python
get_settings()              # Settings singleton
get_llm()                   # LlamaServerClient (Qwen3-8B, HTTP)
get_tei_embedding_client()  # TEIEmbeddingClient (async httpx → :8082)
get_tei_reranker_client()   # TEIRerankerClient (async httpx → :8083)
get_qdrant_store()          # QdrantStore (dense+sparse)
get_sparse_encoder()        # SparseTextEmbedding (fastembed, CPU)
get_hybrid_retriever()      # HybridRetriever (Qdrant RRF)
get_retriever()             # backward-compat алиас
get_reranker()              # RerankerService (sync bridge)
get_query_planner()         # QueryPlannerService
get_qa_service()            # QAService
get_agent_service()         # AgentService + ToolRunner с 8 инструментами
get_redis_client()          # Redis (опционально)
```

**ВАЖНО**: при смене модели/коллекции вызывать `cache_clear()` через `settings.update_*()`.

## SSE стриминг (`/v1/agent/stream`)

- Content-Type: `text/event-stream`
- Timeout соединения: 60 сек
- JWT обязателен (Bearer token в Authorization header)
- Формат событий: `data: {json}\n\n`
- Типы событий: `thought`, `tool_invoked`, `observation`, `citations`, `final`, `error`

## Запуск

```bash
# Docker (рекомендуется)
docker compose -f deploy/compose/compose.dev.yml up
# API: http://localhost:8000
# Qdrant: http://localhost:6333

# Перед этим запустить в WSL2:
# TEI embedding :8082 + TEI reranker :8083
# И llama-server.exe на Windows Host :8080
```

## Security

- JWT токен через `POST /v1/auth/token` (admin роль для чувствительных роутов)
- `SecurityManager` фильтрует prompt injection и PII во входных запросах
- `sanitize_for_logging` — для всего что идёт в логи
- CORS настроен в `main.py`
