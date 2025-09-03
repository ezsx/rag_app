### Модуль: `src/api/v1/endpoints/qa.py`

Назначение: REST эндпоинты для QA: синхронный ответ и SSE‑стрим.

#### Эндпоинты
- `POST /v1/qa` → `QAResponse | QAResponseWithContext`
  - Вход: `QARequest { query, include_context?, collection? }`
  - Кеш Redis по ключу `qa:<hash>` при `settings.redis_enabled`.
  - Опциональное временное переключение `Settings.current_collection`.
- `POST /v1/qa/stream` → SSE
  - Поток токенов от `QAService.stream_answer` до события `end`.

#### Зависимости
- `get_qa_service`, `get_redis_client`, `get_retriever`, `get_query_planner`, `get_reranker`, `get_settings`.





