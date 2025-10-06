# OpenAPI Contracts - Краткий обзор эндпоинтов

## QA API (основной)

### `POST /v1/qa`
- **Назначение**: Ответ на вопрос через RAG
- **Тело**: `{"query": "string", "include_context": bool, "collection": "string"}`
- **Ответ**: `{"answer": "string", "query": "string", "context?": [...], "context_count?": int}`

### `POST /v1/qa/stream`
- **Назначение**: Стриминговый ответ через SSE
- **Тело**: `{"query": "string", "include_context": bool, "collection": "string"}`
- **Ответ**: SSE поток токенов

## Search API

### `POST /v1/search`
- **Назначение**: Семантический поиск без генерации ответа
- **Тело**: `{"query": "string", "k": int, "collection": "string"}`
- **Ответ**: `{"documents": [...], "distances": [...], "metadatas": [...]}`

## Agent API (ReAct)

### `POST /v1/agent/stream`
- **Назначение**: Agentic ReAct-RAG с детерминированной логикой, coverage check и refinement циклами
- **Тело**: `{"query": "string", "collection": "string", "max_steps": int, "planner": bool, "tools_allowlist": ["string"]}`
- **Ответ**: SSE поток событий (step_started, thought, tool_invoked, observation, final)
- **Метаданные в финальном ответе**: `coverage`, `refinements`, `fallback`

### `GET /v1/agent/tools`
- **Назначение**: Список доступных инструментов агента
- **Ответ**: `{"tools": {...}, "total": int, "usage": "string"}`

### `GET /v1/agent/status`
- **Назначение**: Статус и конфигурация Agentic ReAct-RAG
- **Ответ**: `{"status": "active", "configuration": {...}, "features": {...}}`
- **Включает**: coverage_threshold, max_refinements, enable_verify_step, current_llm

## Collections API

### `GET /v1/collections`
- **Назначение**: Список доступных коллекций ChromaDB
- **Ответ**: `{"collections": [...], "current_collection": "string"}`

### `POST /v1/collections/select`
- **Назначение**: Переключение активной коллекции
- **Тело**: `{"collection_name": "string"}`
- **Ответ**: `{"collection_name": "string", "document_count": int}`

## Models API

### `GET /v1/models`
- **Назначение**: Список доступных моделей (LLM/Embedding)
- **Ответ**: `{"llm_models": [...], "embedding_models": [...], "current_llm": "string"}`

### `POST /v1/models/select`
- **Назначение**: Переключение модели
- **Тело**: `{"model_key": "string", "model_type": "llm|embedding"}`
- **Ответ**: `{"model_key": "string", "model_type": "string"}`

## Ingest API

### `POST /v1/ingest/telegram`
- **Назначение**: Запуск индексации Telegram каналов
- **Тело**: `{"channels": [...], "since": "date", "until": "date", "collection": "string"}`
- **Ответ**: `{"job_id": "string", "status": "queued", "estimated_time": "string"}`

### `GET /v1/ingest/status/{job_id}`
- **Назначение**: Статус задачи индексации
- **Ответ**: `{"job_id": "string", "status": "running", "progress": 0.5, "messages_processed": int}`

## System API

### `GET /v1/system/health`
- **Назначение**: Проверка работоспособности сервиса
- **Ответ**: `{"status": "healthy", "components": {...}, "version": "string"}`

### `GET /v1/system/metrics`
- **Назначение**: Метрики производительности
- **Ответ**: `{"memory_usage": {...}, "model_info": {...}, "cache_stats": {...}}`

## Схемы ошибок

### Стандартные HTTP коды
- `200` - Успех
- `422` - Ошибка валидации данных
- `500` - Внутренняя ошибка сервера
- `503` - Сервис недоступен (модели не загружены)

### Формат ошибок
```json
{
  "detail": "Описание ошибки",
  "error_code": "VALIDATION_ERROR|SERVER_ERROR|SERVICE_UNAVAILABLE"
}
```

## SSE События

### Для `/v1/qa/stream`
- `token`: токен текста от LLM
- `end`: завершение генерации ("[DONE]")
- `error`: ошибка в процессе генерации

### Для `/v1/agent/stream`
- `step_started`: начало нового шага мышления
- `thought`: размышление агента
- `tool_invoked`: вызов инструмента
- `observation`: результат инструмента
- `final`: финальный ответ агента
- `error`: ошибка в процессе выполнения
