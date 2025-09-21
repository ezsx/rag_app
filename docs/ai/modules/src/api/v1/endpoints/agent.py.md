# Agent API Endpoints - ReAct агент через REST API

## Обзор

API эндпоинты для работы с ReAct агентом. Предоставляют пошаговый SSE стриминг выполнения агента и управление инструментами.

## Эндпоинты

### `POST /v1/agent/stream`

**Назначение**: Пошаговый ReAct агент с SSE стримингом

**Параметры запроса (AgentRequest)**:
- `query` (str): Вопрос пользователя (обязательно, 1-1000 символов)
- `collection` (str, optional): Название коллекции ChromaDB
- `model_profile` (str, optional): Профиль модели
- `tools_allowlist` (List[str], optional): Разрешенные инструменты
- `planner` (bool): Использовать ли планировщик запросов (по умолчанию true)
- `max_steps` (int): Максимальное количество шагов (1-10, по умолчанию 4)

**SSE События**:
```typescript
// Начало нового шага
{
  "event": "step_started",
  "data": {"step": 1, "request_id": "uuid", "max_steps": 4, "query": "..."}
}

// Мысль агента
{
  "event": "thought", 
  "data": {"content": "Мне нужно найти информацию о...", "step": 1}
}

// Вызов инструмента
{
  "event": "tool_invoked",
  "data": {"tool": "router_select", "input": {...}, "step": 1}
}

// Результат инструмента
{
  "event": "observation",
  "data": {"content": "route: dense", "success": true, "step": 1, "took_ms": 150}
}

// Финальный ответ
{
  "event": "final",
  "data": {"answer": "Ответ агента", "step": 3, "total_steps": 3, "request_id": "uuid"}
}

// Ошибка
{
  "event": "error",
  "data": {"error": "Описание ошибки"}
}
```

**Особенности**:
- Поддержка горячего переключения коллекций
- Автоматическое отключение при разрыве соединения клиента
- Fallback через QAService при превышении max_steps
- Обработка ошибок с корректным завершением стрима

### `GET /v1/agent/tools`

**Назначение**: Список доступных инструментов агента

**Ответ**:
```json
{
  "tools": {
    "router_select": {
      "description": "Выбирает оптимальный маршрут поиска",
      "parameters": {"query": "string"}
    },
    "math_eval": {
      "description": "Вычисляет математические выражения",
      "parameters": {"expression": "string"}
    },
    // ... другие инструменты
  },
  "total": 7,
  "usage": "Используйте формат: Action: tool_name {\"param\": \"value\"}",
  "supported_formats": {
    "json": "Параметры передаются в формате JSON",
    "example": "Action: math_eval {\"expression\": \"2 + 2\"}"
  }
}
```

### `GET /v1/agent/status`

**Назначение**: Статус и конфигурация агента

**Ответ**:
```json
{
  "status": "active",
  "configuration": {
    "max_steps": 4,
    "token_budget": 2000,
    "tool_timeout": 5.0,
    "current_collection": "news_demo4",
    "current_llm": "gpt-oss-20b",
    "enable_query_planner": true
  },
  "features": {
    "react_reasoning": true,
    "sse_streaming": true,
    "tool_execution": true,
    "fallback_qa": true
  }
}
```

## Обработка ошибок

- **500 Internal Server Error**: Ошибки агента или инструментов
- **422 Unprocessable Entity**: Ошибки валидации запроса
- **503 Service Unavailable**: Недоступность зависимостей (LLM, ChromaDB)

## Совместимость

- Полностью совместим с существующими API `/v1/qa/*`
- Использует те же зависимости и настройки
- Поддерживает те же коллекции и модели
- Следует тем же паттернам обработки ошибок и логирования
