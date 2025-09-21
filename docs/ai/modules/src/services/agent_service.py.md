# AgentService - ReAct агент с пошаговым мышлением

## Обзор

`AgentService` — основной сервис для выполнения ReAct (Reasoning and Acting) циклов с пошаговым мышлением и использованием инструментов. Обеспечивает SSE стриминг каждого шага для наблюдаемости.

## Класс AgentService

### Назначение
- Выполнение ReAct петли: Thought → Action → Observation → (повтор) → FinalAnswer
- SSE стриминг всех шагов мышления и действий агента
- Управление инструментами через ToolRunner
- Fallback через QAService при превышении лимита шагов

### Основные методы

#### `stream_agent_response(request: AgentRequest) -> AsyncIterator[AgentStepEvent]`
Основная ReAct петля с SSE стримингом:
- Генерирует события: `step_started`, `thought`, `tool_invoked`, `observation`, `final`
- Парсит ответы LLM в формате ReAct
- Выполняет инструменты через ToolRunner
- Обрабатывает ошибки и fallback-сценарии

#### `get_available_tools() -> Dict[str, Any]`
Возвращает схемы всех доступных инструментов с описаниями и параметрами.

### Внутренние методы

#### `_generate_step(conversation_history, request_id, step) -> str`
Генерирует следующий шаг через LLM с системным промптом ReAct.

#### `_parse_llm_response(response: str) -> tuple[Optional[str], Optional[str], Optional[str]]`
Парсит ответ LLM на компоненты: `thought`, `action`, `final_answer`.

#### `_execute_action(action_text: str, request_id: str, step: int) -> Optional[AgentAction]`
Выполняет действие (вызов инструмента):
- Парсит строку действия: `tool_name {json_params}`
- Создает ToolRequest и выполняет через ToolRunner
- Обрабатывает невалидный JSON с fallback в raw_input

#### `_format_observation(tool_response) -> str`
Форматирует результат инструмента для наблюдения в читаемом виде.

## Системный промпт

Агент использует детальный системный промпт с:
- Описанием формата ReAct (Thought/Action/Observation/FinalAnswer)
- Списком доступных инструментов и их назначения
- Правилами работы и примерами

## Обработка ошибок

1. **Превышение max_steps**: Fallback через QAService
2. **Ошибки LLM**: Логирование и возврат сообщения об ошибке
3. **Ошибки инструментов**: Изоляция через ToolRunner, продолжение работы
4. **Невалидный JSON**: Обработка как raw_input параметр

## Конфигурация

Настройки через `Settings`:
- `agent_max_steps`: максимальное количество шагов (по умолчанию 4)
- `agent_token_budget`: лимит токенов для LLM (по умолчанию 2000)
- `agent_tool_timeout`: таймаут инструментов (по умолчанию 5.0 сек)

## Интеграция

- Создается через `get_agent_service()` в `core.deps`
- Использует те же LLM и Retriever что и QAService
- Поддерживает горячее переключение коллекций
- Совместим с существующей архитектурой кеширования
