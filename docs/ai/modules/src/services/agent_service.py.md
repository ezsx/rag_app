# AgentService - Enhanced ReAct агент с детерминированной логикой

## Обзор

`AgentService` — основной сервис для выполнения улучшенного ReAct алгоритма с детерминированной логикой проверки покрытия и верификации. Обеспечивает SSE стриминг каждого шага для наблюдаемости и поддерживает автоматические refinement раунды.

## Алгоритм работы

Агент реализует улучшенный ReAct алгоритм с детерминированной логикой:

1. **Инициализация:** Создаётся AgentState для отслеживания coverage и refinement_count
2. **Базовый поиск:** router_select → query_plan → search → compose_context
3. **Проверка покрытия:** Если citation_coverage < 0.8 → refinement раунд
4. **Refinement:** Дополнительный поиск с расширенными параметрами (hybrid_top_bm25 * 2)
5. **Верификация:** Перед FinalAnswer проверка через verify инструмент
6. **Итоговый ответ:** С метаданными (coverage, refinements, confidence)

## Класс AgentService

### Назначение
- Выполнение улучшенного ReAct цикла с детерминированной логикой
- Автоматическая проверка citation coverage (>= 80%)
- Детерминированные refinement раунды при недостаточном покрытии
- Верификация финального ответа через verify инструмент
- SSE стриминг всех шагов с метаданными
- Fallback через QAService при превышении лимита шагов

### Основные методы

#### `stream_agent_response(request: AgentRequest) -> AsyncIterator[AgentStepEvent]`
Основная ReAct петля с детерминированной логикой:
- Генерирует события: `step_started`, `thought`, `tool_invoked`, `observation`, `final`
- Парсит ответы LLM в формате ReAct
- Выполняет инструменты через ToolRunner
- Автоматически проверяет citation coverage после compose_context
- Выполняет refinement раунды при недостаточном покрытии
- Верифицирует финальный ответ через verify инструмент
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

#### `_verify_answer(final_answer, conversation_history) -> Dict[str, Any]`
Использует verify инструмент для проверки утверждений в финальном ответе.

#### `_should_attempt_refinement(coverage, refinement_count) -> bool`
Проверяет необходимость refinement раунда на основе покрытия и количества уже выполненных refinement.

#### `_perform_refinement(query, agent_state, request_id, step) -> Optional[AgentAction]`
Выполняет дополнительный поиск с расширенными параметрами для повышения покрытия.

## Системный промпт

Агент использует детальный системный промпт с:
- Описанием формата ReAct (Thought/Action/Observation/FinalAnswer)
- Списком доступных инструментов и их назначения, включая новый инструмент `search`
- Правилами работы и примерами

### Доступные инструменты

- **query_plan**: Создает план поиска для заданного запроса с фильтрами
- **search**: Выполняет гибридный поиск по коллекции с RRF слиянием
- **rerank**: Переранжирует документы по релевантности к запросу (опционально)
- **fetch_docs**: Получает документы по списку ID
- **compose_context**: Собирает контекст из последних результатов поиска (по `hit_ids`), добавляя цитаты и coverage
- **verify**: Проверяет утверждения через поиск в базе знаний с confidence
- **router_select**: Выбирает оптимальный маршрут поиска (bm25/dense/hybrid)

### AgentState класс

```python
class AgentState:
    def __init__(self):
        self.coverage: float = 0.0
        self.refinement_count: int = 0
        self.max_refinements: int = 1
```

### Новые настройки

- **COVERAGE_THRESHOLD=0.8**: Порог покрытия цитирований для принятия решения
- **MAX_REFINEMENTS=1**: Максимум дополнительных раундов поиска
- **ENABLE_VERIFY_STEP=true**: Включение верификации финального ответа

## Обработка ошибок

1. **Превышение max_steps**: Fallback через QAService
2. **Ошибки LLM**: Логирование и возврат сообщения об ошибке
3. **Ошибки инструментов**: Изоляция через ToolRunner, продолжение работы
4. **Невалидный JSON**: Обработка как raw_input параметр

## Конфигурация

Настройки через `Settings`:
- `agent_max_steps`: максимальное количество шагов (по умолчанию 15)
- `agent_default_steps`: значение по умолчанию для max_steps (по умолчанию 8)
- `agent_token_budget`: лимит токенов для LLM (по умолчанию 2000)
- `agent_tool_timeout`: таймаут инструментов (по умолчанию 5.0 сек)
- `coverage_threshold`: порог покрытия цитирований (по умолчанию 0.8)
- `max_refinements`: максимум refinement раундов (по умолчанию 1)
- `enable_verify_step`: включение верификации финального ответа (по умолчанию true)

## Интеграция

- Создается через `get_agent_service()` в `core.deps`
- Использует те же LLM и Retriever что и QAService
- Поддерживает горячее переключение коллекций
- Совместим с существующей архитектурой кеширования
