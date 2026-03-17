# Agent Module — ReAct Агент

## Ключевые файлы

- `src/services/agent_service.py` — основной класс `AgentService`
- `src/services/tools/tool_runner.py` — `ToolRunner` реестр + запуск с таймаутом
- `src/services/tools/` — 7 инструментов агента
- `src/schemas/agent.py` — схемы: `AgentRequest`, `AgentStepEvent`, `ToolRequest`, `AgentAction`
- `src/core/deps.py` — DI: `get_agent_service`, `get_tool_runner`
- `src/api/v1/endpoints/agent.py` — SSE endpoint `/v1/agent/stream`
- `docs/ai/agent_technical_spec.md` — детальная спецификация

## Цикл ReAct (детерминированный)

```
Инициализация (AgentState: coverage=0.0, refinement_count=0)
    ↓
Цикл по шагам (до agent_max_steps=15):
    LLM.generate_step()
    parse(thought / action / final_answer)
    emit thought event
    if final_answer → verify → [refinement?] → emit final
    if action → ToolRunner.run() → emit tool_invoked → observation
        if tool == compose_context:
            check coverage < 0.8 AND refinement_count < 1
            → auto refinement (k *= 2, max 200)
        if tool == verify:
            check confidence < 0.6 → optional refinement
    ↓
Fallback (max_steps exceeded): QAService.answer()
```

## 7 инструментов

| Инструмент | Файл | Назначение |
|-----------|------|-----------|
| `router_select` | `tools/router_select.py` | Выбрать маршрут bm25/dense/hybrid |
| `query_plan` | `tools/query_plan.py` | Разложить запрос на под-запросы (GBNF grammar) |
| `search` | `tools/search.py` | Гибридный поиск (BM25 + Chroma, RRF fusion) |
| `rerank` | `tools/rerank.py` | BGE-reranker-v2-m3, top-N на CPU |
| `compose_context` | `tools/compose_context.py` | Собрать контекст, считает citation_coverage |
| `verify` | `tools/verify.py` | Проверить ответ, возвращает confidence + evidence |
| `final_answer` | `tools/final_answer.py` | Унифицировать финальный payload |

## SSE события (контракт `/v1/agent/stream`)

```
thought        — мысль агента (text)
tool_invoked   — {tool, input}
observation    — результат инструмента
citations      — из compose_context
final          — финальный ответ с метаданными
error          — при ошибке
```

## Настройки (settings.py)

```python
agent_max_steps       = 15
agent_default_steps   = 8
agent_tool_timeout    = 15.0
agent_token_budget    = 2000
agent_tool_temp       = 0.2    # шаги инструментов
agent_tool_max_tokens = 64
agent_final_temp      = 0.3    # финальный ответ
agent_final_max_tokens = 512
coverage_threshold    = 0.8
max_refinements       = 1
enable_verify_step    = True
```

## Типичные задачи

- Добавить новый инструмент: `ToolRunner.register(name, func, timeout_sec)` в `deps.py`
- Изменить coverage threshold: `settings.coverage_threshold`
- Поменять промпт: `AgentService.system_prompt` в `agent_service.py`
- Отладить шаг: JSON trace в stdout/логах (формат `req/step/tool/ok/took_ms`)
- Тесты: `src/tests/test_agent_service.py`
