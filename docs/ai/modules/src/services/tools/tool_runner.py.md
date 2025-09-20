### Модуль `src/services/tools/tool_runner.py`

Назначение: реестр инструментов и единая точка запуска с таймаутами и JSON‑трассировкой шагов.

Ключевые элементы:
- `ToolRunner` — регистрация `name -> func`, запуск через `run(request_id, step, ToolRequest)`.
- Таймаут по умолчанию 5s, измерение `took_ms`, формирование `ToolResponse`.
- Логирование каждого шага в stdout и `logging` в формате JSON: `{req, step, tool, ok, took_ms, error?}`.

Контракты (см. `src/schemas/agent.py`): `ToolRequest`, `ToolResponse`, `ToolMeta`, `AgentAction`.


