# AgentService

`src/services/agent_service.py` реализует агентский цикл поверх native function calling.

## Что делает модуль

- вызывает LLM через `/v1/chat/completions`
- передаёт модели tools schema из 5 функций
- исполняет `tool_calls` через `ToolRunner`
- стримит SSE события `thought`, `tool_invoked`, `observation`, `citations`, `final`
- держит runtime state для coverage, refinements, verify и disclaimer

## Tools schema

LLM-visible инструменты:

1. `query_plan`
2. `search`
3. `rerank`
4. `compose_context`
5. `final_answer`

Системные инструменты:

- `fetch_docs` — догрузка документов внутри `_normalize_tool_params("compose_context", ...)`
- `verify` — проверка ответа после `final_answer`

## Main loop

`stream_agent_response()`:

1. создаёт `messages = [system, user]`
2. вызывает `llm.chat_completion(...)`
3. при `message.content` эмитит `thought`
4. при `tool_calls`:
   - публикует `tool_invoked`
   - выполняет инструмент
   - публикует `observation`
   - добавляет `role="tool"` message обратно в историю
5. после `compose_context` публикует `citations`
6. после `final_answer` выполняет `verify` и публикует `final`

## Нормализация параметров

`_normalize_tool_params()` выполняет системную адаптацию:

- `query_plan`: инжектирует текущий query
- `search`: заполняет `queries` из planner summary или user query
- `rerank`: заполняет `docs` из `_last_search_hits`
- `compose_context`:
  - инжектирует `query`
  - строит `docs` из `_last_search_hits`
  - прокидывает `dense_score`
  - при необходимости догружает тексты через `fetch_docs`
- `verify`: переводит `k -> top_k`

## Guardrails

### Coverage

После `compose_context` агент читает `citation_coverage` и сохраняет его в `AgentState.coverage`.

### Abort guard

Если `max(_last_search_hits[*].dense_score) < 0.30` и refinement ещё не выполнялся:

- refinement не запускается
- coverage сбрасывается к `0.0`
- выставляется `low_coverage_disclaimer`
- агент публикует системный `thought`

### Refinement

Если coverage ниже `settings.coverage_threshold`, агент вызывает `_perform_refinement()`:

- `search`
- затем `compose_context`

Оба результата публикуются как обычные `tool_invoked`/`observation`, но с системными флагами.

### Verify

После `final_answer` агент вызывает `_verify_answer()`.
Если verify не проходит и refinement ещё доступен, запускается ещё один системный refinement search.

## Final payload

`_build_final_payload()` добавляет к ответу:

- `citations`
- `coverage`
- `refinements`
- `route`
- `plan`
- `verification`
- low-coverage disclaimer, если он был выставлен

## Что принципиально удалено

- regex-парсинг `Thought/Action/FinalAnswer`
- `_generate_step()`
- `_parse_llm_response()`
- forced final answer
- strip thinking / CJK logit-bias hacks
