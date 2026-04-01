# SPEC-RAG-20c: agent_service.py Decomposition

> **Status:** Draft v2 (post Codex review)
> **Created:** 2026-03-31
> **Parent:** SPEC-RAG-20 Phase 3
> **Reviewed by:** Codex GPT-5.4 — "approve with changes", 8 findings applied
> **Context:** agent_service.py = 2397 строк, `stream_agent_response` = 853 строки. Невозможно дебажить, тестировать, ревьюить. Каждый fix добавляет ещё один if в монолит.

---

## Текущая карта файла (verified by Codex)

| Блок | Строки | Размер | Целевой модуль |
|------|--------|--------|----------------|
| `_load_routing_data/keywords/policy` | 36-78 | 43 | **routing.py** |
| `SYSTEM_PROMPT` | 80-131 | 52 | **prompts.py** |
| `AGENT_TOOLS` (15 JSON schemas) | 132-578 | 447 | **prompts.py** |
| `AgentState` | 579-596 | 18 | **state.py** |
| `RequestContext` + `_request_ctx` | 597-622 | 26 | **state.py** |
| `AgentService.__init__` | 626-639 | 14 | остаётся |
| `_ctx` property | 640-645 | 6 | остаётся |
| `stream_agent_response` | 646-1498 | **853** | **orchestrator (сжимаем)** |
| `_execute_action` | 1500-1587 | 88 | **executor.py** |
| `_normalize_tool_params` | 1588-1779 | 192 | **executor.py** |
| `_apply_action_state` | 1780-1855 | 76 | **state.py** |
| `_trim_refusal_alternatives` | 1856-1881 | 26 | **finalization.py** |
| `_format_observation` | 1882-1993 | 112 | **formatting.py** |
| `get_available_tools` | 1994-2019 | 26 | **visibility.py** |
| `_verify_answer` | 2020-2054 | 35 | **finalization.py** |
| `_should_attempt_refinement` | 2055-2063 | 9 | остаётся |
| `_perform_refinement` | 2064-2099 | 36 | остаётся |
| `_extract_tool_calls` | 2100-2154 | 55 | **formatting.py** |
| `_assistant_message_for_history` | 2155-2171 | 17 | **formatting.py** |
| `_tool_message_for_history` | 2172-2187 | 16 | **formatting.py** |
| `_serialize_tool_payload` | 2188-2218 | 31 | **formatting.py** |
| `_get_step_tools` | 2219-2295 | 77 | **visibility.py** |
| `_trim_messages` | 2296-2334 | 39 | **formatting.py** |
| `_tool_error_action` | 2335-2353 | 19 | **executor.py** |
| `_build_final_payload` | 2354-2396 | 43 | **finalization.py** |

---

## Целевая структура

```
src/services/
  agent_service.py              — AgentService class, stream_agent_response (orchestrator),
                                  _should_attempt_refinement, _perform_refinement
                                  (~600-800 строк)
  agent/
    __init__.py                 — ПУСТОЙ (avoid barrel re-export cycles)
    state.py                    — AgentState, RequestContext (с final_answer_text field),
                                  _request_ctx ContextVar, apply_action_state()
                                  (~120 строк)
    prompts.py                  — SYSTEM_PROMPT, AGENT_TOOLS
                                  (~500 строк)
    routing.py                  — _load_routing_data, _load_tool_keywords, _load_policy
                                  (~50 строк, fix __file__ path resolution)
    visibility.py               — get_step_tools(), get_available_tools()
                                  (~140 строк)
    executor.py                 — execute_action(), normalize_tool_params(),
                                  temporal guard, tool_error_action()
                                  (~300 строк)
    finalization.py             — build_final_payload(), trim_refusal_alternatives(),
                                  verify_answer()
                                  (~110 строк)
    formatting.py               — format_observation(), extract_tool_calls(),
                                  trim_messages(), message history helpers,
                                  serialize_tool_payload()
                                  (~200 строк, ТОЛЬКО чистые функции без state mutation)
```

### Ключевые решения (по Codex review):

1. **`executor.py`** вместо `params.py` — `_execute_action` + `_normalize_tool_params` + temporal guard + `_tool_error_action` живут вместе, потому что runtime-coupled (execute вызывает normalize, normalize для compose_context дёргает fetch_docs через tool_runner).

2. **`finalization.py`** вместо дампа в `formatting.py` — `_build_final_payload` мутирует request context (`ctx.final_answer_text`), `_trim_refusal_alternatives` — post-processing ответа, `_verify_answer` — финальная верификация. Это одна ответственность: "подготовка и проверка финального ответа".

3. **`formatting.py`** — ТОЛЬКО чистые serialization/history helpers без side effects. Ничего что пишет в ctx.

4. **`__init__.py` пустой** — все imports конкретные: `from services.agent.state import RequestContext`, не через package barrel.

5. **`RequestContext.final_answer_text`** — объявить как `Optional[str] = None` в dataclass, не ставить через `setattr`/`getattr`.

---

## Порядок выноса (инкрементальный, каждый шаг — отдельный коммит)

### Step 1: prompts.py (500 строк, **zero risk**)
- Вынести `SYSTEM_PROMPT` и `AGENT_TOOLS`
- Чистые константы, нет логики, нет зависимостей
- `from services.agent.prompts import SYSTEM_PROMPT, AGENT_TOOLS`
- **Проверка:** контейнер стартует, один запрос через curl

### Step 2: routing.py (50 строк, **low risk**)
- Вынести `_load_routing_data`, `_load_tool_keywords`, `_load_policy`, `_ROUTING_DATA`
- **⚠️ Risk:** `__file__` path resolution для `datasets/tool_keywords.json` — файл будет в `agent/routing.py`, путь вверх изменится
- **Fix:** использовать `Path(__file__).resolve().parents[3]` или передавать base_dir как параметр
- **Проверка:** контейнер стартует, keyword routing работает для entity_tracker/hot_topics

### Step 3: state.py (120 строк, **medium risk**)
- Вынести `AgentState`, `RequestContext`, `_request_ctx` ContextVar, `_apply_action_state`
- **⚠️ Risk:** async generator cleanup в другом context (`ValueError` на `_request_ctx.reset`). `final_answer_text` читается через `getattr` в finally.
- **Fix:** добавить `final_answer_text: Optional[str] = None` в `RequestContext` dataclass. Regression test для generator cleanup.
- **Проверка:** контейнер стартует, analytics_done/navigation_answered корректны, final answer в trace output

### Step 4: visibility.py (140 строк, **low risk**)
- Вынести `_get_step_tools` и `get_available_tools`
- Зависимости: `AgentState` (из state.py), routing.py (`_load_tool_keywords`, `_load_policy`)
- **Проверка:** visible_tools в SSE step_started корректны для PRE-SEARCH / POST-SEARCH / ANALYTICS-COMPLETE / NAV-COMPLETE

### Step 5: formatting.py — pure helpers only (200 строк, **low risk**)
- Вынести ТОЛЬКО чистые функции без state mutation:
  - `_format_observation`, `_extract_tool_calls`, `_trim_messages`
  - `_assistant_message_for_history`, `_tool_message_for_history`, `_serialize_tool_payload`
- **НЕ выносить** `_build_final_payload` (мутирует ctx) — он уйдёт в finalization.py
- **Проверка:** SSE events содержат те же данные, message history корректна

### Step 6: executor.py (300 строк, **medium risk**)
- Вынести `_execute_action`, `_normalize_tool_params`, temporal guard logic, `_tool_error_action`
- **⚠️ Risk:** `_normalize_tool_params` для compose_context вызывает fetch_docs через tool_runner. Temporal guard проверяет даты в params.
- **Проверка:** temporal_search, channel_search, entity_tracker params корректны. Codex review diff.

### Step 7: finalization.py (110 строк, **medium risk**)
- Вынести `_build_final_payload`, `_trim_refusal_alternatives`, `_verify_answer`
- **⚠️ Risk:** `_build_final_payload` пишет `ctx.final_answer_text`. `_verify_answer` вызывает hybrid retriever.
- **Проверка:** full eval 36 Qs. Codex review diff.

---

## Constraints

1. **SSE контракт (INV-01) не меняется** — клиенты (eval script, Web UI) не трогаем
2. **Один модуль за раз** — не выносить два модуля в одном коммите
3. **Smoke test после каждого step** — curl запрос, проверка SSE events
4. **Codex review** после step 3, 6, 7 (medium risk) — pipe diff
5. **Full eval** после step 7 — единственный полный прогон
6. **Не менять логику** — только перемещение кода, никаких "заодно пофикшу"
7. **Imports конкретные** — `from services.agent.state import AgentState`, не barrel re-export
8. **`__init__.py` пустой** — предотвращение circular import cycles

---

## stream_agent_response после декомпозиции

После всех 7 шагов `stream_agent_response` сжимается с 853 до ~450-600 строк.
Все yield'ы остаются внутри — это constraint async generator.

Orchestrator владеет:
- Request init (ctx, langfuse, system prompt) — ~30 строк
- Main loop (step iteration, LLM calls) — ~150 строк
- Forced search / analytics short-circuit / repeat guard — ~80 строк (loop-level guards)
- Tool execution dispatch + state update — ~60 строк
- Coverage/refinement — ~50 строк
- Final answer / verify / fallback / error — ~80 строк

Вызовы модулей заменяют inline логику: `get_step_tools()`, `execute_action()`, `format_observation()`, `build_final_payload()`, etc.

---

## Acceptance Criteria

- [ ] agent_service.py < 900 строк
- [ ] stream_agent_response < 600 строк
- [ ] Каждый модуль в agent/ < 500 строк
- [ ] SSE контракт не изменён (INV-01)
- [ ] Full eval 36 Qs — метрики не хуже текущих
- [ ] Codex review на step 3, 6, 7 — no critical findings
- [ ] `RequestContext.final_answer_text` объявлен как field, не через setattr
- [ ] Unit test: visibility.py phase transitions
- [ ] Unit test: executor.py temporal guard
- [ ] Regression test: async generator cleanup + ContextVar reset

---

## Риски (updated by Codex review)

| Риск | Severity | Митигация |
|------|----------|-----------|
| Guards вплетены в loop — extract сломает flow | High | Loop-level guards остаются в orchestrator. Только execution/finalization guards выносятся |
| Circular imports через barrel `__init__.py` | Medium | `__init__.py` пустой, imports конкретные |
| `__file__` path resolution в routing.py | Medium | Явный base_dir через parents или параметр |
| ContextVar async cleanup в другом context | Medium | `final_answer_text` в RequestContext dataclass. Regression test |
| ThreadPoolExecutor + ContextVar propagation | Medium | Helpers не читают `_request_ctx` из background thread. Regression test для compose_context/verify |
| `_build_final_payload` мутирует ctx | Low | Живёт в finalization.py, не в formatting.py. Explicit ctx parameter |
| Async generator yield из вызванных функций | Low | Модули возвращают данные/decisions, yield остаётся только в orchestrator |

---

## Не входит в scope

- Рефакторинг tool_runner.py
- Рефакторинг observability.py (отдельный SPEC-RAG-20b)
- Изменение SSE контракта
- Новая функциональность
- Изменение system prompt
- Extraction loop-level guards из stream_agent_response (слишком рискованно, оставляем inline)
