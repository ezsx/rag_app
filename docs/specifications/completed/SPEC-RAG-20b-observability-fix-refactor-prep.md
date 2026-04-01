# SPEC-RAG-20b: Observability Fix + Refactor Prep

> **Status:** Draft
> **Created:** 2026-03-31
> **Parent:** SPEC-RAG-20 Phase 3 (prerequisite)
> **Context:** Langfuse traces содержат только скелет (spans, latency) но не содержание (query, answer, tool data, token usage). Без этого невозможно верифицировать рефакторинг agent_service.py — нет способа проверить что ответы не деградировали.

---

## Проблема (из реального trace)

| Что | Текущее состояние | Ожидание |
|-----|------------------|----------|
| `trace.input` | null | user query |
| `trace.output` | `{steps, coverage, search_count, analytics_done}` | + answer text |
| `llm_step_N` spans | пустые input/output | messages → response |
| `tool:*` spans | input передаётся, output передаётся | OK (tool_runner.py уже делает) |
| `llm_chat_completion` generation | input=messages, output=message, usage ✓ | OK (llama_server_client.py уже делает) |
| `hybrid_retrieval` span | пустые | query → hits count |
| `tool[system]:verify` span | пустые | query → verified/not |

**Ключевой вывод**: llama_server_client.py и tool_runner.py **уже правильно** передают input/output/usage. Проблема в agent_service.py — `observe_trace` и `observe_span` вызываются **без данных**.

---

## 7 Instrumentation Points — текущий статус

| # | Файл | Span name | Input | Output | Usage | Status |
|---|------|-----------|-------|--------|-------|--------|
| 1 | agent_service.py:673 | `observe_trace("agent_request")` | tags, session_id | steps/coverage only | — | **BROKEN** |
| 2 | agent_service.py:812 | `observe_span("llm_step_N")` | metadata only | — | — | **BROKEN** |
| 3 | llama_server_client.py:87 | `observe_llm_call("llm_completion")` | prompt[:500] | text[:500], usage ✓ | ✓ | OK |
| 4 | llama_server_client.py:152 | `observe_llm_call("llm_chat_completion")` | messages | message, usage ✓ | ✓ | OK |
| 5 | tool_runner.py:100 | `observe_span("tool:*")` | req.input | {ok, data_keys, took_ms} | — | **PARTIAL** (output truncated) |
| 6 | hybrid_retriever.py:88 | `observe_span("hybrid_retrieval")` | ? | ? | — | **CHECK** |
| 7 | reranker_client.py:72 | `observe_span("reranker")` | ? | ? | — | **CHECK** |

---

## Fixes

### Fix 1: trace root — query as input, answer as output

**agent_service.py:673** — добавить `input_data={"query": request.query}`

```python
_root_trace_cm = observe_trace(
    name=request.trace_name or "agent_request",
    session_id=request.session_id,
    tags=request.tags,
    input_data={"query": request.query},  # NEW
)
```

**agent_service.py:~1450** — добавить answer в output update:

```python
_root_span.update(output={
    "steps": step,
    "coverage": ctx.coverage_score,
    "search_count": ctx.agent_state.search_count,
    "analytics_done": ctx.agent_state.analytics_done,
    "answer": <final_answer_text>,  # NEW
})
```

### Fix 2: llm_step spans — pass input/output через

**agent_service.py:812** — `observe_span` не получает messages. Но вложенный `observe_llm_call` в llama_server_client.py уже получает их. Проблема: `observe_span("llm_step_N")` — это **wrapper span**, его input/output должны зеркалить child generation.

Простой fix: после `response = llm.chat_completion(...)`, обновить span:

```python
with observe_span(llm_span_name, metadata={...}) as llm_span:
    response = llm.chat_completion(messages=trimmed_messages, ...)
    if llm_span:
        llm_span.update(
            input={"message_count": len(trimmed_messages)},
            output={
                "finish_reason": finish_reason,
                "content_len": len(content),
                "tool_calls": len(tool_calls),
            },
        )
```

### Fix 3: trace output — capture final answer

Нужна переменная `_final_answer_text` которая устанавливается при final_answer и записывается в trace output при cleanup.

---

## Acceptance Criteria

- [ ] `trace.input` содержит `{"query": "..."}`
- [ ] `trace.output` содержит `answer` текст
- [ ] `llm_step_N` spans содержат message_count, finish_reason, tool_calls count
- [ ] Smoke test: один запрос → скачать trace из Langfuse → все поля заполнены
- [ ] Существующие observe points (#3, #4, #5) не сломаны

---

## Файлы для изменения

| Файл | Изменение |
|------|-----------|
| `src/services/agent_service.py` | Fix 1 (trace input), Fix 2 (llm_step output), Fix 3 (capture answer for trace output) |

`observability.py` менять не нужно — API уже поддерживает input/output, просто agent_service.py их не передаёт.
