# Техническая спецификация агента

> **Статус:** Phase 1  
> **Актуально на:** 2026-03-17  
> **LLM:** Qwen3-30B-A3B GGUF через `llama-server.exe`  
> **Embedding:** `Qwen/Qwen3-Embedding-0.6B` через TEI HTTP  
> **Reranker:** `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` через TEI HTTP  
> **Store:** Qdrant (dense + sparse, native RRF+MMR)

## Обзор

`AgentService` реализует агентский цикл поверх native function calling через `/v1/chat/completions`.
Старый текстовый ReAct с regex-парсингом `Thought/Action/FinalAnswer` удалён: модель работает через `messages + tools schema`, а вызовы инструментов приходят как structured `tool_calls`.

## Архитектура

### Основные компоненты

- `src/services/agent_service.py` — main loop, tool dispatch, SSE события, deterministic guardrails.
- `src/adapters/llm/llama_server_client.py` — HTTP-клиент к llama-server; `__call__()` для legacy completions и `chat_completion()` для function calling.
- `src/services/tools/tool_runner.py` — единый запуск инструментов с таймаутом и трейсом.
- `src/services/tools/*` — сами инструменты retrieval/compose/verify/finalize.

### Внутреннее состояние агента

`AgentService` держит:

- `_current_request_id`
- `_current_step`
- `_current_query`
- `_last_search_hits`
- `_last_search_route`
- `_last_plan_summary`
- `_last_compose_citations`
- `_last_coverage`

`AgentState` держит runtime guardrails:

- `coverage`
- `refinement_count`
- `max_refinements`
- `low_coverage_disclaimer`

## Tools schema для LLM

Модель видит ровно 5 функций:

1. `query_plan`
2. `search`
3. `rerank`
4. `compose_context`
5. `final_answer`

`router_select` убран из схемы.  
`verify` и `fetch_docs` остаются в `ToolRunner`, но вызываются только системно.

## Main loop

1. `AgentService.stream_agent_response()` создаёт `messages = [system, user]`.
2. LLM вызывается через `llm.chat_completion(...)`.
3. Если в ответе есть `message.content`, агент эмитит SSE `thought`.
4. Если в ответе есть `tool_calls`, каждый вызов:
   - нормализуется через `_normalize_tool_params()`
   - исполняется через `ToolRunner`
   - публикует SSE `tool_invoked`
   - публикует SSE `observation`
   - добавляет `role="tool"` message обратно в историю
5. После `compose_context` агент публикует SSE `citations`.
6. После `final_answer` агент выполняет системную `verify` и эмитит SSE `final`.

Целевой retrieval pipeline для ответа:

`search → rerank → compose_context`

## Sampling

Для Qwen3 non-thinking mode используются параметры:

- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `presence_penalty=1.5`

`temperature=0` не используется, чтобы не провоцировать repetition loops.

## Deterministic guardrails

### Coverage / refinement

- Coverage threshold: **0.65**
- Max refinements: **2**
- Проверка идёт после каждого успешного `compose_context`

### Abort guard

Если максимальный `dense_score` по `_last_search_hits` меньше `0.30` и refinement ещё не запускался:

- агент не делает refinement
- coverage принудительно сбрасывается к `0.0`
- выставляется `low_coverage_disclaimer`
- публикуется системный `thought`

### Hedged disclaimer

Если coverage остаётся ниже `0.50` после исчерпания refinement-лимита, финальный ответ получает пометку о низком покрытии.

### Verify

После `final_answer` агент системно вызывает `verify`.
Если answer не подтверждается и refinement ещё доступен, запускается дополнительный refinement search.

## Retrieval models

- Query embedding: `Qwen3-Embedding-0.6B` с instruction prefix  
  `Instruct: Given a user question about ML, AI, LLM or tech news, retrieve relevant Telegram channel posts\nQuery: `
- Document embedding: тот же encoder, но без prefix
- Reranker: `Qwen3-Reranker-0.6B-seq-cls` через TEI-compatible seq-cls wrapper

## SSE contract

Публичный контракт не меняется. Агент продолжает публиковать:

- `thought`
- `tool_invoked`
- `observation`
- `citations`
- `final`

Дополнительный `step_started` остаётся служебным событием.

## Что удалено по сравнению со старым ReAct

- `_generate_step()`
- `_parse_llm_response()`
- regex strip thinking / forced FinalAnswer
- CJK logit-bias hack
- текстовый парсинг `Thought/Action/FinalAnswer` через `/v1/completions`

## Развёртывание

`llama-server.exe` должен запускаться с chat-template поддержкой:

```bash
set GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1

llama-server.exe ^
  -hf unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M ^
  -c 16384 --parallel 2 ^
  --flash-attn on ^
  --cache-type-k q8_0 --cache-type-v q8_0 ^
  -ngl 99 --main-gpu 0 ^
  --jinja ^
  --reasoning-budget 0
```
