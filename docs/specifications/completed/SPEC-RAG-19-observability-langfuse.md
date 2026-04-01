# SPEC-RAG-19: Observability — Langfuse Integration

> **Статус**: Draft
> **Создан**: 2026-03-30
> **Research basis**: R28-deep-observability-langfuse-phoenix-structlog, R25-deep-production-gap-analysis
> **Depends on**: SPEC-RAG-17 (production hardening), SPEC-RAG-18 (eval pipeline v2)
> **Scope**: self-hosted Langfuse v3, per-component tracing, LLM token tracking, Docker Compose setup

---

## 1. Цель

Добавить production-grade observability в RAG pipeline. Сейчас **ноль structured metrics** per component — не знаем где bottleneck, сколько стоит каждый LLM call, какие tools медленные. Это был top finding из R25 (production gap analysis) и прямая причина долгого debugging сегодня.

После интеграции:
1. Per-component latency breakdown: LLM inference, retrieval, rerank, tool execution
2. Token usage tracking per request (input/output/total)
3. Trace visualization с parent-child span trees
4. Request-level drill-down через web UI (localhost:3000)

---

## 2. Почему Langfuse

| Критерий | Langfuse | Phoenix | structlog |
|----------|---------|---------|-----------|
| Web UI | Полноценный: traces, prompts, eval | Traces only | Нет |
| LLM-aware | Да: generations, token tracking | Да через OTel | Нет |
| Self-hosted | Да, Docker Compose | Да, 1 container | N/A |
| RAM | ~1-2GB idle, до 4-8GB peak (ClickHouse) | ~200MB | 0 |
| OpenAI wrapper | Drop-in, авто-трейсит LLM calls | Нет | Нет |
| Eval integration | Scoring, datasets, prompt management | Нет | Нет |
| Effort | 4-6ч | 2-4ч | 1-2ч |
| Portfolio signal | Сильный — industry standard | Средний | Слабый |

Langfuse = готовый production инструмент (19K+ stars, MIT). На собесе "у нас Langfuse" > "мы написали structlog wrapper". Eval integration полезен для будущих прогонов.

---

## 3. Архитектура

### 3.1 Deployment

```
Существующий стек:
  V100 → llama-server.exe (Qwen3.5-35B-A3B)
  5060 Ti → gpu_server.py (embedding + reranker + ColBERT)
  Docker → API + Qdrant

Добавляется:
  Docker → Langfuse (6 containers: web, worker, postgres, clickhouse, redis, minio)
  UI → http://localhost:3000
```

Langfuse контейнеры в **отдельном** compose файле: `deploy/compose/compose.langfuse.yml`.

**Networking**: API контейнер обращается к Langfuse через `host.docker.internal:3000` (тот же паттерн что llama-server). Langfuse compose работает в своей сети, API контейнер не нужно подключать к ней. Langfuse SDK шлёт traces по HTTP — достаточно чтобы хост видел порт 3000.

**RAM**: ~1-2GB idle, 4-8GB realistic peak, 8-16GB worst-case per R28. При нашей нагрузке (~100 req/day) ожидаем нижнюю границу. У нас 64GB RAM — хватает. Мониторить через `docker stats` после запуска.

### 3.2 Instrumentation points

7 критических точек для tracing:

| # | Span name | Component | Что трейсим |
|---|-----------|-----------|-------------|
| 1 | `agent_request` | AgentService.stream_agent_response | Root span: query, total latency, total tokens |
| 2 | `llm_call` | LlamaServerClient | Каждый LLM call: prompt_tokens, completion_tokens, model, finish_reason |
| 3 | `query_planner` | QueryPlannerService.make_plan | Plan generation: num_subqueries, strategy |
| 4 | `hybrid_retrieval` | HybridRetriever.search_with_plan | BM25 + dense: num_results, route_used |
| 5 | `rerank` | TEIRerankerClient | ColBERT + cross-encoder: input_count, top_score |
| 6 | `tool_execution` | ToolRunner.run | Per-tool: name, ok/error, took_ms |
| 7 | `compose_context` | compose_context tool | Coverage, num_citations |

### 3.3 Trace structure (дерево для одного request)

```
agent_request (root)
├── llm_call (step 1: tool selection)
├── tool_execution: query_plan
│   └── llm_call (query planner LLM)
├── tool_execution: search
│   └── hybrid_retrieval
│       ├── bm25_search
│       └── dense_search
├── tool_execution: rerank
│   ├── colbert_rerank
│   └── crossencoder_rerank
├── llm_call (step 2: tool selection)
├── tool_execution: compose_context
├── llm_call (step 3: final answer generation)
└── tool_execution: final_answer
```

---

## 4. Реализация

### 4.1 Docker Compose

Новый файл: `deploy/compose/compose.langfuse.yml`

Содержимое из R28 §Phase 3 (6 сервисов: web, worker, postgres, clickhouse, redis, minio). Порт web UI: 3000. Все данные в Docker volumes.

### 4.2 Python SDK setup

Новый файл: `src/core/observability.py`

**Все Langfuse imports — lazy внутри этого модуля.** Никакой другой файл не импортирует `langfuse` напрямую. Если пакет не установлен → все функции возвращают nullcontext/None, приложение работает без tracing.

```python
"""
Langfuse observability — lazy imports, graceful degradation.

Если langfuse не установлен или сервер недоступен — все функции
возвращают nullcontext/None, zero impact на runtime.
"""
import logging
from contextlib import contextmanager, nullcontext

logger = logging.getLogger(__name__)

_client = None
_enabled = None  # None = not checked yet

def _try_init():
    """Lazy init: один раз пробуем импортировать и подключиться."""
    global _client, _enabled
    if _enabled is not None:
        return _enabled
    try:
        from langfuse import get_client
        _client = get_client()
        _enabled = True
        logger.info("Langfuse observability enabled")
    except ImportError:
        _enabled = False
        logger.info("langfuse package not installed — observability disabled")
    except Exception as e:
        _enabled = False
        logger.warning("Langfuse init failed — observability disabled: %s", e)
    return _enabled

def get_langfuse():
    """Возвращает Langfuse client или None."""
    if not _try_init():
        return None
    return _client

@contextmanager
def observe_span(name, **kwargs):
    """Context manager для span. Graceful: nullcontext если Langfuse недоступен."""
    client = get_langfuse()
    if client is None:
        yield None
        return
    with client.start_as_current_observation(as_type="span", name=name, **kwargs) as span:
        yield span

@contextmanager
def observe_llm_call(name="llm_call", model="", **kwargs):
    """Context manager для LLM generation span."""
    client = get_langfuse()
    if client is None:
        yield None
        return
    with client.start_as_current_observation(
        as_type="generation", name=name, model=model, **kwargs
    ) as gen:
        yield gen
```

**Правило**: все runtime модули импортируют только `from core.observability import observe_span, observe_llm_call`. Никогда `from langfuse import ...` напрямую.

### 4.3 LLM call tracing (manual instrumentation)

**ВАЖНО**: `LlamaServerClient` построен на `requests.Session` с двумя интерфейсами:
- `__call__()` → `/v1/completions` (legacy, для qa_service и query_planner)
- `chat_completion()` → `/v1/chat/completions` (agent, function calling)

Drop-in OpenAI wrapper **не применим**. Instrumentation через manual spans.

В `src/adapters/llm/llama_server_client.py`:

```python
from core.observability import observe_llm_call

def chat_completion(self, messages, tools=None, max_tokens=512, ...):
    with observe_llm_call(
        name="llm_chat_completion",
        model=self.model,
        input_messages=messages,
        tools=tools,
    ) as span:
        resp = self._session.post(...)
        data = resp.json()
        # Извлекаем token usage из response
        usage = data.get("usage", {})
        if span:
            span.update(
                output=data.get("choices", [{}])[0].get("message", {}),
                usage={"input": usage.get("prompt_tokens", 0),
                       "output": usage.get("completion_tokens", 0)},
            )
        return data

def __call__(self, prompt, max_tokens=512, ...):
    with observe_llm_call(
        name="llm_completion",
        model=self.model,
        input_prompt=prompt,
    ) as span:
        resp = self._session.post(...)
        data = resp.json()
        usage = data.get("usage", {})
        if span:
            span.update(
                output=data.get("choices", [{}])[0].get("text", ""),
                usage={"input": usage.get("prompt_tokens", 0),
                       "output": usage.get("completion_tokens", 0)},
            )
        return data
```

`observe_llm_call` — context manager из `observability.py` который создаёт Langfuse generation span если клиент доступен, иначе nullcontext.

### 4.4 Agent loop tracing

В `src/services/agent_service.py` — `stream_agent_response`:

**ВАЖНО**: SSE async generator чувствителен к ContextVar cleanup (см. existing `_request_ctx.reset` try/except в finally блоке). Langfuse span **не оборачивает** весь generator. Вместо этого:

1. **Start span** в начале generator (before yield loop)
2. **Update span** по мере выполнения (tools, tokens)
3. **End span** в `finally` блоке с explicit `flush`

```python
from core.observability import get_langfuse

async def stream_agent_response(self, query, ...):
    langfuse = get_langfuse()
    root_span = None
    try:
        if langfuse:
            root_span = langfuse.start_as_current_observation(
                as_type="span", name="agent_request",
                input={"query": query, "request_id": request_id}
            ).__enter__()

        # ... existing agent loop with yields ...

    finally:
        if root_span:
            try:
                root_span.update(output={"coverage": coverage, "steps": step})
                root_span.__exit__(None, None, None)
            except Exception:
                pass  # Не crash'им на observability cleanup
        # ... existing _request_ctx.reset ...
```

Не используем `with` statement — async generator может cleanup'иться в другом context. Manual enter/exit безопаснее.

### 4.5 Tool execution tracing

В `src/services/tools/tool_runner.py` — `run`:

```python
from core.observability import observe_span

def run(self, name, params, ...):
    with observe_span(f"tool:{name}", input=params) as span:
        result = self._tools[name](**params)
        if span:
            span.update(output={"ok": result.ok, "took_ms": result.took_ms})
        return result
```

### 4.6 Retrieval tracing

В `src/adapters/search/hybrid_retriever.py`:

```python
from core.observability import observe_span

def search_with_plan(self, query, plan):
    with observe_span("hybrid_retrieval",
                      input={"query": query, "strategy": plan.strategy}) as span:
        # ... existing search logic ...
        if span:
            span.update(output={"num_results": len(results), "route": route})
```

Все модули импортируют **только** `from core.observability import observe_span, observe_llm_call`. Никогда langfuse напрямую.

---

## 5. Graceful degradation

**Langfuse не должен быть обязательной зависимостью.**

Принцип: **все langfuse imports lazy внутри `src/core/observability.py`**. Никакой другой файл не делает `from langfuse import ...`.

- Пакет `langfuse` не установлен → `_try_init()` ловит `ImportError` → `_enabled = False` → все `observe_span/observe_llm_call` = nullcontext → zero impact
- Пакет установлен, но сервер недоступен → `_try_init()` ловит connection error → `_enabled = False` → то же самое
- Пакет установлен, сервер работает → tracing активен
- Runtime exception в span → catch в `observe_span`, log warning, не crash
- SSE generator cleanup → explicit enter/exit в finally, не with statement

---

## 6. Что НЕ входит в scope

- Prompt management через Langfuse UI (позже)
- Eval scoring через Langfuse (у нас свой eval pipeline)
- Alerting / SLO monitoring
- Custom dashboards
- Multi-user auth в Langfuse

---

## 7. Acceptance criteria

1. Langfuse compose поднимается одной командой: `docker compose -f deploy/compose/compose.langfuse.yml up -d`
2. UI доступен на localhost:3000
3. Каждый agent request создаёт trace с parent-child spans
4. LLM calls трейсятся через `observe_llm_call` в LlamaServerClient (tokens, model, latency)
5. Tool execution трейсится (name, ok/error, took_ms)
6. Retrieval трейсится (num_results, route, strategy)
7. API работает без Langfuse (graceful degradation)
8. Нет дополнительной latency > 5ms per request от tracing

---

## 8. Чеклист реализации

### Infrastructure
- [ ] Создать `deploy/compose/compose.langfuse.yml`
- [ ] Добавить Langfuse env vars в `.env.example`
- [ ] Проверить что compose.dev.yml и compose.langfuse.yml не конфликтуют по портам
- [ ] Добавить `langfuse>=4.0.0` в requirements (optional)

### Code
- [ ] Создать `src/core/observability.py` (lazy singleton, graceful degradation)
- [ ] Инструментировать `LlamaServerClient` (manual `observe_llm_call` в `__call__` и `chat_completion`)
- [ ] Инструментировать `AgentService.stream_agent_response` (root span)
- [ ] Инструментировать `ToolRunner.run` (per-tool spans)
- [ ] Инструментировать `HybridRetriever.search_with_plan` (retrieval span)
- [ ] Инструментировать `TEIRerankerClient` (rerank spans)
- [ ] Инструментировать `QueryPlannerService.make_plan` (planner span)

### Verification
- [ ] Запустить один agent request, проверить trace в UI
- [ ] Проверить что tokens считаются корректно
- [ ] Проверить что API работает без Langfuse (stop langfuse containers)
- [ ] Запустить eval на 3 вопросах, проверить что traces создаются

### Docs
- [ ] Обновить `docs/planning/project_scope.md` (observability = done)
- [ ] Обновить команды запуска в infra memo
- [ ] Добавить запись в decision log
