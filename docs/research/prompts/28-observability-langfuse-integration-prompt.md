# Prompt 28: Observability — Langfuse integration into self-hosted RAG + ReAct agent

## Как использовать

Прочитай attached context (`docs/research/prompts/25-growth-ceiling-context.md`) для общего понимания проекта. Ниже — специфичный контекст для этого research.

## Роль

Ты — production engineer который интегрировал Langfuse в несколько LLM-приложений. Нужна конкретная инструкция по интеграции, не обзор возможностей.

## Задача

Исследовать как интегрировать Langfuse (или альтернативу) в наш self-hosted RAG + ReAct agent стек. Дать конкретный план с кодом.

## Текущий стек

### Инфраструктура
- **LLM**: Qwen3-30B-A3B через llama-server.exe (OpenAI-compatible API, localhost:8080)
- **Embedding + Reranker + ColBERT**: gpu_server.py — custom HTTP server (не OpenAI-compatible), localhost:8082
- **Vector DB**: Qdrant в Docker, localhost:6333
- **API**: FastAPI, SSE streaming через `/v1/agent/stream`
- **Docker**: API + Qdrant в Docker. GPU workloads нативно в WSL2 / Windows host
- **Принцип**: self-hosted first. Никаких managed API для core inference

### Agent architecture
- Custom ReAct loop в `src/services/agent_service.py` (~2200 строк)
- 15 LLM-visible tools, dynamic visibility (max 5 visible per step)
- Tool calls через native function calling (OpenAI-compatible `/v1/chat/completions`)
- Per-request isolation через `RequestContext` + `ContextVar`
- SSE events: `step_started`, `thought`, `tool_invoked`, `observation`, `citations`, `final`
- Coverage-based refinement (threshold 0.65, max 2 refinements)

### Retrieval pipeline
```
User query → QueryPlannerService (LLM, 3-6 subqueries)
  → HybridRetriever (BM25 top-100 + Dense top-20 → weighted RRF 3:1)
  → ColBERT rerank (jina-colbert-v2, MaxSim)
  → Cross-encoder rerank (bge-reranker-v2-m3)
  → Channel dedup (max 2/channel)
  → compose_context → LLM generation
```

### Что уже есть по observability
- `time.perf_counter()` / `time.monotonic()` в некоторых tools (took_ms в SSE observation)
- `logging` (Python standard, INFO/DEBUG)
- SSE events содержат step/request_id
- Cooperative deadline (wall-clock budget per request)
- **Нет**: structured tracing, per-component latency breakdown, token counting, cost tracking, dashboard

### Constraints
- Self-hosted Langfuse (Docker) или lightweight альтернатива
- Не SaaS — данные остаются локально
- Минимальный overhead на latency (agent уже 20-40s per request)
- Не ломать SSE контракт
- Python 3.11+

## Что нужно исследовать

### 1. Langfuse integration architecture

- Как Langfuse интегрируется с **custom agent loop** (не LangChain/LlamaIndex)?
- `@observe()` decorator vs manual `trace.span()` — что лучше для нашего ReAct loop?
- Как трейсить **multi-step agent** с dynamic tool calls? Один trace per request, spans per step/tool?
- Как трейсить **streaming** (SSE)? Langfuse поддерживает streaming LLM calls?
- Integration с **OpenAI-compatible API** (llama-server) — Langfuse умеет wrap OpenAI client?

### 2. What to trace (конкретные spans)

Нам нужна latency breakdown по этим компонентам:

| Component | Where | What to measure |
|-----------|-------|----------------|
| LLM inference | llama-server call | TTFT, total generation time, tokens in/out |
| Query planning | QueryPlannerService | LLM call time, plan quality |
| BM25 + Dense search | HybridRetriever.search_with_plan | Qdrant query time per subquery |
| ColBERT rerank | TEIRerankerClient (colbert endpoint) | Rerank time, batch size |
| Cross-encoder rerank | TEIRerankerClient (rerank endpoint) | Rerank time |
| Tool execution | ToolRunner.run() | Per-tool time, success/failure |
| compose_context | compose_context tool | Coverage computation time |
| Total request | agent_service.stream_agent_response | End-to-end latency |

### 3. Token tracking

- llama-server возвращает `usage.prompt_tokens` и `usage.completion_tokens` в response
- Как Langfuse считает tokens для non-OpenAI models?
- Можно ли передать token counts вручную?
- Cost per query — как считать для self-hosted (electricity + GPU amortization)?

### 4. Deployment

- Self-hosted Langfuse: Docker Compose setup (PostgreSQL + ClickHouse + Langfuse server)
- Как добавить в наш existing `deploy/compose/compose.dev.yml` без конфликтов?
- Resource requirements (RAM, disk) для Langfuse при нашей нагрузке (~100 requests/day max)
- Retention policy — сколько хранить traces?

### 5. Alternatives to Langfuse

Если Langfuse overkill для нашего масштаба, что проще:

- **OpenTelemetry + Jaeger** — standard, но не LLM-aware (нет token tracking, prompt management)
- **LangSmith** — LangChain ecosystem, SaaS only, не подходит
- **Lunary** — open-source, lightweight, MIT
- **Phoenix (Arize)** — open-source, хороший для eval + traces
- **Custom structured logging** — `structlog` + JSON logs + simple dashboard (Grafana?)

Для каждого: effort интеграции, что даёт, что не даёт, trade-offs.

### 6. Minimal viable observability (если Langfuse слишком тяжёл)

Если ответ "Langfuse overkill для 1 user / 100 req/day":
- Что минимально нужно? `structlog` + JSON + per-component timing?
- Можно ли обойтись расширением текущих SSE events (добавить timing metadata)?
- Как визуализировать без UI? Python script → markdown table?

## Формат ответа

```markdown
## Recommended Approach
- Langfuse / alternative / custom — с обоснованием

## Integration Plan
- конкретные файлы, декораторы, spans
- code examples для нашего стека

## Deployment
- Docker Compose additions
- configuration

## What to Trace (prioritized)
- must-have spans
- nice-to-have spans

## Token & Cost Tracking
- approach

## Minimal Viable Alternative
- if Langfuse is overkill

## Effort Estimate
- hours/days per approach
```

Нужна конкретика: какой файл менять, какой декоратор ставить, какой span создавать. Не abstract "add observability".
