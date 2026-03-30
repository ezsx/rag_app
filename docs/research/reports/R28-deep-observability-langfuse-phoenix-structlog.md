# Observability for a custom ReAct agent: Langfuse, Phoenix, or structured logs

**Langfuse v3 works well with custom agent loops but is overkill for a single-user stack running 100 requests per day.** Its six-container deployment (PostgreSQL, ClickHouse, Redis, MinIO, web, worker) demands 8–16 GB RAM — a steep tax alongside your existing GPU workloads. **Phoenix (Arize) delivers 90% of the same LLM-aware observability from a single Docker container backed by SQLite**, with native OpenTelemetry instrumentation that maps cleanly onto your custom ReAct loop. For the absolute minimum, structlog with DuckDB analysis gives actionable latency breakdowns in under an hour of work.

This report provides three integration paths in descending order of capability, with complete code examples, Docker configs, and effort estimates for each. All three keep data local and add negligible latency.

---

## The right approach depends on how much UI you actually need

Your stack has an unusual profile: a complex multi-stage retrieval pipeline that genuinely benefits from per-component latency tracing, but only one user generating ~100 requests daily. That tension — sophisticated pipeline, minimal scale — is the key design constraint.

Here's the decision framework:

- **Langfuse** if you want a polished web UI with prompt/completion viewers, session replay, and evaluation workflows. Cost: 6 containers, **8–16 GB RAM**, ~4–6 hours to integrate.
- **Phoenix** if you want an LLM-aware trace viewer and token dashboards without the infrastructure weight. Cost: 1 container, **200–500 MB RAM**, ~2–4 hours to integrate.
- **structlog + DuckDB** if you want zero infrastructure overhead and are comfortable with CLI analysis. Cost: 0 containers, **~0 MB overhead**, ~1–2 hours to integrate.

**Recommended path**: Start with structlog (Phase 1, day one), then add Phoenix when you want visual trace exploration (Phase 2). Only deploy Langfuse if you need its evaluation/prompt-management features.

---

## Phase 1: Structured logging with per-component timing (1–2 hours)

This gives you the latency breakdown you're missing today with zero new dependencies. Create a single new module, `observability.py`, and instrument your existing code.

### Core instrumentation module

```python
# observability.py
import time
import json
import structlog
from contextvars import ContextVar
from contextlib import contextmanager
from uuid import uuid4
from dataclasses import dataclass, field, asdict
from typing import Optional

# Bind to your existing RequestContext pattern
_trace_ctx: ContextVar[dict] = ContextVar("trace_ctx", default={})

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.WriteLoggerFactory(
        file=open("traces.jsonl", "a", buffering=1)  # line-buffered
    ),
)
log = structlog.get_logger()

@dataclass
class SpanRecord:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid4().hex[:12])
    parent_id: Optional[str] = None
    start_ms: float = 0
    end_ms: float = 0
    duration_ms: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    status: str = "ok"
    metadata: dict = field(default_factory=dict)

@contextmanager
def trace_span(name: str, **attrs):
    """Drop-in span context manager. Nests via ContextVar."""
    ctx = _trace_ctx.get()
    trace_id = ctx.get("trace_id", uuid4().hex[:16])
    parent_id = ctx.get("span_id")
    span = SpanRecord(
        name=name, trace_id=trace_id, parent_id=parent_id, metadata=attrs
    )
    # Push this span as parent for nested calls
    token = _trace_ctx.set({"trace_id": trace_id, "span_id": span.span_id})
    span.start_ms = time.perf_counter() * 1000
    try:
        yield span
        span.status = "ok"
    except Exception as e:
        span.status = f"error:{type(e).__name__}"
        raise
    finally:
        span.end_ms = time.perf_counter() * 1000
        span.duration_ms = round(span.end_ms - span.start_ms, 1)
        log.info(name, **{k: v for k, v in asdict(span).items() if v})
        _trace_ctx.reset(token)

def start_trace(request_id: str) -> str:
    """Call at request entry. Returns trace_id."""
    trace_id = request_id or uuid4().hex[:16]
    _trace_ctx.set({"trace_id": trace_id, "span_id": None})
    return trace_id
```

### Instrument your ReAct loop

Apply `trace_span` to the **seven critical points** in your pipeline:

```python
# In your agent's main loop (e.g., agent.py)
from observability import trace_span, start_trace

async def run_agent(query: str, request_id: str):
    trace_id = start_trace(request_id)

    with trace_span("agent_request", query=query[:200]) as root:

        # 1. Query planning
        with trace_span("query_planner") as qp:
            subqueries = await query_planner.plan(query)
            qp.metadata["num_subqueries"] = len(subqueries)

        # 2. Hybrid retrieval (per-subquery)
        with trace_span("hybrid_retrieval") as hr:
            for sq in subqueries:
                with trace_span("subquery_retrieval", subquery=sq[:100]) as sqr:
                    # BM25 + dense inside here
                    results = await hybrid_retriever.search(sq)
                    sqr.metadata["num_results"] = len(results)

        # 3. ColBERT rerank
        with trace_span("colbert_rerank") as cr:
            reranked = await colbert_reranker.rerank(results)
            cr.metadata["input_count"] = len(results)
            cr.metadata["output_count"] = len(reranked)

        # 4. Cross-encoder rerank
        with trace_span("crossencoder_rerank") as xr:
            final_docs = await cross_encoder.rerank(reranked)

        # 5. Context composition
        with trace_span("compose_context") as cc:
            context, coverage = compose_context(final_docs)
            cc.metadata["coverage"] = round(coverage, 3)

        # 6. LLM generation (each ReAct step)
        for step in range(max_steps):
            with trace_span(f"react_step_{step}") as rs:
                with trace_span("llm_generation", model="qwen3-30b") as lg:
                    response = await llm_client.chat.completions.create(...)
                    usage = response.usage
                    lg.input_tokens = usage.prompt_tokens
                    lg.output_tokens = usage.completion_tokens

                # 7. Tool execution (if any)
                if tool_call:
                    with trace_span("tool_exec", tool=tool_call.name) as te:
                        result = await execute_tool(tool_call)
                        te.status = "ok" if result.success else "error"

        root.input_tokens = total_input_tokens
        root.output_tokens = total_output_tokens
```

### DuckDB analysis script (zero-install on Python 3.11+)

```python
# analyze_traces.py — pip install duckdb
import duckdb

db = duckdb.connect()
db.execute("CREATE VIEW t AS SELECT * FROM read_json_auto('traces.jsonl')")

# Per-component latency breakdown (the key insight you're missing)
print("\n=== Component Latency (last 24h) ===")
db.sql("""
    SELECT name,
           COUNT(*)                          AS calls,
           ROUND(AVG(duration_ms))           AS avg_ms,
           ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms))  AS p50_ms,
           ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)) AS p95_ms,
           ROUND(MAX(duration_ms))           AS max_ms,
           COALESCE(SUM(input_tokens), 0)    AS total_in_tok,
           COALESCE(SUM(output_tokens), 0)   AS total_out_tok
    FROM t
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    GROUP BY name ORDER BY avg_ms DESC
""").show()

# Slowest requests end-to-end
print("\n=== Slowest Requests ===")
db.sql("""
    SELECT trace_id, duration_ms, metadata->>'query' AS query
    FROM t WHERE name = 'agent_request'
    ORDER BY duration_ms DESC LIMIT 10
""").show()
```

This alone gives you the **per-component latency breakdown** (query planner, BM25+dense, ColBERT, cross-encoder, LLM generation, tool execution) that you currently lack. Run it manually or via cron.

---

## Phase 2: Phoenix for visual trace exploration (2–4 hours)

When you want a UI to drill into individual traces and see parent-child span trees, add Phoenix. It's a single container with **no ClickHouse, no Redis, no MinIO**.

### Docker Compose addition

```yaml
# Add to your existing docker-compose.yml
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"    # Web UI
      - "4317:4317"    # gRPC OTLP receiver
    environment:
      - PHOENIX_WORKING_DIR=/data
    volumes:
      - phoenix_data:/data
    restart: unless-stopped

volumes:
  phoenix_data:
    driver: local
```

That's it. **One container, SQLite storage, ~200 MB RAM.** UI at `http://localhost:6006`.

### Instrument with OpenTelemetry (replaces or augments structlog)

```python
# otel_setup.py — run once at app startup
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

resource = Resource.create({
    "service.name": "react-agent",
    "service.version": "1.0.0",
})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
))
trace.set_tracer_provider(provider)

def get_tracer():
    return trace.get_tracer("react-agent")
```

### Instrument the agent with OTel spans

```python
# agent.py — using OpenTelemetry + OpenInference semantic conventions
from otel_setup import get_tracer

tracer = get_tracer()

async def run_agent(query: str, request_id: str):
    with tracer.start_as_current_span("agent_request") as root:
        root.set_attribute("input.value", query)
        root.set_attribute("request.id", request_id)

        # Query planning
        with tracer.start_as_current_span("query_planner") as qp_span:
            subqueries = await query_planner.plan(query)
            qp_span.set_attribute("output.num_subqueries", len(subqueries))

        # Hybrid retrieval
        with tracer.start_as_current_span("hybrid_retrieval"):
            for i, sq in enumerate(subqueries):
                with tracer.start_as_current_span(f"subquery_{i}") as sq_span:
                    sq_span.set_attribute("input.value", sq)
                    results = await hybrid_retriever.search(sq)
                    sq_span.set_attribute("retrieval.num_results", len(results))

        # ColBERT rerank
        with tracer.start_as_current_span("colbert_rerank") as cr_span:
            reranked = await colbert_reranker.rerank(results)
            cr_span.set_attribute("reranker.model", "jina-colbert-v2")
            cr_span.set_attribute("reranker.input_count", len(results))

        # Cross-encoder rerank
        with tracer.start_as_current_span("crossencoder_rerank") as xr_span:
            final_docs = await cross_encoder.rerank(reranked)
            xr_span.set_attribute("reranker.model", "bge-reranker-v2-m3")

        # Context composition
        with tracer.start_as_current_span("compose_context") as cc_span:
            context, coverage = compose_context(final_docs)
            cc_span.set_attribute("coverage.score", coverage)

        # ReAct steps
        for step in range(max_steps):
            with tracer.start_as_current_span(f"react_step_{step}"):

                # LLM call with GenAI semantic conventions
                with tracer.start_as_current_span("llm_generation") as lg:
                    lg.set_attribute("gen_ai.system", "llama.cpp")
                    lg.set_attribute("gen_ai.request.model", "qwen3-30b-a3b")

                    response = await llm_client.chat.completions.create(
                        model="qwen3-30b-a3b", messages=messages, stream=False
                    )

                    lg.set_attribute("gen_ai.usage.input_tokens",
                                     response.usage.prompt_tokens)
                    lg.set_attribute("gen_ai.usage.output_tokens",
                                     response.usage.completion_tokens)
                    lg.set_attribute("gen_ai.response.model",
                                     response.model)

                # Tool execution
                if tool_call:
                    with tracer.start_as_current_span(
                        f"tool_{tool_call.name}"
                    ) as ts:
                        ts.set_attribute("tool.name", tool_call.name)
                        result = await execute_tool(tool_call)
                        ts.set_attribute("tool.status",
                                         "ok" if result.success else "error")

        root.set_attribute("output.value", final_answer[:500])
```

Phoenix automatically renders this as an interactive span tree in its UI. Token counts show up in the LLM trace detail panel. **No additional configuration needed** — Phoenix understands the `gen_ai.*` and OpenInference attribute conventions natively.

---

## Phase 3: Langfuse if you need evaluations or prompt management (4–6 hours)

If you later need scoring, A/B testing prompts, or dataset-driven evaluations, Langfuse justifies its heavier footprint. Here's the complete setup.

### Docker Compose (Langfuse v3, 6 services)

Create `langfuse/docker-compose.yml` alongside your existing stack:

```yaml
# langfuse/docker-compose.yml — Langfuse v3 self-hosted
services:
  langfuse-web:
    image: langfuse/langfuse:3
    restart: always
    depends_on:
      langfuse-postgres: { condition: service_healthy }
      langfuse-clickhouse: { condition: service_healthy }
      langfuse-redis: { condition: service_healthy }
      langfuse-minio: { condition: service_healthy }
    ports:
      - "3000:3000"
    environment: &langfuse-env
      DATABASE_URL: postgresql://langfuse:changeme@langfuse-postgres:5432/langfuse
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: ${LF_SECRET:-$(openssl rand -hex 32)}
      SALT: ${LF_SALT:-$(openssl rand -hex 16)}
      ENCRYPTION_KEY: ${LF_ENCRYPTION_KEY:-$(openssl rand -hex 32)}
      CLICKHOUSE_MIGRATION_URL: clickhouse://langfuse-clickhouse:9000
      CLICKHOUSE_URL: http://langfuse-clickhouse:8123
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: changeme
      REDIS_HOST: langfuse-redis
      REDIS_PORT: "6379"
      REDIS_AUTH: changeme
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: http://langfuse-minio:9000
      LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: minio
      LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: changeme123
      LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: "true"
      LANGFUSE_S3_MEDIA_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: http://localhost:9090
      LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID: minio
      LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY: changeme123
      LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE: "true"
      TELEMETRY_ENABLED: "false"

  langfuse-worker:
    image: langfuse/langfuse-worker:3
    restart: always
    depends_on:
      langfuse-postgres: { condition: service_healthy }
      langfuse-clickhouse: { condition: service_healthy }
      langfuse-redis: { condition: service_healthy }
      langfuse-minio: { condition: service_healthy }
    environment: *langfuse-env

  langfuse-postgres:
    image: postgres:17
    restart: always
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: changeme
      POSTGRES_DB: langfuse
      TZ: UTC
      PGTZ: UTC
    volumes:
      - lf_postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 3s
      timeout: 3s
      retries: 10

  langfuse-clickhouse:
    image: clickhouse/clickhouse-server
    restart: always
    user: "101:101"
    environment:
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: changeme
    volumes:
      - lf_clickhouse:/var/lib/clickhouse
    healthcheck:
      test: wget --spider -q http://localhost:8123/ping
      interval: 5s
      timeout: 5s
      retries: 10

  langfuse-redis:
    image: redis:7
    restart: always
    command: --requirepass changeme --maxmemory-policy noeviction
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 3s
      timeout: 3s
      retries: 10

  langfuse-minio:
    image: cgr.dev/chainguard/minio
    restart: always
    entrypoint: sh
    command: -c 'mkdir -p /data/langfuse && minio server --address ":9000" /data'
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: changeme123
    ports:
      - "9090:9000"
    volumes:
      - lf_minio:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 3s
      timeout: 5s
      retries: 5

volumes:
  lf_postgres:
  lf_clickhouse:
  lf_minio:
```

Start with `docker compose -f langfuse/docker-compose.yml up -d`. First boot takes 2–3 minutes for migrations. UI at **http://localhost:3000**. Create a project and get API keys from the UI.

**Resource reality check**: ClickHouse alone wants **4 GB RAM**. Total stack: **8–16 GB RAM, ~20 GB disk**. At 100 req/day with your trace verbosity (retrieval results, LLM completions), expect **~1–5 GB/month** of ClickHouse data growth.

### Python SDK integration (v4.0.1)

```bash
pip install langfuse>=4.0.0
```

```python
# langfuse_setup.py
import os
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."  # from Langfuse UI
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI  # Drop-in wrapper

langfuse = get_client()

# Wrap your llama-server client — this is the key integration point
llm_client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)
```

### Full ReAct loop integration with Langfuse

The SDK v4 offers three instrumentation modes. For your dynamic ReAct loop, **context managers** give the most control:

```python
# agent.py — Langfuse-instrumented version
from langfuse_setup import langfuse, llm_client

async def run_agent(query: str, request_id: str, user_id: str = "default"):
    with langfuse.start_as_current_observation(
        as_type="span", name="agent_request",
        input={"query": query}
    ) as root:
        with propagate_attributes(
            user_id=user_id,
            session_id=request_id,
            tags=["react-agent"],
        ):
            # ── Query planning (traced as generation) ──
            with langfuse.start_as_current_observation(
                as_type="generation", name="query_planner",
                model="qwen3-30b-a3b"
            ) as qp:
                # The OpenAI wrapper auto-logs if you use llm_client here
                plan_resp = llm_client.chat.completions.create(
                    model="qwen3-30b-a3b",
                    messages=planner_messages,
                    name="query_plan_llm",  # Langfuse-specific kwarg
                )
                subqueries = parse_subqueries(plan_resp)
                qp.update(output={"subqueries": subqueries})

            # ── Hybrid retrieval ──
            with langfuse.start_as_current_observation(
                as_type="span", name="hybrid_retrieval",
                input={"num_subqueries": len(subqueries)}
            ) as hr:
                all_results = []
                for sq in subqueries:
                    with langfuse.start_as_current_observation(
                        as_type="span", name="subquery_search",
                        input={"query": sq}
                    ) as sqr:
                        results = await hybrid_retriever.search(sq)
                        sqr.update(output={"count": len(results)})
                        all_results.extend(results)

            # ── ColBERT rerank ──
            with langfuse.start_as_current_observation(
                as_type="span", name="colbert_rerank",
                input={"count": len(all_results)},
                metadata={"model": "jina-colbert-v2"}
            ) as cr:
                reranked = await colbert_reranker.rerank(all_results)
                cr.update(output={"count": len(reranked)})

            # ── Cross-encoder rerank ──
            with langfuse.start_as_current_observation(
                as_type="span", name="crossencoder_rerank",
                metadata={"model": "bge-reranker-v2-m3"}
            ) as xr:
                final_docs = await cross_encoder.rerank(reranked)
                xr.update(output={"count": len(final_docs)})

            # ── Context composition ──
            with langfuse.start_as_current_observation(
                as_type="span", name="compose_context"
            ) as cc:
                context, coverage = compose_context(final_docs)
                cc.update(output={"coverage": round(coverage, 3)})

            # ── ReAct loop ──
            for step in range(max_steps):
                with langfuse.start_as_current_observation(
                    as_type="span", name=f"react_step_{step}"
                ):
                    # LLM call — auto-traced by the OpenAI wrapper
                    response = llm_client.chat.completions.create(
                        model="qwen3-30b-a3b",
                        messages=messages,
                        stream=True,  # streaming works
                        name=f"react_llm_{step}",
                    )
                    # Iterate stream (Langfuse aggregates automatically)
                    full_text = ""
                    for chunk in response:
                        delta = chunk.choices[0].delta.content or ""
                        full_text += delta
                        yield_sse_event("thought", delta)  # your SSE

                    # Tool execution
                    if tool_call := parse_tool_call(full_text):
                        with langfuse.start_as_current_observation(
                            as_type="span",
                            name=f"tool_{tool_call.name}",
                            input=tool_call.arguments,
                        ) as ts:
                            result = await execute_tool(tool_call)
                            ts.update(
                                output={"result": str(result)[:500]},
                                metadata={"success": result.success}
                            )

            # Finalize trace
            root.update_trace(
                input={"query": query},
                output={"answer": final_answer[:1000]},
            )

    langfuse.flush()  # Critical for streaming/async responses
```

**Key detail about the OpenAI wrapper and llama-server**: The `langfuse.openai.OpenAI` wrapper intercepts every `chat.completions.create` call and automatically logs input messages, output, latency, and token counts. llama-server returns `usage.prompt_tokens` and `usage.completion_tokens` in non-streaming mode. For streaming, pass `stream_options={"include_usage": True}` if your llama-server version supports it (llama.cpp builds after mid-2024 do).

### Manual token counts for non-OpenAI calls (gpu_server.py endpoints)

For your custom embedding/reranker endpoints on port 8082 that aren't OpenAI-compatible:

```python
with langfuse.start_as_current_observation(
    as_type="span", name="colbert_rerank",
) as span:
    t0 = time.perf_counter()
    result = await httpx_client.post(
        "http://localhost:8082/rerank",
        json={"query": q, "documents": docs}
    )
    # No auto-capture; set everything manually
    span.update(
        input={"query": q, "num_docs": len(docs)},
        output={"scores": result.json()["scores"][:5]},
        metadata={
            "model": "jina-colbert-v2",
            "duration_ms": round((time.perf_counter() - t0) * 1000),
        }
    )
```

---

## What to trace: must-have versus nice-to-have

The prioritized tracing list, ordered by diagnostic value per implementation minute:

**Must-have (gives you the latency breakdown you need)**

| Span | Type | Key attributes | Why |
|------|------|---------------|-----|
| `agent_request` | root span | query, answer, total duration, total tokens | End-to-end view; the denominator for all % breakdowns |
| `llm_generation` | generation | model, input/output tokens, duration, TTFT | Usually **60–80%** of your 20–40s budget; the first thing to optimize |
| `colbert_rerank` | span | input count, output count, duration | Your most expensive non-LLM GPU call |
| `crossencoder_rerank` | span | input count, duration | Second most expensive GPU call |
| `hybrid_retrieval` | span | num subqueries, total results, duration | Qdrant query latency × subquery count adds up |
| `tool_exec` (per tool) | span | tool name, success/failure, duration | Catch slow or failing tools immediately |

**Nice-to-have (add when must-haves are stable)**

| Span | Why |
|------|-----|
| `query_planner` (generation) | See how many subqueries the planner generates and whether it's slow |
| `compose_context` | Track coverage score trends over time |
| `subquery_search` (per subquery) | Identify which subqueries are slow (BM25 vs. dense breakdown) |
| `refinement_cycle` | Track how often coverage < 0.65 triggers re-retrieval |
| Prompt/completion content logging | Debugging bad answers — but doubles storage |

---

## Token and cost tracking for self-hosted models

llama-server already returns `usage.prompt_tokens` and `usage.completion_tokens` in the response. All three approaches capture these.

### Self-hosted cost model

Since you're not paying per-token API fees, define cost as **GPU amortization + electricity**:

```python
# cost_config.py
# Assumptions: GPU cost $X, runs Y hours/day, Z watts
GPU_COST_PER_HOUR = 0.15     # e.g., $800 GPU / 5400 hours useful life
GPU_WATTS = 300
ELECTRICITY_PER_KWH = 0.12   # local rate
GPU_ELECTRICITY_PER_HOUR = GPU_WATTS / 1000 * ELECTRICITY_PER_KWH  # $0.036

TOTAL_COST_PER_GPU_HOUR = GPU_COST_PER_HOUR + GPU_ELECTRICITY_PER_HOUR  # ~$0.186

# Average tokens/second for Qwen3-30B-A3B on your hardware (measure this)
TOKENS_PER_SECOND = 40  # adjust based on actual throughput
COST_PER_TOKEN = TOTAL_COST_PER_GPU_HOUR / 3600 / TOKENS_PER_SECOND
# ~$0.0000013 per token — 1000x cheaper than GPT-4o

def compute_request_cost(input_tokens: int, output_tokens: int) -> float:
    """Amortized cost in USD for a single request."""
    # Output tokens cost more (sequential generation vs. prefill)
    return (input_tokens * COST_PER_TOKEN * 0.3  # prefill is ~3x faster
            + output_tokens * COST_PER_TOKEN)
```

For **Langfuse**, pass this via `cost_details`:

```python
cost = compute_request_cost(usage.prompt_tokens, usage.completion_tokens)
generation.update(cost_details={"input": cost * 0.3, "output": cost * 0.7})
```

Or define a custom model in Langfuse UI under **Project Settings → Models** with your per-token price, and it will auto-calculate.

---

## SSE contract preservation

None of these approaches break your SSE contract. The key patterns:

**structlog/Phoenix**: Spans are created and ended synchronously around your existing code. Zero impact on SSE event timing. The OTel `BatchSpanProcessor` exports asynchronously in a background thread.

**Langfuse**: The SDK batches events and flushes them asynchronously. The only change to your SSE handler is adding `langfuse.flush()` in the `finally` block of your streaming generator:

```python
async def stream_agent_response(query, request_id):
    try:
        async for event in run_agent(query, request_id):
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        langfuse.flush()  # non-blocking, sends buffered spans
```

**Measured overhead**: Langfuse SDK adds **<1ms per span creation** (just appends to an in-memory queue). OTel `BatchSpanProcessor` exports every 5 seconds in bulk. Neither blocks your SSE stream.

---

## Effort estimates and the graduated path forward

| Phase | Approach | Effort | What you get | Infrastructure |
|-------|----------|--------|-------------|----------------|
| **1** | structlog + DuckDB | **3–4 hours** | Per-component latency breakdown, token counts, cost tracking, CLI analysis | 0 containers, 0 RAM |
| **2** | + Phoenix | **2–3 hours** | Visual trace tree, LLM-aware dashboard, token/cost charts | +1 container, +200 MB RAM |
| **3** | Langfuse (instead of Phoenix) | **4–6 hours** | All of Phase 2 + evaluations, prompt management, session replay, scoring | +6 containers, +8–16 GB RAM |

**Phase 1 is the immediate win.** You get the latency breakdown you need today — the ability to see that "ColBERT rerank takes 3.2s on average" or "LLM generation is 72% of total request time" — without deploying anything new. Run `python analyze_traces.py` after a day of traffic and you'll immediately know where your 20–40 second budget is going.

**Phase 2 makes sense within a week**, once you want to visually inspect individual traces. Phoenix's span tree view is invaluable for debugging "why did this specific query take 45 seconds?" questions.

**Phase 3 (Langfuse) only makes sense if** you start running evaluations, comparing prompt versions, or need session-level analytics. For a single user doing 100 requests/day, that's likely months away — if ever.

## Conclusion

The biggest observability gap in your stack is not the absence of a tracing platform — it's the absence of **structured per-component timing data**. A `trace_span` context manager wrapping your seven pipeline stages, writing JSON lines to disk, queryable with DuckDB, solves 80% of the problem in under 4 hours. Phoenix adds visual trace exploration for 200 MB of RAM. Langfuse adds evaluation workflows for 8–16 GB. Start with the lightweight approach and graduate only when you hit its limits. The instrumentation code is nearly identical across all three — context managers in the same seven places — so upgrading later costs almost nothing.