# Async architecture patterns for FastAPI with LLM inference

**Migrating a synchronous FastAPI + vLLM application to fully async requires coordinated changes across five domains: the HTTP client, shared state management, tool execution, SSE streaming, and service lifecycle.** The core problem is that synchronous `requests.Session.post()` calls block the uvicorn event loop, while a `@lru_cache` singleton with mutable instance state creates cross-request data corruption under concurrency. This report provides production-ready patterns for each domain, targeting Python 3.11+ and vLLM's OpenAI-compatible API. Every code example includes error handling, timeout configuration, and graceful shutdown semantics.

---

## 1. Migrating from requests to httpx.AsyncClient

### The minimal migration path

The `httpx` library was designed as a drop-in async replacement for `requests`, but several API differences will bite you during migration. The most consequential: **httpx defaults to a 5-second timeout** (requests defaults to infinite), **does not auto-follow redirects**, and **does not raise on 4xx/5xx status codes** without an explicit call.

| Behavior | `requests` | `httpx` |
|---|---|---|
| Timeout default | None (hangs forever) | **5 seconds** |
| Redirects | Auto-follow | Must set `follow_redirects=True` |
| Status errors | `response.raise_for_status()` optional | Same, but also `response.is_success` replaces `response.ok` |
| Raw bytes param | `data=b"bytes"` | `content=b"bytes"` |
| Exceptions | `requests.RequestException` | `httpx.RequestError` (network) + `httpx.HTTPStatusError` (status) |

**Before** (synchronous, blocks the event loop):

```python
import requests

session = requests.Session()
session.headers.update({"Authorization": "Bearer token"})

response = session.post(
    "http://vllm:8000/v1/chat/completions",
    json={"model": "my-model", "messages": [{"role": "user", "content": "Hello"}]},
    timeout=30,
)
result = response.json()
```

**After** (async, non-blocking):

```python
import httpx

# Created once at app startup via lifespan (see Section 5)
client = httpx.AsyncClient(
    base_url="http://vllm:8000",
    headers={"Authorization": "Bearer token"},
    timeout=httpx.Timeout(30.0, connect=5.0),
)

response = await client.post(
    "/v1/chat/completions",
    json={"model": "my-model", "messages": [{"role": "user", "content": "Hello"}]},
)
response.raise_for_status()
result = response.json()
```

The migration checklist: replace `session.post()` → `await client.post()`, update exception handling from `requests.RequestException` → `httpx.RequestError`, set explicit timeouts with `httpx.Timeout`, and ensure every call is `await`ed inside an `async def` endpoint.

### Why asyncio.run_in_executor is only a stop-gap

Wrapping synchronous `requests.post()` in `asyncio.run_in_executor(None, ...)` unblocks the event loop by offloading the call to a thread pool, and it works as a 1–2 sprint bridge:

```python
import asyncio
import functools

async def call_vllm(payload: dict) -> dict:
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        functools.partial(session.post, "http://vllm:8000/v1/chat/completions",
                          json=payload, timeout=30),
    )
    return response.json()
```

The problem is scalability. The default `ThreadPoolExecutor` caps at **~36 workers** (`min(32, os.cpu_count() + 4)`). Each concurrent LLM request holds a thread idle for the entire inference duration — often seconds. At 50+ concurrent users, threads are exhausted and requests queue. Native async httpx uses coroutines at **~few KB each** versus **~8 MB per thread stack**, and avoids GIL contention entirely. Use `run_in_executor` only if you need an immediate fix before the httpx migration is complete.

### Connection pool tuning for vLLM workloads

A single `httpx.AsyncClient` instance should be shared across the entire application. Creating a client per request defeats connection pooling and forces a new TCP handshake every time — the httpx docs explicitly warn against this pattern.

```python
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        base_url="http://vllm:8000",
        timeout=httpx.Timeout(
            connect=5.0,     # TCP handshake — vLLM is on local network
            read=120.0,      # LLM inference with long sequences can take minutes
            write=10.0,      # prompt payloads are small JSON
            pool=10.0,       # fail fast if pool is exhausted
        ),
        limits=httpx.Limits(
            max_connections=200,            # align with vLLM's --max-num-seqs
            max_keepalive_connections=50,   # warm connections for reuse
            keepalive_expiry=30.0,          # longer than default 5s
        ),
    )
    yield
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)
```

The **`read` timeout** is the critical parameter for LLM workloads. Set it to match your maximum expected generation time (60–300 seconds depending on `max_tokens` and model size). Set `max_connections` to match or slightly exceed vLLM's `--max-num-seqs` parameter so you don't bottleneck at the HTTP layer. Always call `aclose()` on shutdown — failing to do so leaks sockets and triggers `ResourceWarning`.

### Benchmarking the migration

Measure three things: **p50/p95/p99 latency** (tail latency matters most for LLM serving), **throughput in requests/second**, and **time-to-first-token (TTFT)** for streaming endpoints.

```python
import asyncio, time, httpx

async def benchmark(url: str, payload: dict, n: int, concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def single():
        async with sem:
            t0 = time.perf_counter()
            resp = await client.post(url, json=payload)
            latencies.append(time.perf_counter() - t0)
            return resp.status_code

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=5.0),
        limits=httpx.Limits(max_connections=concurrency),
    ) as client:
        t_wall = time.perf_counter()
        results = await asyncio.gather(*[single() for _ in range(n)],
                                        return_exceptions=True)
        elapsed = time.perf_counter() - t_wall

    latencies.sort()
    errors = sum(1 for r in results if isinstance(r, Exception))
    print(f"Throughput: {n/elapsed:.1f} req/s | "
          f"p50={latencies[len(latencies)//2]*1000:.0f}ms | "
          f"p99={latencies[int(len(latencies)*0.99)]*1000:.0f}ms | "
          f"Errors: {errors}")
```

For external load testing, `hey` or `locust` work well. Run identical load profiles against both the `run_in_executor` and native `httpx` implementations on the same vLLM backend. The async advantage becomes visible at **50+ concurrent requests**, where thread exhaustion becomes the bottleneck.

### The final target — openai.AsyncOpenAI

The `openai` Python SDK uses `httpx.AsyncClient` internally, so migrating to `AsyncOpenAI` is a natural final step that adds typed response models, automatic retries, and built-in SSE parsing:

```python
import httpx
from openai import AsyncOpenAI
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Custom httpx client for tuned connection pooling
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50,
                            keepalive_expiry=30.0),
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0),
    )
    app.state.llm = AsyncOpenAI(
        base_url="http://vllm:8000/v1",
        api_key="EMPTY",               # vLLM doesn't require a real key
        http_client=http_client,        # inject tuned pool
        max_retries=2,
    )
    yield
    await app.state.llm.close()         # closes the httpx client too

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(request: Request):
    response = await request.app.state.llm.chat.completions.create(
        model="my-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=200,
    )
    return {"reply": response.choices[0].message.content}
```

Three known pitfalls with `AsyncOpenAI` to watch for. First, **pool exhaustion under high concurrency** — the default httpx pool is only 100 connections; pass a custom `http_client` with higher `max_connections`. Second, **stream connections not returned to the pool** if you bail early — always fully consume the stream or call `await stream.response.aclose()`. Third, **stale connections after 24+ hours** — set `keepalive_expiry` on the httpx client. Per-request timeout overrides are available via `client.with_options(timeout=300.0).chat.completions.create(...)`.

---

## 2. Shared state isolation in singleton services

### How the bug manifests

When `AgentService` is a singleton via `@lru_cache` with mutable instance variables like `_current_step` and `_current_request_id`, **every concurrent async request shares the same object**. Python's asyncio uses cooperative multitasking on a single thread, so any `await` point can yield control to another request's coroutine, which overwrites those shared variables. This is not a threading race condition — it is an **interleaving corruption** bug unique to async code.

The Python docs explicitly warn: `lru_cache` should not be used for "functions that need to create distinct mutable objects on each call." Additionally, `lru_cache` has no teardown hook for cleanup, making it unsuitable for async resources.

### Four approaches compared

**Approach A — Request-scoped instances via Depends()**: Create a lightweight `AgentService` per request. Expensive shared resources (LLM client, retriever) are injected via constructor:

```python
from fastapi import Depends, Request
from typing import Annotated

class AgentService:
    def __init__(self, llm_client: AsyncOpenAI):
        self._current_step: int = 0
        self._request_id: str = str(uuid.uuid4())
        self._llm = llm_client  # shared, immutable reference

    async def run_step(self, prompt: str) -> dict:
        self._current_step += 1
        result = await self._llm.chat.completions.create(...)
        return {"step": self._current_step, "request_id": self._request_id}

def get_agent_service(request: Request) -> AgentService:
    return AgentService(llm_client=request.app.state.llm)

AgentSvc = Annotated[AgentService, Depends(get_agent_service)]
```

**Approach B — Stateless singleton with parameter passing**: Keep `AgentService` as a `@lru_cache` singleton, but move mutable state into a `RequestContext` dataclass passed explicitly:

```python
from dataclasses import dataclass, field

@dataclass
class RequestContext:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_step: int = 0

    def next_step(self) -> int:
        self.current_step += 1
        return self.current_step

class AgentService:  # Stateless singleton — safe to cache
    def __init__(self, llm: AsyncOpenAI):
        self._llm = llm

    async def run_step(self, ctx: RequestContext, prompt: str) -> dict:
        step = ctx.next_step()
        result = await self._llm.chat.completions.create(...)
        return {"step": step, "request_id": ctx.request_id}
```

**Approach C — contextvars.ContextVar**: Use `ContextVar` to store per-request mutable state. Each asyncio task gets its own copy of the context, so `asyncio.gather` and `TaskGroup` automatically isolate child tasks:

```python
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
current_step_var: ContextVar[int] = ContextVar("current_step", default=0)
```

**Approach D — Hybrid (recommended)**: Combine a singleton for shared resources with `ContextVar` for per-request state. This is the production pattern:

```python
from __future__ import annotations
from contextvars import ContextVar
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from starlette.types import ASGIApp, Receive, Scope, Send
import uuid

@dataclass
class RequestState:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_step: int = 0
    def next_step(self) -> int:
        self.current_step += 1
        return self.current_step

_request_state: ContextVar[RequestState | None] = ContextVar("request_state", default=None)

def get_request_state() -> RequestState:
    state = _request_state.get()
    if state is None:
        raise RuntimeError("RequestState not initialized")
    return state

# Pure ASGI middleware — NOT BaseHTTPMiddleware (see pitfall below)
class RequestStateMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = _request_state.set(RequestState())
        try:
            await self.app(scope, receive, send)
        finally:
            _request_state.reset(token)

class AgentService:
    """Singleton. Shared resources via constructor, per-request state via ContextVar."""
    def __init__(self, llm: AsyncOpenAI, retriever: HybridRetriever):
        self._llm = llm
        self._retriever = retriever

    async def run_step(self, prompt: str) -> dict:
        state = get_request_state()
        step = state.next_step()
        docs = await self._retriever.search(prompt)
        response = await self._llm.chat.completions.create(...)
        return {"step": step, "request_id": state.request_id}
```

| Criterion | Request-scoped DI | Parameter passing | ContextVar | Hybrid |
|---|---|---|---|---|
| Async safety | ✅ New object per request | ✅ Explicit data flow | ✅ Designed for asyncio | ✅ Best of both |
| Performance overhead | ⚠️ Object creation per request | ✅ Minimal | ✅ ~50ns per get/set | ✅ Shared resources + light state |
| Deep call stacks | ⚠️ Must thread through layers | ❌ Every function needs `ctx` | ✅ Accessible anywhere | ✅ Accessible anywhere |
| Testing | ✅ `dependency_overrides` | ✅ Pass test data | ⚠️ Must manually set/reset | ✅ Good with helpers |

### Critical ContextVar pitfall with Starlette middleware

**Never use `BaseHTTPMiddleware` with `ContextVar`.** Starlette's `BaseHTTPMiddleware` spawns internal tasks, which breaks ContextVar propagation — values set in endpoints are invisible to the middleware after `call_next()`. The Starlette docs confirm this is a known limitation. Always use **pure ASGI middleware** (as shown above) or set ContextVars in a **FastAPI dependency** instead:

```python
async def init_request_context():
    token = _request_state.set(RequestState())
    yield
    _request_state.reset(token)
```

Also note that `asyncio.gather()` and `TaskGroup` give each child task a **copy** of the parent's context at creation time. Changes in child tasks do not propagate back to the parent. This is usually the desired behavior for request isolation.

---

## 3. Async ToolRunner

### Converting sync tools to async

Use `asyncio.to_thread()` (Python 3.9+) for I/O-bound sync tools — it's the modern replacement for `run_in_executor` that automatically propagates `contextvars`:

```python
import asyncio, inspect
from functools import partial
from concurrent.futures import ProcessPoolExecutor

async def invoke_tool(func, arguments: dict, process_pool=None):
    if inspect.iscoroutinefunction(func):
        return await func(**arguments)
    if getattr(func, '_cpu_bound', False) and process_pool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(process_pool, partial(func, **arguments))
    return await asyncio.to_thread(func, **arguments)
```

The critical caveat: **cancelling `asyncio.to_thread()` does not stop the underlying thread**. The `await` is cancelled, but the sync function continues running until it returns naturally. For sync tools that may run for a long time, use cooperative cancellation with a `threading.Event` checked inside the function.

### gather vs TaskGroup for tool execution

For LLM agent frameworks, **`asyncio.gather(return_exceptions=True)` is usually the right choice** over `TaskGroup`. The reason: when one tool fails, the LLM still needs results from all other tools to reason about next steps. `TaskGroup` cancels all remaining tasks on the first failure, which loses partial results.

`TaskGroup` is better when tools are interdependent and one failure should abort the batch (e.g., a multi-step pipeline where step 2 depends on step 1).

### Production AsyncToolRunner implementation

```python
from __future__ import annotations
import asyncio, inspect, logging, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Callable

logger = logging.getLogger(__name__)

class ToolCallStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]

@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    status: ToolCallStatus
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0

@dataclass
class ToolRunnerConfig:
    max_concurrency: int = 5
    default_timeout: float = 30.0
    tool_timeouts: dict[str, float] = field(default_factory=dict)

class AsyncToolRunner:
    """Runs LLM tool calls concurrently with semaphore limiting and per-tool timeouts."""

    def __init__(
        self,
        registry: dict[str, Callable],
        config: ToolRunnerConfig | None = None,
        process_pool: ProcessPoolExecutor | None = None,
    ):
        self._registry = registry
        self._config = config or ToolRunnerConfig()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrency)
        self._process_pool = process_pool

    async def run(self, tool_calls: list[ToolCall]) -> dict[str, ToolResult]:
        if not tool_calls:
            return {}
        # gather is preferred: we want ALL results even if some tools fail
        results = await asyncio.gather(
            *[self._execute(tc) for tc in tool_calls],
            return_exceptions=False,  # safe because _execute never raises
        )
        return {r.call_id: r for r in results}

    async def _execute(self, tc: ToolCall) -> ToolResult:
        """Execute one tool. Never raises — always returns a ToolResult."""
        start = time.monotonic()
        timeout = self._config.tool_timeouts.get(tc.tool_name, self._config.default_timeout)

        try:
            async with self._semaphore:
                func = self._registry.get(tc.tool_name)
                if func is None:
                    return ToolResult(tc.call_id, tc.tool_name, ToolCallStatus.ERROR,
                                     error=f"Unknown tool: {tc.tool_name}",
                                     duration_ms=self._elapsed(start))
                try:
                    async with asyncio.timeout(timeout):
                        output = await self._invoke(func, tc.arguments)
                except TimeoutError:
                    return ToolResult(tc.call_id, tc.tool_name, ToolCallStatus.TIMEOUT,
                                     error=f"Timed out after {timeout}s",
                                     duration_ms=self._elapsed(start))

                return ToolResult(tc.call_id, tc.tool_name, ToolCallStatus.SUCCESS,
                                 output=output, duration_ms=self._elapsed(start))

        except asyncio.CancelledError:
            return ToolResult(tc.call_id, tc.tool_name, ToolCallStatus.CANCELLED,
                             duration_ms=self._elapsed(start))
        except Exception as e:
            logger.exception(f"Tool '{tc.tool_name}' failed")
            return ToolResult(tc.call_id, tc.tool_name, ToolCallStatus.ERROR,
                             error=f"{type(e).__name__}: {e}",
                             duration_ms=self._elapsed(start))

    async def _invoke(self, func: Callable, args: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(func):
            return await func(**args)
        if getattr(func, '_cpu_bound', False) and self._process_pool:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._process_pool, partial(func, **args))
        return await asyncio.to_thread(func, **args)

    @staticmethod
    def _elapsed(start: float) -> float:
        return (time.monotonic() - start) * 1000
```

The flow for each tool call: **semaphore acquisition** (limits concurrent executions to `max_concurrency`) → **timeout wrapping** via `asyncio.timeout()` → **smart dispatch** (native async, `to_thread` for sync I/O, `ProcessPoolExecutor` for CPU-bound). Results are mapped back by `call_id` so the agent framework can match them to the LLM's tool call requests.

Two key pitfalls to avoid. **Semaphore starvation**: if 5 long-running tools hold all permits, subsequent tools block indefinitely — always combine the semaphore with per-tool timeouts. **Swallowing `CancelledError`**: always re-raise it after cleanup, or `TaskGroup` and `asyncio.timeout()` will break.

---

## 4. SSE streaming and client disconnect handling

### How sse-starlette detects disconnects

`sse-starlette`'s `EventSourceResponse` uses a three-layer disconnect detection architecture. **Layer 1 (passive)**: a background task awaits `{"type": "http.disconnect"}` from the ASGI receive channel and sets `self.active = False`. **Layer 2 (proactive)**: your code calls `await request.is_disconnected()` to check before expensive operations. **Layer 3 (reactive)**: when the background task completes, it cancels the anyio task group's cancel scope, propagating `asyncio.CancelledError` into your generator.

Both `request.is_disconnected()` and `CancelledError` are needed — they are complementary, not alternatives. Polling `is_disconnected()` exits the loop faster (before the next yield), while `CancelledError` is the guaranteed fallback if polling is missed.

### Cancelling upstream LLM requests on disconnect

When using `httpx.AsyncClient.stream()` as an async context manager, cancellation propagates cleanly: `CancelledError` is raised at the next `await` point, the context manager's `__aexit__` calls `response.aclose()`, and the connection returns to the pool. One important nuance: `Response.aclose()` performs client-side cleanup but may **not** send a TCP FIN due to HTTP/1.1 keep-alive. vLLM may continue generating until it notices the connection is gone.

### Production SSE endpoint

```python
import asyncio, json, logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

logger = logging.getLogger(__name__)
_stream_semaphore = asyncio.Semaphore(50)  # max concurrent SSE streams

class ChatRequest(BaseModel):
    model: str = "default-model"
    messages: list[dict]
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = True

@app.post("/v1/chat/completions")
async def stream_completions(chat_req: ChatRequest, request: Request):
    async with _stream_semaphore:
        return EventSourceResponse(
            _stream_tokens(request, request.app.state.http_client, chat_req),
            ping=15,            # keep-alive through proxies
            send_timeout=30,    # detect clients that stop reading
        )

async def _stream_tokens(
    request: Request,
    client: httpx.AsyncClient,
    chat_req: ChatRequest,
) -> AsyncGenerator[dict, None]:
    tokens_sent = 0
    try:
        if await request.is_disconnected():
            return

        async with client.stream(
            "POST", "/v1/chat/completions",
            json=chat_req.model_dump(),
        ) as upstream:
            if upstream.status_code != 200:
                body = await upstream.aread()
                yield {"event": "error",
                       "data": json.dumps({"error": body.decode()})}
                return

            async for line in upstream.aiter_lines():
                if await request.is_disconnected():
                    logger.info(f"Client disconnected after {tokens_sent} tokens")
                    break

                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    yield {"data": "[DONE]"}
                    break
                yield {"data": data}
                tokens_sent += 1

    except asyncio.CancelledError:
        logger.info(f"Stream cancelled after {tokens_sent} tokens")
        raise  # MUST re-raise — context manager handles upstream aclose()

    except httpx.TimeoutException as e:
        yield {"event": "error",
               "data": json.dumps({"error": "LLM request timed out"})}

    except httpx.HTTPError as e:
        yield {"event": "error",
               "data": json.dumps({"error": f"Upstream error: {e}"})}

    finally:
        logger.debug(f"Generator finalized (tokens_sent={tokens_sent})")
```

### Memory leak prevention checklist

The most common leak sources with SSE connections are **httpx responses left open** (always use `async with client.stream()`), **clients that stop reading** (set `send_timeout=30`), and **generators that swallow `CancelledError`** (always re-raise it).

Since Python 3.8, `asyncio.CancelledError` is a `BaseException` subclass, so `except Exception:` blocks will not accidentally catch it. This means your error handling in generators is safe by default — but always include an explicit `except asyncio.CancelledError: raise` for clarity and future-proofing.

Configure `EventSourceResponse(ping=15)` to send keep-alive comments every 15 seconds. This both detects stale connections and prevents reverse proxies from timing out the SSE connection. Add `X-Accel-Buffering: no` to prevent Nginx from buffering the event stream.

---

## 5. FastAPI lifespan for service initialization

### Why lifespan replaces @lru_cache for async resources

`@lru_cache` is synchronous — you cannot `await` inside it. It has no shutdown hook for cleanup. Resources are initialized lazily on the first request (causing a cold-start latency penalty). And `lru_cache` is not guaranteed to initialize exactly once under concurrent access. The FastAPI lifespan context manager solves all of these problems.

| Aspect | `@lru_cache` | Lifespan |
|---|---|---|
| Async initialization | ❌ Cannot `await` | ✅ Full async support |
| Cleanup on shutdown | ❌ No hook | ✅ Code after `yield` |
| First-request latency | ⚠️ Cold start | ✅ All requests equally fast |
| Resource leaks | ⚠️ Common | ✅ Explicit cleanup |

### Complete lifespan with dependency injection

```python
from __future__ import annotations
import asyncio, logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Request
from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.services.retriever import HybridRetriever

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # 1. HTTP client with tuned pool
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50,
                            keepalive_expiry=30.0),
    )
    app.state.http_client = http_client

    # 2. LLM client pointed at vLLM
    llm_client = AsyncOpenAI(
        base_url=settings.VLLM_BASE_URL,
        api_key="EMPTY",
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0),
        ),
        max_retries=2,
    )
    app.state.llm = llm_client

    # 3. Retriever (depends on http_client)
    try:
        retriever = await HybridRetriever.create(
            http_client=http_client,
            bm25_index_path=settings.BM25_INDEX_PATH,
        )
        app.state.retriever = retriever
    except Exception:
        logger.exception("Failed to initialize retriever")
        await llm_client.close()
        await http_client.aclose()
        raise  # prevent app from starting in broken state

    logger.info("All services initialized")
    yield

    # Shutdown in reverse dependency order
    for name, closer in [
        ("retriever",   retriever.close()),
        ("llm_client",  llm_client.close()),
        ("http_client", http_client.aclose()),
    ]:
        try:
            await asyncio.wait_for(closer, timeout=5.0)
            logger.info(f"Closed {name}")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout closing {name}")
        except Exception:
            logger.exception(f"Error closing {name}")

app = FastAPI(title="RAG API", lifespan=lifespan)

# --- Typed dependency accessors ---
def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client

def get_llm(request: Request) -> AsyncOpenAI:
    return request.app.state.llm

def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever

HttpClient = Annotated[httpx.AsyncClient, Depends(get_http_client)]
LLM        = Annotated[AsyncOpenAI,       Depends(get_llm)]
Retriever  = Annotated[HybridRetriever,   Depends(get_retriever)]

# --- Routes use typed annotations ---
@app.post("/query")
async def query(q: str, retriever: Retriever, llm: LLM):
    docs = await retriever.search(q, top_k=5)
    completion = await llm.chat.completions.create(
        model="my-model",
        messages=[{"role": "system", "content": "\n".join(d.text for d in docs)},
                  {"role": "user", "content": q}],
    )
    return {"answer": completion.choices[0].message.content}
```

The migration from `@lru_cache` is straightforward: move resource creation into lifespan, store on `app.state`, and change dependency functions from `@lru_cache` factories to `request.app.state` accessors. Route signatures remain unchanged — `Depends(get_retriever)` still works. Keep `@lru_cache` for simple synchronous config objects like `Settings()` — it's still the recommended pattern for that use case per the FastAPI docs.

### Testing with lifespan

The `TestClient` **must** be used as a context manager to trigger lifespan events. Without `with`, startup and shutdown code never runs:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def client():
    with TestClient(app) as c:  # lifespan runs here
        yield c

@pytest.fixture
def mock_client():
    """Override dependencies for unit tests."""
    app.dependency_overrides[get_retriever] = lambda: MagicMock(
        search=AsyncMock(return_value=[]))
    app.dependency_overrides[get_llm] = lambda: MagicMock()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

For async tests with `httpx.AsyncClient`, use `asgi-lifespan`:

```python
from asgi_lifespan import LifespanManager

@pytest.fixture
async def async_client():
    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client
```

### Docker and signal handling

Uvicorn handles both **SIGTERM** and **SIGINT** for graceful shutdown in recent versions. In Docker, ensure your process receives signals by either using `tini` as PID 1, adding `STOPSIGNAL SIGINT` to the Dockerfile, or using exec-form `CMD ["python", "-m", "uvicorn", ...]`. Set `--timeout-graceful-shutdown 10` in uvicorn to allow in-flight SSE streams to complete before forced exit.

---

## Conclusion

The five changes described here form a coherent migration path. **Start with the lifespan pattern** (Section 5) — it unblocks everything else by providing proper async resource initialization and cleanup. Then **migrate the HTTP client** (Section 1) from `requests` to `httpx.AsyncClient`, stored on `app.state`. **Fix shared state** (Section 2) using the hybrid pattern: singleton services with `ContextVar` for per-request mutable state. **Convert the ToolRunner** (Section 3) to use `asyncio.gather` with per-tool semaphore limiting and `asyncio.timeout`. Finally, **wire up SSE disconnect handling** (Section 4) with the three-layer approach: proactive `is_disconnected()` polling, passive sse-starlette detection, and reactive `CancelledError` cleanup.

The single most impactful change is eliminating the synchronous `requests.post()` call — it currently serializes all LLM requests through the event loop. The second most impactful is fixing the singleton state corruption, which is a silent data-integrity bug. Both are prerequisite to serving concurrent users reliably. The `asyncio.run_in_executor` wrapper is an acceptable one-sprint bridge for the HTTP client migration, but should not be treated as a permanent solution given its ~36-thread concurrency ceiling.