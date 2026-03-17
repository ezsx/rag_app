# vLLM wins for production ReAct agents on V100

**vLLM is the right choice for your setup.** Its robust structured JSON decoding via xgrammar, production-grade reliability, and the Qwen team's explicit endorsement make it the stronger option for a latency-sensitive ReAct agent — even in a single-user scenario where Ollama's simplicity is tempting. The migration from llama.cpp to vLLM's OpenAI-compatible API will also unblock your async FastAPI event loop, since all inference moves to HTTP calls against a separate process. Below is everything you need: configuration, code, and the reasoning behind each decision.

---

## Why vLLM over Ollama for this specific workload

The decision hinges on **structured output reliability**, not throughput. Both servers deliver comparable single-user latency (~150ms TTFT baseline), but they diverge sharply on constrained JSON generation — the feature your ReAct agent depends on most.

**vLLM's xgrammar backend** uses a pushdown automaton that pre-compiles token masks from your JSON schema, then caches them. For a ReAct agent reusing the same action schema on every cycle, overhead after warmup is **<1ms per token**. vLLM also supports `response_format` with `json_schema` through the standard OpenAI API, `guided_json` via `extra_body`, regex constraints, and EBNF grammars — giving you multiple fallback strategies if one approach hits edge cases.

**Ollama's grammar-based approach** converts JSON schemas to GBNF grammars (inherited from llama.cpp). It works, but deeply nested or recursive schemas can cause issues, and Ollama does not validate the final output against the schema if generation stops mid-JSON. For a production ReAct loop where a single malformed JSON breaks the Thought → Action → Observation cycle, this gap matters.

Other factors favoring vLLM:

- **Qwen team endorsement**: "We recommend you trying vLLM for your deployment of Qwen" (qwen.readthedocs.io). The chat template, sampling parameters, and tool calling format are all tested against vLLM first.
- **Native tool calling**: Launch with `--enable-auto-tool-choice --tool-call-parser hermes` to get Qwen2.5's Hermes-style `<tool_call>` format parsed automatically.
- **OpenAI API completeness**: vLLM's `/v1/chat/completions` endpoint is more complete than Ollama's, including `stream_options`, `logprobs`, and the newer Responses API.
- **Prefix caching**: Your ReAct agent reuses the same system prompt on every call. vLLM caches the KV states for shared prefixes, eliminating redundant prefill computation.

Ollama remains a valid fallback if vLLM proves too heavy operationally. Since both expose OpenAI-compatible APIs, migration is a one-line `base_url` change.

---

## V100 SXM2 32GB: what to expect and what to watch for

The Tesla V100 is viable but has three constraints you must account for: **no bfloat16, no FlashAttention, and xformers deprecation in vLLM v0.17+**.

**Performance expectations for Qwen2.5-7B FP16 at batch=1**: decode throughput of **~30–45 tokens/sec**, first-token latency of **100–300ms** depending on prompt length. The V100 SXM2's 900 GB/s HBM2 bandwidth is the bottleneck at batch=1 — a 14GB FP16 model theoretically caps at ~64 tok/s at 100% memory bandwidth utilization, and realistic MBU is 50–70%. This is roughly **2–2.5× slower than an A100** and adequate for an interactive single-user agent.

**Memory budget is comfortable.** Qwen2.5-7B FP16 weights consume ~14GB. The KV cache for 10,000 tokens is remarkably small — only **~0.5GB** — thanks to Grouped Query Attention (4 KV heads instead of 32). Adding an embedding model (~0.5GB) and reranker (~0.6GB) brings the total to ~18GB, leaving **14GB headroom** on your 32GB card. Do not quantize to INT8/INT4: the V100 lacks optimized INT4 Tensor Core kernels, dequantization overhead can negate bandwidth savings, and you have ample VRAM.

**The critical compatibility issue**: vLLM v0.17.0 (March 7, 2026) deprecated the xformers attention backend, which is the *only* backend that works on V100 (compute capability 7.0). FlashAttention 2 requires SM80+ (Ampere). **Pin to vLLM v0.15.1 or v0.16.x**, where xformers is fully supported and auto-selected on V100. If you must use v0.17+, test the Triton attention backend (pure-Triton fallback that should work on SM70) — but this is less battle-tested on Volta hardware.

You must always pass `--dtype half` (float16). The V100 does not support bfloat16 (requires compute capability ≥8.0), and vLLM will throw a `ValueError` without this flag. Qwen2.5 models are trained in BF16 but convert cleanly to FP16 for inference with no meaningful quality degradation.

---

## Docker Compose configuration for vLLM

```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:v0.15.1  # Pinned for V100 xformers support
    ports:
      - "8000:8000"
    volumes:
      - ${HF_CACHE_DIR:-~/.cache/huggingface}:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
      - VLLM_ATTENTION_BACKEND=XFORMERS
    command:
      - "--model"
      - "Qwen/Qwen2.5-7B-Instruct"
      - "--dtype"
      - "half"
      - "--max-model-len"
      - "8192"
      - "--gpu-memory-utilization"
      - "0.85"
      - "--max-num-seqs"
      - "4"
      - "--enable-auto-tool-choice"
      - "--tool-call-parser"
      - "hermes"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s  # Model loading takes time

  # Qwen2.5-3B on CPU via Ollama (Query Planner)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
      - NVIDIA_VISIBLE_DEVICES=  # Force CPU-only
      - CUDA_VISIBLE_DEVICES=    # Force CPU-only
      - OLLAMA_NUM_PARALLEL=1
      - OLLAMA_KEEP_ALIVE=30m
    restart: unless-stopped

volumes:
  ollama_data:
```

**Key configuration rationale**:

- **`--max-model-len 8192`** — Qwen2.5's default is 32,768 tokens, which over-allocates KV cache. For a ReAct agent, 8K is generous (system prompt + multi-turn history + tool outputs). Saves ~3GB of KV cache pre-allocation versus 32K.
- **`--gpu-memory-utilization 0.85`** — Reserves 85% of 32GB (~27.2GB) for vLLM. The remaining ~4.8GB stays free for the embedding model and reranker running in a separate process. Increase to 0.90–0.95 if no other GPU models are co-located.
- **`--max-num-seqs 4`** — Limits concurrent sequences. For single-user, even `2` suffices. Lower values reduce scheduling overhead and memory pressure.
- **`--enable-auto-tool-choice --tool-call-parser hermes`** — Activates Qwen2.5's native tool calling with the Hermes-style `<tool_call>` XML wrapper format.
- **Ollama for the 3B CPU model** — Ollama excels at simple CPU inference. Set `NVIDIA_VISIBLE_DEVICES=` to force CPU-only mode. After starting, run `docker exec ollama ollama pull qwen2.5:3b` to download the model.

---

## Python LLMClient abstraction with async streaming

This client works identically with vLLM, Ollama, and OpenAI — switching providers is a `base_url` change:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_connect: float = 5.0
    timeout_read: float = 300.0
    max_retries: int = 1
    default_temperature: float = 0.7
    default_max_tokens: int = 1024


class LLMClient:
    """Async LLM client compatible with any OpenAI-compatible API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=httpx.Timeout(
                config.timeout_read,
                connect=config.timeout_connect,
            ),
            max_retries=config.max_retries,
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming generation. Returns the full response text."""
        kwargs = self._build_kwargs(
            messages, json_schema=json_schema,
            temperature=temperature, max_tokens=max_tokens,
        )
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Streaming generation. Yields tokens as they arrive."""
        kwargs = self._build_kwargs(
            messages, json_schema=json_schema,
            temperature=temperature, max_tokens=max_tokens,
            stream=True,
        )
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> dict:
        """Generate constrained JSON matching the given schema."""
        raw = await self.generate(
            messages,
            json_schema=schema,
            temperature=temperature,
        )
        return json.loads(raw)

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.default_temperature,
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "stream": stream,
        }
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "output"),
                    "strict": True,
                    "schema": json_schema,
                },
            }
        return kwargs

    async def close(self):
        await self._client.close()


# --- Factory ---

def create_agent_client() -> LLMClient:
    """Main 7B agent on vLLM."""
    return LLMClient(LLMConfig(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-7B-Instruct",
        api_key="EMPTY",
        default_temperature=0.7,
    ))

def create_planner_client() -> LLMClient:
    """3B query planner on Ollama (CPU)."""
    return LLMClient(LLMConfig(
        base_url="http://localhost:11434/v1/",
        model="qwen2.5:3b",
        api_key="ollama",
        default_temperature=0.0,
        timeout_read=60.0,
    ))
```

---

## ReAct agent with SSE streaming through FastAPI

```python
from __future__ import annotations

import json
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
agent_client = create_agent_client()

REACT_SYSTEM_PROMPT = """You are a ReAct agent. On each turn, output EXACTLY one
JSON object with the following schema, and nothing else:
{"thought": "...", "action": "...", "action_input": "..."}
When you have the final answer, use action="finish" and put the answer in action_input."""


class ReActAction(BaseModel):
    thought: str
    action: str
    action_input: str


# --- Structured generation (non-streaming, for tool dispatch) ---

async def react_step(
    conversation: list[dict[str, str]],
) -> ReActAction:
    """Single ReAct step with guaranteed valid JSON."""
    result = await agent_client.generate_structured(
        messages=conversation,
        schema=ReActAction.model_json_schema(),
        temperature=0.0,
    )
    return ReActAction(**result)


# --- Streaming endpoint for the final answer ---

async def stream_final_answer(user_query: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_query},
    ]
    async for token in agent_client.generate_stream(messages):
        yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(request: dict):
    return StreamingResponse(
        stream_final_answer(request["message"]),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Prevents nginx buffering
        },
    )


# --- Full ReAct loop example ---

TOOLS = {
    "search": lambda q: f"Search results for: {q}",
    "calculate": lambda expr: str(eval(expr)),
}

async def run_react_agent(user_query: str, max_steps: int = 5) -> str:
    conversation = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for _ in range(max_steps):
        action = await react_step(conversation)

        if action.action == "finish":
            return action.action_input

        # Execute tool
        tool_fn = TOOLS.get(action.action)
        observation = tool_fn(action.action_input) if tool_fn else "Unknown tool"

        # Append to conversation
        conversation.append({"role": "assistant", "content": action.model_dump_json()})
        conversation.append({"role": "user", "content": f"Observation: {observation}"})

    return "Max steps reached."
```

---

## Structured output: the belt-and-suspenders approach

For a ReAct agent, JSON reliability is non-negotiable. Use **three layers of defense**:

**Layer 1 — Constrained decoding (xgrammar).** Pass `response_format` with `json_schema` on every call. vLLM's xgrammar backend masks invalid tokens at each decoding step, making it structurally impossible to produce malformed JSON. After the first request warms the grammar cache, overhead is negligible (<1ms/token). This is the GBNF equivalent from llama.cpp but more performant.

**Layer 2 — Model-native capability.** Qwen2.5-7B scores **95% on simple function calls** and **90% on multiple functions** in the Berkeley Function Calling Leaderboard. It was specifically optimized for "more reliable generation of structured outputs, particularly in JSON format." With constrained decoding active, the model's natural JSON tendency and the structural constraint reinforce each other.

**Layer 3 — Pydantic validation with retry.** Always validate the parsed JSON against your Pydantic model. On the rare occasion that generation truncates (hitting `max_tokens` mid-object), catch the `ValidationError` and retry with a higher token limit:

```python
async def safe_react_step(conversation, retries=2):
    for attempt in range(retries + 1):
        try:
            result = await agent_client.generate_structured(
                messages=conversation,
                schema=ReActAction.model_json_schema(),
                temperature=0.0,
                max_tokens=512 * (attempt + 1),  # Increase on retry
            )
            return ReActAction(**result)
        except (json.JSONDecodeError, ValidationError):
            if attempt == retries:
                raise
```

One caveat: Qwen2.5-7B's **multi-turn agentic score is only 13.5%** on the BFCL leaderboard. This means the model can struggle with complex multi-step tool-use chains where context management matters. Constrained decoding guarantees valid *syntax* but not valid *semantics* — the model might call the wrong tool or pass incorrect arguments. Mitigate this with clear system prompts, concise observation formatting, and keeping conversation history trimmed.

---

## Optimal V100 parameters at a glance

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--dtype` | `half` | **Mandatory** — V100 lacks bfloat16 (SM70) |
| `--max-model-len` | `8192` | Sufficient for ReAct; saves ~3GB KV cache vs 32K |
| `--gpu-memory-utilization` | `0.85` | Leaves ~5GB for embedding/reranker co-location |
| `--max-num-seqs` | `4` | Single-user; reduces scheduling overhead |
| `--enable-auto-tool-choice` | — | Activates native Qwen2.5 tool parsing |
| `--tool-call-parser` | `hermes` | Hermes-style `<tool_call>` XML format |
| vLLM version | **v0.15.1** | Last fully xformers-supported release for V100 |
| Docker image | `vllm/vllm-openai:v0.15.1` | Pinned for compatibility |
| Attention backend | xformers (auto) | Only viable backend on SM70 hardware |
| Quantization | None (FP16) | 14GB model fits in 32GB; quantization adds overhead without memory pressure |
| CUDA graphs | On (default) | Try `--enforce-eager` only if you hit errors |
| Expected decode speed | **30–45 tok/s** | Memory-bandwidth bound at batch=1 on V100 |
| Expected TTFT | **100–300ms** | Depends on prompt length; no FlashAttention penalty |
| KV cache (10K ctx) | **~0.5GB** | Qwen2.5's GQA (4 KV heads) is very cache-efficient |

---

## Conclusion

**vLLM + Qwen2.5-7B on V100 SXM2 32GB is a solid production stack for single-user ReAct agents.** The combination of xgrammar-backed structured decoding, Hermes-style native tool calling, and the standard OpenAI API eliminates the two biggest pain points of your current setup: blocking synchronous calls and unreliable JSON output. Pin to vLLM v0.15.1 for reliable xformers support on Volta hardware. Run the 3B query planner on Ollama (CPU-only) in a separate container — Ollama's simplicity shines for lightweight auxiliary models. Use a single `AsyncOpenAI` client class for both, differing only in `base_url`. The ~30–45 tok/s decode speed is the main trade-off versus newer hardware, but for an interactive single-user agent generating short action JSONs, perceived latency will be well under a second per ReAct step.