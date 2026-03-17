# Best LLM for a V100 32GB Russian RAG agent

**Qwen3-8B in FP16 is the optimal model for this setup.** It consumes ~16.4 GB of VRAM, leaves ~14 GB for KV-cache and overhead, matches Qwen2.5-14B in quality benchmarks, and ships with native Hermes-style tool calling that works out of the box with vLLM's `hermes` parser. This single model replaces the current two-model split (7B GPU + 3B CPU) while delivering better quality across every task and eliminating the CPU inference bottleneck entirely. The V100's Volta architecture imposes hard constraints — no AWQ/GPTQ-Marlin kernels, no FlashAttention2, no FP8 — that make FP16 on a smaller, newer model strictly superior to quantized 14B alternatives on this hardware.

---

## Which models actually fit on V100 32GB

The V100 SXM2 (SM 7.0, Volta) presents a unique deployment challenge in 2025-2026: while its **32 GB HBM2** is generous by capacity, its compute architecture lacks the quantization acceleration that makes larger models viable on newer GPUs. AWQ requires SM ≥ 7.5 (Turing), GPTQ-Marlin requires SM ≥ 8.0 (Ampere), and FP8 requires SM ≥ 8.9. vLLM confirms this — issue #1488 explicitly states "Minimum capability: 75. Current capability: 70" when attempting AWQ on V100. The only quantization paths that work are BitsAndBytes (general-purpose, slow) and basic INT8 weight-only via compressed-tensors, both with a **10–30% throughput penalty** from dequantization overhead without optimized kernels.

Additionally, vLLM's V1 engine requires CC ≥ 8.0, so V100 falls back to the legacy **V0 engine** — functional but slower. BFloat16 is not supported; you must use `--dtype half`. FlashAttention2 is unavailable; use `--enforce-eager` or xFormers.

| Model | Params | FP16 VRAM | Fits with KV headroom? | Tool calling parser | Notes |
|---|---|---|---|---|---|
| **Qwen3-8B** | 8.2B | ~16.4 GB | ✅ ~14 GB free | `hermes` | Best quality-per-GB; ≈ Qwen2.5-14B |
| Qwen2.5-7B-Instruct | 7.6B | ~15.2 GB | ✅ ~15 GB free | `hermes` | Proven, stable baseline |
| T-lite-it-1.0 | 8B | ~16 GB | ✅ ~14 GB free | ❌ None | Excellent Russian but no tool calling |
| Gemma 2 9B-IT | 9.2B | ~18.4 GB | ✅ ~12 GB free | ❌ No vLLM parser | Decent Russian, weak tool calling |
| Mistral Nemo 12B | 12.2B | ~24.4 GB | ⚠️ ~6 GB tight | `mistral` | Poor raw Russian (40.0 Ru Arena Hard) |
| Vikhr-Nemo-12B | 12.2B | ~24.4 GB | ⚠️ ~6 GB tight | ❌ Not optimized | Best Russian quality (65.5) but tight fit |
| Gemma 3 12B | ~12B | ~24 GB | ⚠️ ~6 GB tight | ❌ No vLLM parser | Good multilingual, VRAM often higher than expected |
| Qwen3-14B | 14.8B | ~29.6 GB | ❌ No KV room in FP16 | `hermes` | Needs INT8; throughput penalty on V100 |
| Qwen2.5-14B-Instruct | 14.7B | ~29.4 GB | ❌ No KV room in FP16 | `hermes` | Same issue as Qwen3-14B |
| GigaChat3 Lightning 10B-A1.8B | 10B MoE | ~20 GB FP16 | ⚠️ Custom MLA arch | `gigachat3` (SGLang only) | Risky on V100 — MLA/MTP compatibility uncertain |

**The FP16-on-V100 sweet spot is 7B–9B models.** The 12B class fits but leaves only ~6 GB for KV-cache, limiting batch size to 1 and risking OOM with longer contexts. The 14B class does not fit in FP16 at all — the weights alone consume the entire budget. INT8 14B models (~15 GB weights) fit by capacity but suffer throughput degradation without Marlin kernels, making them slower than FP16 8B models while being more complex to deploy.

### INT8 quantization verdict for V100

**FP16 on a smaller model beats INT8 on a larger model on V100.** A Qwen3-14B in INT8 via BitsAndBytes would offer Qwen2.5-32B-level quality, but the 10–30% throughput hit from unoptimized dequantization, the added deployment complexity, and the risk of subtle quality artifacts make it a poor trade versus Qwen3-8B in native FP16. On Ampere or Hopper GPUs with Marlin kernel support, this calculus reverses entirely — INT4 AWQ on a 14B model becomes the sweet spot. But **V100 + vLLM = FP16 only** in practice.

---

## Russian language quality across candidates

The **Ru Arena Hard** benchmark (style-controlled, GPT-3.5-turbo baseline = 50.0) provides the clearest head-to-head comparison of Russian generation quality for open models. The rankings reveal a striking pattern: **Russian-specific fine-tuning is worth approximately a 2× parameter multiplier** in effective quality.

| Model | Size | Ru Arena Hard (style) | Russian quality tier |
|---|---|---|---|
| Vikhr-Nemo-12B-Instruct | 12B | **65.5** | Excellent — best open 12B |
| T-lite-it-1.0 | 7B | **64.38** | Excellent — matches models 2× its size |
| Qwen2.5-14B-Instruct | 14B | 59.0 | Good |
| Qwen2.5-7B-Instruct | 7B | 54.29 | Adequate |
| Gemma 2 9B-IT | 9B | 54.3 | Adequate |
| Phi-3-Medium-14B | 14B | 45.0 | Below average |
| Mistral Nemo 12B (raw) | 12B | 40.0 | Poor |
| Saiga-LLaMA3-8B | 8B | 39.2 | Poor |

**Qwen3-8B** is not yet on Ru Arena Hard, but its official benchmarks place it at Qwen2.5-14B parity across most tasks. For Russian specifically, it supports **119 languages** (up from 29 in Qwen2.5), suggesting broader multilingual capability. A reasonable estimate is **~57–61** on Ru Arena Hard — solid but not exceptional.

The standout discovery is **T-lite-it-2.1** (by T-Technologies/T-Bank), built on Qwen3 architecture. Unlike its predecessor T-lite-it-1.0 (which lacks tool calling), version 2.1 adds tool-calling support and reportedly outperforms Qwen3-8B on tool-calling scenarios while maintaining superior Russian quality. This model warrants serious consideration if available and stable in vLLM at deployment time.

### Language mixing risk

Models trained primarily on English/Chinese data tend to inject English terms or switch languages mid-response. For a production Russian RAG agent, this matters. **Qwen2.5/Qwen3** models occasionally produce Chinese or English artifacts, especially with ambiguous prompts. Qwen3's broader 119-language training may increase confusion between Cyrillic scripts (reported Ukrainian/Bulgarian switches with ALL CAPS input). An explicit "Always respond in Russian" system prompt instruction mitigates most mixing, but Russian-adapted models like T-lite and Vikhr-Nemo produce cleaner Russian natively without prompt engineering workarounds.

---

## Tool calling reliability: BFCL scores and vLLM support

**Qwen models dominate the ≤15B tool-calling space**, and the native Hermes parser in vLLM makes them the path of least resistance.

From the Berkeley Function Calling Leaderboard (BFCL V3/V4) and official technical reports, the estimated ranking for models ≤15B is:

1. **Qwen3-14B** — matches Qwen2.5-32B; native Hermes tool calling
2. **Qwen3-8B** — matches Qwen2.5-14B; native Hermes tool calling
3. **Qwen2.5-14B-Instruct** — well-tested, proven ecosystem
4. **LLaMA 3.1 8B** — 76.1% BFCL overall but weak on parallel calls
5. **Qwen2.5-7B-Instruct** — solid baseline
6. **Mistral Nemo 12B** — struggles with parallel calls per vLLM docs
7. **Gemma 3 12B-IT** — no native vLLM tool parser

The BFCL Magnet project demonstrated that fine-tuning a Qwen2.5-Coder-14B backbone specifically for multi-turn tool calling improved scores by **32.5 points** on multi-turn cases — showing that fine-tuning matters enormously for agentic reliability beyond what base instruction tuning provides.

### Native tool calling vs text-based ReAct

**Switch from your current regex-based ReAct to native Hermes tool calling.** This is the single highest-impact reliability improvement available.

Your current setup parses `Thought/Action/Action Input/Observation/Final Answer` markers with regex. This approach has three fundamental weaknesses: fragile parsing when the model adds extra text around markers, wasted tokens on verbose reasoning traces, and a critical incompatibility with Qwen3's thinking mode (the model may emit "Action:" inside `<think>` blocks, triggering premature parsing). Native tool calling via vLLM's `hermes` parser eliminates all three issues.

The vLLM `hermes` parser works with both Qwen2.5 and Qwen3. It intercepts `<tool_call>` XML tags in the model's output stream and routes them through the OpenAI-compatible `tool_calls` response field. Your FastAPI application receives structured `tool_calls` objects identical to the OpenAI API format — no regex needed.

### Constrained decoding with xgrammar

vLLM's default structured output backend, **xgrammar**, uses a Pushdown Automaton compiled in C to enforce JSON schema validity during decoding. Benchmarks show it pushes valid JSON generation from **90–94% to 96–98%+** with roughly 5–15% throughput overhead. For named function calling with `tool_choice` set to a specific function, vLLM automatically applies schema-constrained decoding.

Important caveat: constrained decoding guarantees **syntactic validity** (well-formed JSON matching the schema) but not **semantic correctness** (right tool name, sensible parameter values). You still need validation logic for the latter.

### Temperature for tool calling

**Use temperature 0.0 for tool calling steps.** Databricks research found accuracy varies by up to 10% between temperature 0.0 and 0.7 on BFCL categories. The BFCL evaluation itself uses temperature 0.0.

One exception: **Qwen3 in thinking mode should never use greedy decoding** — it causes degeneration and repetition loops. For thinking mode, use temperature 0.6, top_p 0.95. But since non-thinking mode is recommended for agent tasks (see below), temperature 0.0 is safe and optimal.

---

## Prompt engineering for Qwen3-8B agent

### Use English system prompts with Russian output instructions

Research confirms that multilingual LLMs process through English-like internal representations regardless of input/output language. Studies on Arabic and other non-English languages consistently show that **English prompts outperform native-language prompts** for structural tasks like instruction following and JSON generation. The optimal strategy is:

- **System prompt**: English (structural instructions, tool definitions, format rules)
- **Output instruction**: "Always respond to the user in Russian" (in English)
- **User content and RAG documents**: Russian (preserve original language)
- **Tool calls**: JSON (language-agnostic)

This approach maximizes instruction following for the structural ReAct/tool-calling components while the model naturally handles Russian input/output. Bonus: English system prompts consume **~30–40% fewer tokens** than equivalent Russian text due to tokenizer efficiency.

### Non-thinking mode is correct for this use case

Qwen3-8B supports both thinking (`<think>...</think>` blocks) and non-thinking modes. **Use non-thinking mode** (`enable_thinking=False`) for the RAG agent:

- Saves **250–1250 tokens** over a 5-step chain (thinking blocks consume 100–300 tokens each)
- Avoids the stopword conflict where Qwen3 emits ReAct-like markers inside think blocks
- Avoids the greedy decoding prohibition (temperature 0.0 is safe in non-thinking mode)
- Reduces latency per step
- For RAG tasks where the model is selecting tools and composing answers from retrieved documents, full chain-of-thought reasoning adds minimal quality benefit

### Few-shot: one concise example for 8B models

With an **8K context window**, every token matters. The budget breaks down approximately as:

- System prompt + tools: ~700–1000 tokens
- One few-shot example: ~200–300 tokens
- User query: ~50–100 tokens
- RAG documents (2–3 chunks): ~2500–3500 tokens
- Agent reasoning chain (3–5 steps): ~1500–2000 tokens
- Output buffer: ~500–1000 tokens

**Include one concise few-shot example** (~200 tokens) showing a complete tool call → observation → final answer cycle. Research shows this dramatically improves format adherence for 8B models with diminishing returns beyond 2 examples. For 14B+ models, zero-shot is typically sufficient, but for 8B, the single example is worth the token cost.

### Context management with 8K tokens

Retrieve **3–5 chunks of 400–512 tokens each**, then truncate to fit the remaining budget. Position the most relevant chunks at the **beginning and end** of the context (the "lost in the middle" effect means models attend less to central content). After each tool execution, truncate observation results to ~500–1000 tokens maximum. In multi-turn ReAct chains, summarize completed steps and discard intermediate reasoning to free tokens for new steps. Add a hard cutoff rule: if context exceeds 7500 tokens, force a Final Answer with available information.

### Recommended vLLM launch command

```bash
vllm serve Qwen/Qwen3-8B \
  --dtype half \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Key flags explained: `--dtype half` because V100 lacks BFloat16 support; `--enforce-eager` because FlashAttention2 requires SM ≥ 8.0; `--gpu-memory-utilization 0.92` leaves headroom for CUDA overhead while maximizing KV-cache. The `hermes` parser handles Qwen3's native `<tool_call>` format and streams structured `tool_calls` through the OpenAI-compatible API.

---

## Parsing reliability and error recovery playbook

### Four-layer JSON parsing fallback chain

Even with native tool calling and constrained decoding, edge cases occur. Implement this ordered fallback:

1. **Constrained decoding (prevention)**: Use vLLM's `guided_json` or named `tool_choice` to guarantee syntactically valid JSON at generation time. This is the first and most effective defense, achieving near-100% syntactic validity with xgrammar.

2. **Direct JSON parse**: If constrained decoding is not active (e.g., for the model's free-form reasoning steps), attempt `json.loads()` on extracted JSON content.

3. **json-repair library**: The `json_repair` package (v0.58+, actively maintained) fixes missing quotes, trailing commas, unescaped characters, Python-style booleans, and prose wrapped around JSON. Use `repair_json(broken_json, ensure_ascii=False)` for Cyrillic content. A Rust-based `fast_json_repair` drop-in is available for production throughput.

4. **Regex extraction + retry**: Extract any `{...}` block via regex, attempt repair, and if all else fails, inject an error message into the conversation: "Your previous output was malformed. Please output ONLY valid JSON: {\"param\": \"value\"}" and regenerate.

### Hallucination loop detection

Track action history as `(tool_name, serialized_params)` tuples. Detect three failure modes:

- **Exact repetition**: If the same `(tool, params)` pair appears twice, inject the previous observation: "You already called this tool with these parameters. The result was: [previous_result]. Use this information to proceed."
- **Near-duplicate oscillation**: If consecutive actions have >85% string similarity (via `SequenceMatcher`), or an A→B→A→B pattern emerges over 4 steps, force a Final Answer.
- **Maximum iterations**: Hard limit of **5 tool calls per query**. At the limit, force the model to answer with available information. Encode this in the system prompt: "You have a maximum of 5 tool calls per question."

### Mandatory step enforcement

For RAG, the retrieval step is non-negotiable — skipping it means the model hallucinates from parametric memory. Two complementary strategies:

**Programmatic forcing**: Always execute the first search tool call programmatically before the model makes any decisions. Pass the user query directly to `search_documents`, inject results as the first Observation, then let the model reason from there. This removes one decision point and guarantees RAG grounding.

**Validation gate**: Before accepting a Final Answer, check that all mandatory tools (e.g., `search_documents`, `compose_context`) were called. If not, inject: "You must call [missing_tool] before providing Final Answer." This catches cases where the model tries to shortcut.

---

## One large model vs two smaller models: use one Qwen3-8B

**A single Qwen3-8B on the V100 is strictly superior to the current 7B + 3B split.** The reasoning is straightforward:

- **Quality**: Qwen3-8B ≈ Qwen2.5-14B >> Qwen2.5-3B. The 3B planner model is the weakest link in the current pipeline. Every search plan it generates below the quality threshold degrades the entire RAG chain.
- **Latency**: The 3B model on CPU via Ollama runs at ~5–15 tok/s. Even a simple search plan of 200 tokens takes 13–40 seconds. Qwen3-8B on V100 FP16 eliminates this bottleneck entirely — expect **40–60 tok/s** for generation.
- **Complexity**: A single model means one serving process, one set of prompts, no routing logic, no CPU/GPU coordination. Every removed component is a removed failure mode.
- **VRAM**: Qwen3-8B at ~16.4 GB leaves ~14 GB for KV-cache — ample for an 8K context with single-request serving and room for modest batching.

The only scenario where two models wins is if the planner requires fundamentally different capabilities than the main model (e.g., structured output in a completely different format). In this case, both search planning and final answering require Russian language understanding, tool awareness, and JSON output — Qwen3-8B handles all of these in a single model.

---

## Conclusion

The optimal deployment for a V100 SXM2 32GB running a Russian-language RAG agent is **Qwen3-8B in FP16 with native Hermes tool calling**, served via vLLM with `--dtype half --enforce-eager --tool-call-parser hermes`. This configuration delivers Qwen2.5-14B-class quality in a 16.4 GB VRAM footprint, leaves ample room for KV-cache, and avoids every V100-specific pitfall (no AWQ, no Flash Attention, no quantization overhead).

Three forward-looking options deserve monitoring. **T-lite-it-2.1** (T-Technologies, Qwen3-based) adds purpose-built Russian optimization and tool calling to a Qwen3 backbone — if stability and vLLM compatibility are confirmed, it could surpass vanilla Qwen3-8B for this exact use case. **GigaChat3 Lightning** (10B-A1.8B MoE, MIT license) offers the cleanest Russian output with tool calling support, but its custom MLA architecture creates V100 compatibility risk. **Qwen3.5-9B** (if released in the ~9B range) could offer a generational improvement while remaining in the V100 FP16 sweet spot.

The decisive technical insight is that **V100's Volta architecture makes FP16 the only production-viable precision** in vLLM. This constraint collapses the model selection problem: forget quantized 14B models and choose the best 7–9B model in native FP16. Among those, Qwen3-8B wins on the composite of tool-calling reliability, Russian language capability, and ecosystem maturity.