# Optimal LLM setup for V100 32GB with llama-server

**Upgrade from Qwen3-8B Q8_0 to Qwen3-14B Q8_0 — it fits comfortably at 18.7 GB total VRAM with 16K context and parallel 2, delivering a substantial quality leap for RAG, function calling, and Russian.** If maximum quality matters more than speed, Qwen3-32B Q4_K_M fits at 24.3 GB with parallel 1. For fastest generation with 30B-class quality, Qwen3-30B-A3B MoE (Q4_K_M, ~21 GB) activates only 3B parameters per token, yielding ~2× the token/s of a comparable dense model. All three are Qwen3-family models — the only family combining native llama.cpp function calling, 119-language training (including Russian), hybrid thinking control, and Apache 2.0 licensing. Add `--cache-type-k q8_0 --cache-type-v q8_0 --jinja` to your current command line for an immediate win.

---

## VRAM budget table for every viable configuration

The formula is **model weights + KV cache (FP16) + ~500 MB overhead = total**. KV cache depends on architecture (layers × KV heads × head_dim × 2 bytes × 2 [K+V] × context × parallel slots). With `--cache-type-k q8_0 --cache-type-v q8_0`, KV cache halves. All figures below use FP16 KV cache as baseline; the "with Q8 KV" column shows the savings.

| Model + Quant | Weights | KV 16K p=1 | KV 16K p=2 | Total (16K, p=2, FP16 KV) | Total (16K, p=1, Q8 KV) | Fits 32 GB? |
|---|---|---|---|---|---|---|
| **Qwen3-8B Q8_0** | 8.7 GB | 2.25 GB | 4.50 GB | **13.7 GB** | **10.3 GB** | ✅ 18 GB margin |
| **Qwen3-14B Q8_0** | 15.7 GB | 2.50 GB | 5.00 GB | **21.2 GB** | **17.5 GB** | ✅ 11–15 GB margin |
| **Qwen3-14B Q5_K_M** | 10.5 GB | 2.50 GB | 5.00 GB | **16.0 GB** | **12.3 GB** | ✅ 16–20 GB margin |
| **Qwen3-14B Q4_K_M** | 9.0 GB | 2.50 GB | 5.00 GB | **14.5 GB** | **10.8 GB** | ✅ 18–21 GB margin |
| **Qwen3-32B Q4_K_M** | 19.8 GB | 4.00 GB | 8.00 GB | 28.3 GB (p=2 tight) | **22.3 GB** (p=1) | ⚠️ p=1 only, 10 GB margin |
| **Qwen3-32B Q5_K_M** | 23.2 GB | 4.00 GB | — | — | **25.7 GB** (p=1) | ⚠️ p=1 only, 6 GB margin |
| **Qwen3-30B-A3B MoE Q4_K_M** | ~18 GB | ~2.5 GB | ~5.0 GB | **~23.5 GB** | **~19.8 GB** | ✅ 8–12 GB margin |
| Gemma 3 27B Q4_K_M | 16.5 GB | 1.8–7.8 GB* | — | 18.8–24.8 GB | — | ⚠️ Varies |
| Phi-4 14B Q5_K_M | ~10.1 GB | 3.1 GB | 6.3 GB | ~16.9 GB | ~12.9 GB | ✅ Fits, but poor Russian |
| Mistral Nemo 12B Q8_0 | 13.0 GB | 2.50 GB | 5.00 GB | **18.5 GB** | **14.8 GB** | ✅ Fits, but outclassed |
| Mixtral 8x7B Q4_K_M | 26.4 GB | 2.00 GB | 4.00 GB | **30.9 GB** | — | ❌ Too tight |

*Gemma 3 27B KV cache varies dramatically depending on whether llama.cpp applies sliding-window optimization (51 of 62 layers use 1024-token windows).

**Key architecture details** used for KV cache: Qwen3-8B has 36 layers/8 KV heads; Qwen3-14B has 40 layers/8 KV heads; Qwen3-32B has 64 layers/8 KV heads — all with 128 head_dim and GQA. The MoE Qwen3-30B-A3B shares this KV architecture (MoE affects only FFN, not attention), so its KV cache is comparable to the 14B.

---

## Top-3 model recommendations

### 1. Qwen3-14B Q8_0 — best overall upgrade

**Why this model wins**: it doubles the parameter count from your current 8B while staying near-lossless at Q8_0 quantization. At **18.7 GB total with 16K context and parallel 2** (FP16 KV), it leaves 13 GB of headroom on your V100. Enable KV cache Q8_0 and total drops to ~17.5 GB. Benchmark-wise, Qwen3-14B scores **ArenaHard 85.5** and **Tau2-Bench 65.1** — a class above the 8B. Function calling works natively via the Hermes 2 Pro chat template in llama.cpp, and Russian language quality benefits from training on **36 trillion tokens across 119 languages**.

| Detail | Value |
|---|---|
| HuggingFace repo | `unsloth/Qwen3-14B-GGUF` (Q8_0 file) |
| File size | **15.7 GB** |
| Total VRAM (16K, p=2, Q8 KV) | **~18.2 GB** |
| Est. generation speed on V100 | **35–45 tok/s** |
| Est. prompt processing | **800–1,000 tok/s** |

```bat
set GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1

llama-server.exe ^
  -hf unsloth/Qwen3-14B-GGUF:Q8_0 ^
  -c 16384 --parallel 2 ^
  --flash-attn ^
  --cache-type-k q8_0 --cache-type-v q8_0 ^
  -ngl 99 --main-gpu 0 ^
  -t 2 ^
  --jinja ^
  --reasoning-budget 0
```

### 2. Qwen3-32B Q4_K_M — maximum quality at the cost of parallel slots

Qwen3-32B is the strongest dense model that physically fits your V100. At Q4_K_M (**19.8 GB weights**), total VRAM reaches **~22.3 GB with 16K context, parallel 1, and Q8 KV cache**. You lose the second parallel slot and drop to Q4_K_M quantization, but gain a model that scores **ArenaHard 89.5** and **BFCL 68.2** — substantially better grounding, fewer hallucinations, and more reliable function call formatting. For a single-user RAG agent, parallel 1 is perfectly acceptable.

| Detail | Value |
|---|---|
| HuggingFace repo | `unsloth/Qwen3-32B-GGUF` (Q4_K_M file) |
| File size | **19.8 GB** |
| Total VRAM (16K, p=1, Q8 KV) | **~22.3 GB** |
| Est. generation speed on V100 | **25–30 tok/s** |
| Est. prompt processing | **600–800 tok/s** |

```bat
set GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1

llama-server.exe ^
  -hf unsloth/Qwen3-32B-GGUF:Q4_K_M ^
  -c 16384 --parallel 1 ^
  --flash-attn ^
  --cache-type-k q8_0 --cache-type-v q8_0 ^
  -ngl 99 --main-gpu 0 ^
  -t 2 ^
  --jinja ^
  --reasoning-budget 0
```

### 3. Qwen3-30B-A3B MoE Q4_K_M — 30B quality at 3B inference cost

This is the dark horse. Qwen3-30B-A3B is a **Mixture-of-Experts model with 30B total parameters but only 3B active per token**. All 30B weights load into VRAM (~18 GB at Q4_K_M), but computation runs through just the active 3B, yielding **dramatically faster token generation** — potentially **80–120 tok/s** on V100, comparable to or faster than the dense 8B. Total VRAM with 16K/p=2 is ~23.5 GB (FP16 KV) or ~19.8 GB (Q8 KV). Remarkably, it scores **ArenaHard 91.0** — beating even the dense 32B. The tradeoff: MoE models can be slightly less predictable on edge cases, and prompt processing is slower than a 3B dense model because all expert weights are accessed during routing.

| Detail | Value |
|---|---|
| HuggingFace repo | `unsloth/Qwen3-30B-A3B-GGUF` (Q4_K_M file) |
| File size | **~18 GB** |
| Total VRAM (16K, p=2, Q8 KV) | **~19.8 GB** |
| Est. generation speed on V100 | **80–120 tok/s** (3B active) |

```bat
set GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1

llama-server.exe ^
  -hf unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M ^
  -c 16384 --parallel 2 ^
  --flash-attn ^
  --cache-type-k q8_0 --cache-type-v q8_0 ^
  -ngl 99 --main-gpu 0 ^
  -t 2 ^
  --jinja ^
  --reasoning-budget 0
```

---

## Why every alternative loses to Qwen3 for this use case

**Llama 4 Scout/Maverick** — disqualified outright. Llama 4 supports only 12 languages; **Russian is not among them**. Scout's 109B MoE requires ~61 GB at Q4_K_M. Maverick (402B) is entirely impractical for local inference.

**Gemma 3 27B** — strong model, 140+ language training, but **no native function calling template** in llama.cpp. It falls back to a generic handler that is unreliable for ReAct tool-use patterns. Without `--jinja` tool-calling support, you'd need custom prompt engineering for every tool call.

**Phi-4 14B** — exceptional reasoning (MMLU **84.8%**, MATH **80.4%**), but Microsoft explicitly states it is "not intended for multilingual use" and was "trained primarily on English text." Its 16K max context is also limiting for RAG. No native function calling template.

**Mistral Nemo 12B** — has native llama.cpp function calling and decent multilingual support, but released July 2024 and now outclassed by Qwen3-14B on every benchmark. Community consensus treats it as superseded.

**DeepSeek V3/V3.1** — frontier-level 671B MoE. Requires **170–245 GB** even at aggressive quantization. Physically impossible on a single V100. DeepSeek-R1-Distill variants (7B–32B) are dense reasoning models that lack proper function calling support in llama.cpp.

**Mixtral 8x7B** — all 46.7B parameters must load into VRAM (MoE weights). At Q4_K_M that's **26.4 GB** before KV cache. Fits technically at 8K context, but with only 3 GB margin — too fragile for production. And the model is now outdated compared to Qwen3.

---

## Thinking mode: when to pay the token tax

Qwen3's thinking mode wraps reasoning in `<think>...</think>` tags, adding **200–2,000+ tokens** per response. On a V100 generating at ~40 tok/s (14B Q8_0), 300 thinking tokens cost **~7.5 seconds**. Across a 4-step ReAct chain, that compounds to **30+ seconds of pure overhead** — turning a responsive 8-second workflow into a 40-second one.

**For RAG with function calling, disable thinking.** RAG is fundamentally an extraction and synthesis task: the model reads provided context and formats a response. Chain-of-thought adds latency without meaningfully improving grounding or tool-call accuracy. Qwen3's non-thinking mode already matches or exceeds Qwen2.5-Instruct quality on information extraction tasks.

Three ways to control thinking in llama-server:

- **`--reasoning-budget 0`** in the command line — globally disables thinking. Simplest approach, confirmed working in llama.cpp discussion #18424.
- **`/no_think` soft switch** in user messages — per-turn control. The model outputs an empty `<think></think>` block and proceeds directly to the answer. Works because the model itself handles the tag, not the framework.
- **Custom Jinja2 template** via `--chat-template-file` with `enable_thinking=False` hardcoded — the most reliable workaround if other methods behave inconsistently on Windows.

**Hybrid strategy** (if you want occasional thinking): launch the server with thinking enabled (default), prepend `/no_think` to every tool-calling step in your agent code, and optionally send `/think` only on the final synthesis step for complex multi-hop queries. Note that `--chat-template-kwargs '{"enable_thinking":false}'` has multiple open bugs (issues #13160, #13189, #20182) and **does not reliably work** as of build 8354.

**Alternative reasoning models** (QwQ-32B, DeepSeek-R1-Distill) are always-on thinkers with no hybrid mode — every response forces full chain-of-thought. Avoid them for multi-step agents. The Qwen3-Instruct-2507 update (July 2025) is specifically optimized for non-thinking mode with improved tool usage and instruction following — prefer these "-2507" variants on HuggingFace if available.

---

## V100 optimization: four changes that matter

**Flash attention works correctly on V100.** Unlike Dao-AILab's FlashAttention (SM 80+ only), llama.cpp implements its own CUDA FA kernel using MMA tensor core primitives that explicitly support Volta via a `DATA_LAYOUT_I_MAJOR_MIRRORED` code path. GitHub issue #13008 confirms: "significantly improved speed, and memory usage is also reduced considerably" on V100. Your current `--flash-attn on` is correct — keep it.

**KV cache quantization is the biggest free VRAM win.** Adding `--cache-type-k q8_0 --cache-type-v q8_0` halves KV cache size with negligible quality loss (perplexity increase of **+0.002 to +0.05**). For Qwen3-14B at 16K/p=2, this saves **~2.5 GB**. Avoid Q4_0 on Qwen3 models — aggressive GQA (only 8 KV heads) makes the V cache particularly sensitive to 4-bit quantization. One caveat: KV quantization requires flash attention enabled, and it is reportedly **incompatible with context shifting** — add `--no-context-shift` if you observe anomalies in long conversations.

**Speculative decoding with Qwen3-0.6B draft** is viable but optional. The syntax is `-md Qwen3-0.6B-Q8_0.gguf --draft-max 16 --draft-min 4 --draft-p-min 0.75`. Community testing with Qwen2.5 14B + 0.5B draft shows **~1.8× speedup** on token generation. The draft model adds only ~0.6 GB VRAM. However, speculative decoding requires `--parallel 1` and may conflict with flash attention in some configurations — test carefully. For the Qwen3-14B Q8_0 configuration, baseline generation is already **35–45 tok/s**, so the speedup may not justify the complexity.

**Set `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1`** before launching the server. This forces FP16 compute type in cuBLAS — specifically called out in llama.cpp documentation for V100 (which defaults to FP32 otherwise). This should meaningfully improve **prompt processing speed**, which is compute-bound on V100. Token generation (memory bandwidth-bound) benefits less but V100's **900 GB/s HBM2** already makes it competitive with RTX 3090 for generation.

---

## Performance expectations on V100 SXM2 32GB

Localscore.ai benchmarks show V100 achieving **88 tok/s generation** and **2,204 tok/s prompt processing** on Llama 3.1 8B Q4_K_M. Scaling to Qwen3 models:

| Configuration | Est. prompt (tok/s) | Est. generation (tok/s) | Latency per RAG step* |
|---|---|---|---|
| Qwen3-8B Q8_0 (current) | ~1,500 | ~70–85 | ~1.5–2.5s |
| Qwen3-14B Q8_0 | ~800–1,000 | ~35–45 | ~2.5–4.0s |
| Qwen3-14B Q4_K_M | ~1,200–1,500 | ~50–60 | ~2.0–3.0s |
| Qwen3-32B Q4_K_M | ~600–800 | ~25–30 | ~4.0–6.0s |
| Qwen3-30B-A3B MoE Q4_K_M | ~400–600 (all experts touched) | ~80–120 (3B active) | ~1.5–3.0s |

*Estimated for a typical RAG step: ~2K token prompt + ~150 token generation, no thinking mode.

The **2–5 second per step** target is comfortably met by Qwen3-14B Q8_0 and the MoE variant. Qwen3-32B Q4_K_M may occasionally exceed 5 seconds on longer prompts but stays within tolerance for single-user use.

---

## Migration from Qwen3-8B: what changes in your code

**Almost nothing changes.** All three recommended models are Qwen3 family, share the same chat template, tokenizer, and function calling format. Your existing system prompts, tool definitions, and ReAct orchestration logic work unchanged.

- **Chat template**: Identical across Qwen3-8B/14B/32B/30B-A3B. The `--jinja` flag handles everything.
- **Function calling format**: Same Hermes 2 Pro format. Tool calls arrive as `<tool_call>` blocks; tool results go in `<tool_response>` blocks. No code changes needed.
- **API compatibility**: llama-server's `/v1/chat/completions` endpoint is model-agnostic. SSE streaming works identically.
- **Token limits**: If you hardcoded max_tokens values, review them — larger models may produce slightly longer or shorter outputs, but no structural change is needed.
- **Sampling parameters**: For non-thinking mode, Qwen recommends **temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5**. Your current sampling settings may need adjustment — greedy decoding (temp=0) works fine without thinking mode but causes repetitions if thinking is ever enabled.

The only required command-line changes: swap the model file path, add `--jinja --cache-type-k q8_0 --cache-type-v q8_0 --reasoning-budget 0`, and reduce `--parallel` to 1 if using the 32B model. Set the environment variable `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1` before launching.

---

## Conclusion

The V100 SXM2 32GB is remarkably well-suited for this workload. Its **900 GB/s HBM2 bandwidth** keeps token generation competitive with modern consumer GPUs, while 32 GB of VRAM opens the door to models far beyond the current 8B. The strongest upgrade path is **Qwen3-14B Q8_0** — near-lossless quantization of a model that meaningfully outperforms the 8B on grounding, function calling, and Russian language tasks, while consuming only 18.7 GB and generating at 35–45 tok/s. For users willing to accept Q4_K_M quantization and parallel 1, the 32B dense model and 30B-A3B MoE both fit and deliver a further quality jump. Disable thinking mode for all tool-calling steps — the 23–57 second overhead across a ReAct chain is not justified for context extraction tasks. Finally, KV cache quantization to Q8_0 is essentially free quality-wise and should be enabled unconditionally — it saves 1–4 GB of VRAM that can be redirected to larger models or longer context.