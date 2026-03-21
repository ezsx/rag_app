# Upgrading a Russian-language Telegram RAG pipeline to 2025-2026 standards

The single highest-impact change for this system is switching from text-based ReAct to native function calling — Qwen3's own documentation explicitly warns against regex-based ReAct parsing because thinking-mode `<think>` blocks can contain stopwords that break it. Combining this with three other near-free wins — enabling the already-deployed reranker, upgrading to Qwen3-Embedding-0.6B (a drop-in replacement scoring **+7.5 points on retrieval** vs multilingual-e5-large), and implementing a two-tier chunking strategy — would transform retrieval quality without new hardware. The system's current 885-document corpus and RTX 5060 Ti 16GB GPU are well-suited for these changes, with all recommended models fitting comfortably in VRAM alongside the main Qwen3 LLM.

---

## Block 1: Short posts should stay whole, digests need recursive splitting

The research literature from 2025-2026 converges on a clear principle: **chunking helps multi-topic documents but actively hurts short, single-topic ones**. Weaviate's documentation states directly that social media posts, FAQs, and product descriptions should not be chunked. Pinecone's research confirms that query–chunk size mismatch degrades cosine similarity scores. For the Telegram corpus where 60-70% of posts are 100-500 character news items, keeping these whole is essential.

For longer digest posts (2000-5000 chars), chunking is not optional — it is mandatory. The `multilingual-e5-large` model has a **hard ceiling of 512 tokens**, and Russian Cyrillic text tokenizes at roughly 1 token per 3-4 characters using XLM-RoBERTa's SentencePiece tokenizer. A 2000-char Russian post produces approximately 500-660 tokens, meaning content is silently truncated. This is actively destroying retrieval quality for every digest post in the current system.

**Recommended two-tier strategy:**

| Post length | Token estimate | Action | Rationale |
|---|---|---|---|
| ≤1000 chars | ~250-330 tokens | Keep whole | Fits in 512-token window, single topic |
| 1000-1500 chars | ~330-500 tokens | Keep whole, verify with tokenizer | Borderline; check actual token count |
| >1500 chars | >400 tokens | Recursive chunking | Likely exceeds 512 tokens; multi-topic |

The optimal chunking method is **recursive character splitting** with separator hierarchy `["\n\n", "\n", ". ", " "]`, targeting ~400 tokens per chunk. Telegram digests naturally separate topics with blank lines, making `\n\n` the ideal primary separator. A February 2026 benchmark by Vecta across 50 academic papers ranked recursive 512-token splitting first at 69% accuracy, while semantic chunking scored only 54% with fragments averaging just 43 tokens. A NAACL 2025 Findings paper from Vectara concluded that "the computational costs associated with semantic chunking are not justified by consistent performance gains."

**Overlap is unnecessary.** The most recent empirical evidence (January 2026 systematic analysis and a Chemistry RAG study, arXiv:2506.17277) found that recursive non-overlapping chunking was the strongest default. The Chemistry RAG paper recommended it explicitly: "Use recursive, non-overlapping chunking as a strong default." With hybrid search using RRF fusion, BM25 sparse vectors compensate for any boundary effects that overlap would theoretically address.

The impact of chunking on `dense_score` is empirically confirmed to be substantial. The Chemistry RAG study found that **chunking configuration has impact comparable to or greater than the embedding model itself**, with tenfold variation in IoU across strategies. A clinical decision support study (MDPI Bioengineering, November 2025) measured F1 scores of 0.64 for adaptive chunking versus 0.24 for a fixed baseline — same model, same embeddings, chunking alone responsible for the difference. For Qdrant's hybrid search specifically, focused chunks produce higher dense scores because the embedding captures a single topic, and higher sparse scores because BM25 term frequency concentration increases in shorter chunks.

For Russian text specifically, use the `razdel` library (natasha/razdel) for sentence boundary detection — it correctly handles Russian abbreviations like "т.е.", "т.д.", and "и т.п." that trip up generic NLTK or spaCy tokenizers. Strip emojis before embedding, as they waste tokens. Always validate chunk sizes with the actual tokenizer rather than relying on character-count heuristics.

**Priority:** Must-have. **Effort:** 4-8 hours. **Impact:** 20-40% retrieval precision improvement for long digest posts; eliminates silent truncation.

---

## Block 2: Enabling the reranker is the easiest high-impact win available

The `bge-reranker-v2-m3` is already deployed via TEI and simply needs to be wired into the agent pipeline. NVIDIA's BEIR benchmark measured a **+4.85 NDCG@10 improvement** from adding bge-reranker-v2-m3 after retrieval. An AIMultiple benchmark (February 2026) found rerankers improve Hit@1 by **15-20 percentage points** — the difference between finding the right answer at position 1 versus position 3-5. A production RAG benchmark on dev.to reported reranking improved answer quality by 31%.

Counterintuitively, reranking actually **reduces total pipeline latency by 60-80%**. Without reranking, 20+ chunks get passed to the LLM (4000-8000ms generation time). With reranking at 40-80ms, only the top 5 chunks reach the LLM, cutting generation to 600-1600ms. The reranker pays for itself many times over in saved LLM tokens.

On the RTX 5060 Ti 16GB, the `bge-reranker-v2-m3` (568M parameters, ~1.5GB VRAM) should rerank 10-20 documents in **20-80ms** — well within any real-time budget. The recommended approach is to retrieve k=20 candidates from hybrid search and rerank to top 5.

However, bge-reranker-v2-m3 is now clearly outpaced by 2025 models. The upgrade path is straightforward:

| Model | MMTEB-R (multilingual) | Size | VRAM | TEI compatible | License |
|---|---|---|---|---|---|
| bge-reranker-v2-m3 (current) | 58.36 | 568M | ~1.5GB | ✅ Native | Apache 2.0 |
| **Qwen3-Reranker-0.6B** | **66.36 (+8.0)** | 0.6B | ~1.5GB | ✅ via seq-cls conversion | Apache 2.0 |
| Jina Reranker v3 | ~65-66 | 560M | ~1.2GB | Needs verification | Commercial |
| Qwen3-Reranker-4B | **72.74** | 4B | ~8GB | ✅ via seq-cls conversion | Apache 2.0 |
| Contextual AI Reranker v2 1B | Competitive | 1B | ~2GB | Needs verification | Open-source |

**Qwen3-Reranker-0.6B is the recommended upgrade** — same parameter count as bge-v2-m3 but +8 points on multilingual benchmarks, with explicit Russian language support. A critical TEI compatibility note: Qwen3-Reranker uses causal LM yes/no scoring not natively supported by TEI. Use the `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` conversion on HuggingFace, which wraps it into SequenceClassification format for TEI compatibility.

**Priority:** Must-have (enable current reranker today, upgrade model later). **Effort:** 2-4 hours to wire in; 4-8 hours for model upgrade. **Impact:** +15-20 percentage points on top-1 precision; reduced total pipeline latency.

---

## Block 3: Qwen3-Embedding-0.6B is a drop-in upgrade with dramatic retrieval gains

The comparison between the current multilingual-e5-large and Qwen3-Embedding-0.6B is decisive. Both are 0.6B parameter models producing 1024-dimensional embeddings, but Qwen3-Embedding scores **64.64 versus 57.12 on MMTEB retrieval tasks** — a +7.52 point gap that represents a generation leap in embedding quality. On English retrieval specifically, the gap is even wider at +8.36 points.

| Model | MMTEB overall | MMTEB retrieval | Max context | MRL support | VRAM |
|---|---|---|---|---|---|
| multilingual-e5-large (current) | ~61-62 | 57.12 | **512 tokens** | No | ~1.5GB |
| **Qwen3-Embedding-0.6B** | **64.33** | **64.64** | **32K tokens** | Yes (32-1024) | ~1.8GB |
| BGE-M3 | 59.56 | 54.60 | 8K | No | ~1.5GB |
| jina-embeddings-v3 | ~64.44 | — | 8K | Yes | ~1.5GB |
| Qwen3-Embedding-4B | **69.45** | **69.60** | 32K | Yes | ~9-10GB |

Beyond benchmark scores, the **32K context window** versus e5-large's 512 tokens is transformative. With Qwen3-Embedding, even the longest 5000-char digest posts can be embedded without any truncation risk, making the chunking strategy more about semantic precision than token-limit compliance. The model also supports Matryoshka Representation Learning (MRL), enabling future dimension reduction if the corpus grows — though at 885 documents, the full 1024 dimensions use only ~3.5MB of storage, making reduction pointless today.

The instruction format changes from `"query: "` / `"passage: "` to a richer format: queries use `"Instruct: <task description>\nQuery: <query>"` while documents need no prefix at all. Instructions should be written in English even for Russian content. A domain-specific instruction like `"Instruct: Given a search query about ML/AI/LLM topics, retrieve relevant Telegram channel posts\nQuery: ..."` can provide **1-5% additional improvement** over generic instructions.

**TEI compatibility warning:** Qwen3-Embedding is officially supported since TEI v1.7.2, but GitHub issues #642 and #668 documented embedding inconsistencies between TEI and SentenceTransformers outputs. The issues appear resolved in v1.8.0+, but validation against SentenceTransformers reference output is essential before production deployment. TEI inference speed is approximately **20ms per query** on RTX 4060-class hardware, comparable to the current e5-large setup.

BGE-M3 is not recommended despite its unique dense+sparse+multi-vector capability — its MMTEB score of 59.56 is actually lower than e5-large-instruct. Jina-embeddings-v3 is competitive in quality but carries CC-BY-NC-4.0 license restrictions. GTE-Qwen2 is superseded by Qwen3-Embedding in every metric.

Migration requires re-embedding all 885 documents, which is trivial — perhaps 2-3 minutes of GPU time. The Qdrant collection dimensions remain 1024, so no vector database schema changes are needed.

**Priority:** Must-have. **Effort:** 8-16 hours (TEI upgrade, client changes, re-embedding, validation). **Impact:** +7.5 points on multilingual retrieval; eliminates 512-token truncation.

---

## Block 4: Qwen3 explicitly forbids ReAct regex parsing — switch to function calling immediately

The Qwen3 documentation contains an unambiguous warning: **"For reasoning models like Qwen3, it is not recommended to use tool call template based on stopwords, such as ReAct, because the model may output stopwords in the thought section, potentially leading to unexpected behavior in tool calls."** The `<think>...</think>` blocks in Qwen3's thinking mode routinely contain text like "Action:" or "Observation:" that will trigger regex-based parsers to misfire. This is not a theoretical concern — it is a documented failure mode that the Qwen team specifically warns against.

Native function calling via `/v1/chat/completions` with a `tools` schema is the official replacement. Qwen3 uses a Hermes-style tool-calling template and supports parallel function calls natively. The migration path is to define each tool as a JSON schema with name, description, and parameters, then use the structured `tool_calls` field in the response instead of parsing freetext. Consider the Qwen-Agent framework, which encapsulates these templates and parsers internally.

The current 7+1 tool pipeline (`router_select → query_plan → search → [rerank] → compose_context → verify → final_answer`) has both redundancies and gaps relative to 2025 state-of-the-art. If there is only one search backend, `router_select` adds a wasted LLM reasoning step and should be merged into `query_plan` or eliminated. The skipped rerank step is a critical gap — it should be re-enabled immediately. Compared to Corrective RAG (CRAG), the 2025 standard architecture, the pipeline is missing query rewriting/expansion (essential for Russian morphological variants), document relevance grading (filtering irrelevant retrievals before LLM consumption), and a corrective retrieval fallback (retrying with rewritten queries when initial results are poor).

The recommended architecture follows the CRAG pattern as a single agent with self-correction:

```
query_plan → [query_rewrite] → search → rerank → grade_relevance →
  ├─ (good results) → compose_context → generate_with_citations → verify → final_answer
  └─ (poor results) → rewrite query → search again → compose_context → generate
```

**A single agent with CRAG is preferred over a two-stage retrieval/generation split.** On RTX 5060 Ti, each Qwen3 LLM call costs 1-3 seconds; a two-agent architecture doubles this latency for a Telegram bot where response time matters. The single-agent approach with tool-based self-correction achieves the same quality benefits without the latency penalty.

For grounding and hallucination prevention, the most effective practical technique in 2025-2026 is **citation-forced generation**. Label each retrieved chunk with a unique ID (`[1]`, `[2]`, etc.) and require the LLM to cite sources inline. A production RAG benchmark found this **reduced hallucinations by 60%** compared to naive context concatenation. Combining this with a Russian-language system prompt that explicitly instructs "Отвечай ТОЛЬКО на основе предоставленного контекста" and requiring the `final_answer` tool to include a `sources` array parameter creates multiple enforcement layers. Grammar-constrained decoding (available in vLLM and llama.cpp) can guarantee format compliance by making it physically impossible for the model to output uncited claims.

Qwen3's thinking mode should be used strategically: enable it for query planning and verification steps (where deep reasoning improves quality), disable it for final answer generation (where speed matters). Critical note: **never use greedy decoding (temperature=0) with Qwen3** — it causes performance degradation and endless repetitions. Use temperature 0.6 with TopP 0.95 for thinking mode, and temperature 0.7 with TopP 0.8 for non-thinking mode.

**Priority:** Must-have (function calling switch). **Effort:** 16-24 hours for function calling migration; 8-16 hours for CRAG pattern; 4-8 hours for citation-forced generation. **Impact:** Eliminates parsing failures; +31% answer quality from reranking integration; -60% hallucination rate.

---

## Prioritized implementation roadmap

The four blocks are not independent — they compound. Enabling the reranker feeds better context to the LLM, reducing hallucinations. Better embeddings improve the candidates available for reranking. Proper chunking ensures embeddings actually capture the right content. And a reliable agent architecture ensures none of these improvements are lost to parsing failures.

**Week 1 (must-haves, ~24-40 hours):**
- Switch from ReAct regex to native function calling (16-24h, eliminates parsing failures)
- Enable bge-reranker-v2-m3 in the pipeline at k=20→5 (2-4h, +15-20pp precision)
- Implement two-tier chunking with recursive splitting for long posts (4-8h, +20-40% precision on digests)
- Add citation-forced generation to system prompt (2-4h, -60% hallucinations)

**Week 2-3 (high-impact upgrades, ~24-32 hours):**
- Upgrade to Qwen3-Embedding-0.6B via TEI v1.8.0+ (8-16h, +7.5 retrieval points)
- Upgrade to Qwen3-Reranker-0.6B-seq-cls (4-8h, +8 multilingual reranking points)
- Add query rewriting tool for Russian morphological expansion (8h)
- Implement document relevance grading / CRAG fallback pattern (8h)

**Month 2 (nice-to-haves):**
- Evaluate Qwen3-30B-A3B MoE as LLM upgrade (only 3B active parameters, similar inference cost to 8B but dramatically better quality)
- Add grammar-constrained decoding for structured outputs
- Implement Qwen3 thinking mode for verification, non-thinking for generation
- Explore DSPy for automatic prompt optimization

The total VRAM budget on RTX 5060 Ti 16GB accommodates all recommended models simultaneously: Qwen3-8B at Q4_K_M (~5-6GB) + Qwen3-Embedding-0.6B (~1.8GB) + Qwen3-Reranker-0.6B (~1.5GB) = ~9-10GB, leaving comfortable headroom for KV cache and batching.