# Deep Research: Advanced Retrieval Strategies for RAG Pipeline

## Project Context

I'm building a **portfolio-grade RAG system** (for Applied LLM Engineer job interviews) — a Telegram-channel aggregator with a ReAct agent pipeline. The system ingests posts from 36 AI/ML Telegram channels (~13,000 documents) and answers user questions with citations.

**The goal**: demonstrate that a custom pipeline with adaptive retrieval **outperforms** LlamaIndex/LangChain on domain-specific corpora. This requires real metrics on a golden dataset, not just "it works."

### Full Architecture

```
User Query → ReAct Agent (Qwen3-30B, native function calling)
  → query_plan tool: LLM generates sub-queries + metadata filters
  → search tool: each sub-query → Qdrant hybrid search (dense + BM25 sparse → RRF fusion)
  → rerank tool: BGE-M3 cross-encoder re-scores top candidates
  → compose_context tool: builds context window with citations
  → final_answer tool: LLM generates grounded answer with source references
  → verify (system): checks answer against retrieved docs
```

### Tech Stack Details

- **Vector DB**: Qdrant v1.13, named vectors:
  - `dense_vector`: 1024-dim cosine (Qwen3-Embedding-0.6B, instruct-aware)
  - `sparse_vector`: BM25 via fastembed (Qdrant/bm25 model)
- **Embedding**: Qwen3-Embedding-0.6B — latest model (May 2025), top MTEB scores, but only 0.6B params. Running natively on RTX 5060 Ti via PyTorch cu128 (Docker GPU unavailable due to V100 TCC NVML conflict).
- **Reranker**: BAAI/bge-m3 (XLMRoberta-based seq-cls) — same RTX 5060 Ti, same process as embedding.
- **LLM**: Qwen3-30B-A3B-Q4_K_M (MoE, 3B active) via llama-server on V100 SXM2 32GB. Context: 16K tokens, `--reasoning-budget 0`.
- **Agent**: ReAct loop with 5 tools (query_plan, search, rerank, compose_context, final_answer). Coverage threshold 0.65, max 2 refinements.
- **Corpus**: Russian-language Telegram posts about AI/ML/LLM, period July 2025 – March 2026. Posts are 300-1500 chars, mix Russian/English, contain links, emojis, code snippets. Smart chunking: posts <1500 chars = single chunk, >1500 = recursive split.

### Current Retrieval Pipeline (HybridRetriever)

```python
# Step 1: Embed query with instruct prefix
dense_vector = embed_query("Instruct: Given a user question...\nQuery: {query}")

# Step 2: BM25 sparse encoding
sparse_vector = fastembed_bm25.query_embed(query)

# Step 3: Qdrant RRF fusion (both prefetch branches have equal weight)
result = qdrant.query_points(
    prefetch=[
        Prefetch(query=dense_vector, using="dense_vector", limit=20),
        Prefetch(query=sparse_vector, using="sparse_vector", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),  # final step = RRF, no re-scoring
    limit=10
)

# Step 4: Reranker (BGE-M3) re-scores top candidates
reranked = reranker.rerank(query, top_candidates, top_n=5)

# Step 5: LLM generates answer from reranked context
```

## The Problem (Detailed)

### Eval Results: Recall@5 = 0.33 on Golden Dataset

10 questions, 5 types: factual (3), temporal (2), channel-specific (2), comparative (1), multi-hop (1), negative (1).

| # | Type | Query | Recall | Problem |
|---|------|-------|--------|---------|
| Q1 | factual | "Кого FT назвала человеком года 2025?" | 0 | **LLM skipped search** (2s latency, 0 coverage) |
| Q2 | factual | "Параметры open-source GPT от OpenAI?" | 0 | LLM sub-queries lost "OpenAI" entity |
| Q3 | factual | "За сколько Meta купила Manus AI?" | **1.0** | Fixed by removing dense re-score |
| Q4 | temporal | "Продукты Google/NVIDIA декабрь 2025?" | 0 | Found Dec posts but not the specific ones |
| Q5 | temporal | "AI-каналы январь 2026?" | 0 | Found Jan posts but from wrong channels |
| Q6 | channel | "llm_under_hood про reasoning GPT-5?" | **1.0** | Fixed by RRF (BM25 found it) |
| Q7 | channel | "Gemini 3 Flash бенчмарк boris_again?" | **1.0** | Fixed by RRF (BM25 found it) |
| Q8 | comparative | "Deep Think vs o3-pro?" | 0 | seeallochnaya:2711 not in top-10 |
| Q9 | multi-hop | "LLM в production: llm_under_hood + boris_again?" | 0 | Found wrong msg_ids from right channels |
| Q10 | negative | "GPT-6?" | N/A | Correctly refused |

### Root Cause Analysis

**Problem 1: Dense embedding "attractor documents"**
Cosine similarity between ANY AI-related query and certain generic posts is 0.78-0.83. These "attractor" documents (nanochat announcement, Tinker API, conference listings) appear in top-10 for EVERY query. The embedding model collapses diverse AI topics into a narrow region of embedding space.

Evidence:
```
Query: "Meta купила Manus AI" → dense top-3:
  0.82 | gonzo_ml:4069 (Tinker API — IRRELEVANT)
  0.81 | gonzo_ml:4121 (nanochat — IRRELEVANT)
  0.81 | rybolos_channel:1561 (ACL 2025 — IRRELEVANT)

Same query → BM25 top-3:
  21.3 | aioftheday:3988 (Meta покупает Manus — RELEVANT!)
  21.0 | ai_newz:4355 (Meta купила Manus за $2 млрд — RELEVANT!)
  18.2 | data_secrets:8582 (Meta купила Manus — RELEVANT!)
```

BM25 perfectly finds the answer. Dense embeddings completely miss it.

**Problem 2: RRF works but dense still dilutes**
After RRF fusion, BM25-found documents get into the candidate pool, but they compete with dense-found "attractors." When we had dense re-score after RRF, it completely killed BM25 results. Without re-score, RRF keeps BM25 contributions but they still need to outweigh dense noise.

**Problem 3: LLM query expansion loses entities**
LLM transforms "Meta bought Manus AI for $2B" → ["meta покупка manus стоимость", "цена приобретения manus"]. The original phrasing (which BM25 would match perfectly) is lost. We're fixing this by always including the original query.

**Problem 4: No query-type-aware strategy**
Channel-specific queries ("what did llm_under_hood write about...") should filter by channel. Temporal queries should use date filters. Comparative queries need parallel multi-query. Currently all go through the same pipeline.

**Problem 5: MMR doesn't help**
Classic cosine-based MMR re-promotes "attractor documents" because they have high cosine with query. Tested lambda 0.7 and 0.9 — both dropped recall from 0.33 to 0.11.

### Ablation Results

| Configuration | Recall@5 | Notes |
|---------------|----------|-------|
| RRF → dense re-score (original) | 0.15 | Dense kills BM25 |
| **Pure RRF** | **0.33** | Best so far |
| RRF → MMR (lambda=0.7) | 0.11 | Cosine MMR re-promotes attractors |
| RRF → MMR (lambda=0.9) | 0.11 | Same even with high relevance weight |

## What I'm Looking For

### 1. Solving the "Attractor Document" Problem

Dense embeddings collapse diverse topics into narrow cosine range (0.78-0.83). How to fix this without abandoning dense search?

Specifically interested in:
- **Weighted RRF** — can we give BM25 branch higher weight in fusion? (e.g., BM25 weight=2x dense weight?)
- **Late interaction models** (ColBERT, ColPali) — per-token matching should be more discriminative than single-vector cosine. Do they solve this?
- **Contextual retrieval** (Anthropic's technique) — prepending "This is a post from channel X about topic Y, published on date Z" before embedding. Would this push different topics apart in embedding space?
- **Larger embedding model** — would Qwen3-Embedding-1.5B or 7B fix the collapse? Or is it fundamental to 0.6B scale?
- **Fine-tuning embedding** — is it feasible to fine-tune Qwen3-Embedding on our corpus with contrastive learning?
- **Learned sparse (SPLADE)** — would SPLADE outperform BM25 fastembed? Any multilingual SPLADE models?
- **Hypothetical Document Embedding (HyDE)** — generate a fake answer, embed it, search for similar. Does this help when the corpus is short informal posts?

### 2. Diversity Without Cosine

MMR fails because cosine is the broken signal. What alternatives exist?

- **Channel-based dedup** — max N docs per channel in results
- **Temporal spread** — ensure results cover multiple dates
- **Cluster-first diversity** — cluster documents, pick from different clusters
- **Submodular function optimization** — formalize diversity as facility location problem
- **Novelty detection** — penalize docs similar to already-seen ones via BM25 overlap, not cosine

### 3. Adaptive / Multi-Strategy Retrieval

This is the key differentiator from frameworks. How to implement:

- **Query classifier** → selects strategy (factual/temporal/channel/comparative/multi-hop)
- **Tool router** → gives agent only relevant tools per query type
- **Iterative retrieval** — first search → analyze gaps → targeted second search
- **Multi-index** — separate indexes for news, research, opinions (different embedding strategies)
- **Self-reflection retrieval** — agent evaluates its own search results quality before answering

Any concrete implementations with measured improvements?

### 4. News/Social Media Specific Techniques

Our corpus is Telegram posts: short, informal, multilingual (RU/EN mix), with links and emojis.

- **Entity extraction → filter pipeline** — extract named entities from query, use as Qdrant filters
- **Temporal decay** — weight by freshness or by distance from queried time period
- **Channel authority scoring** — not all channels are equal; weight by author expertise
- **Forward/reply chain awareness** — some posts are part of discussion threads
- **Link expansion** — should we index the linked content (papers, blog posts)?

### 5. Reranker Optimization

Our BGE-M3 reranker shows scores very close to 0 (after sigmoid). Is this normal? Should we:
- Use raw logits instead of sigmoid?
- Try a different reranker model?
- Fine-tune BGE-M3 on our domain?
- Use reranker earlier in pipeline (before RRF)?

### 6. Benchmarks & Realistic Targets

What recall@5 / recall@10 / MRR numbers are realistic for:
- RAG over short social media posts (not long docs)?
- Domain-specific corpora with 10K-50K documents?
- Russian-language retrieval?
- Is 0.33 terrible or normal for this corpus type?

What do LlamaIndex / LangChain / Haystack / Ragatouille achieve on comparable tasks?

### 7. Concrete Implementations

For each technique, I need:
- **Paper/blog with measured results** (not just theory)
- **Code examples** or library integrations (Python, Qdrant-compatible preferred)
- **Expected recall improvement** (rough estimate based on literature)
- **Implementation complexity** (hours / days / weeks)
- **Dependencies** (new models, compute requirements, etc.)

## Output Format

For each recommended technique:
1. **Name** and one-line description
2. **How it works** (2-3 sentences, concrete)
3. **Why it helps our specific case** (address attractor documents, entity loss, etc.)
4. **Expected recall improvement** (cite evidence)
5. **Implementation complexity** and dependencies
6. **Priority** (must-have / should-have / nice-to-have for portfolio)

**Prioritize by impact/effort ratio.** I need actionable engineering recommendations, not a literature survey. Each recommendation should be something I can implement in 1-3 days and measure on our golden dataset.

## Hardware Constraints

- **V100 SXM2 32GB** (Windows host, TCC mode) — LLM inference only
- **RTX 5060 Ti 16GB** (WSL2 native, PyTorch cu128) — embedding + reranker. No Docker GPU (V100 TCC blocks NVML). No Flash Attention (sm_120 not supported).
- Total latency budget: <30s per query end-to-end
- Python 3.11, Qdrant, FastAPI, no cloud APIs
