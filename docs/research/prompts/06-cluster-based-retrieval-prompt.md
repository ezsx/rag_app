# Deep Research: Topic-Based Clustering for RAG Retrieval

## Project Context

Portfolio-grade RAG system for Applied LLM Engineer interviews. Telegram-channel aggregator: 36 AI/ML Russian-language channels, ~13,000 posts (July 2025 – March 2026), self-hosted on V100 32GB (LLM) + RTX 5060 Ti 16GB (embedding + reranker).

### Full Pipeline

```
User Query → ReAct Agent (Qwen3-30B-A3B, native function calling)
  → query_plan: LLM generates sub-queries + metadata filters
  → search: each sub-query → Qdrant hybrid search (dense 1024-dim + BM25 sparse → RRF fusion)
  → rerank: BGE-M3 cross-encoder re-scores top candidates
  → compose_context: builds context window with citations
  → final_answer: LLM generates grounded answer
```

### Tech Stack
- **Qdrant** v1.13: single collection "news", named vectors (dense_vector: 1024-dim cosine, sparse_vector: BM25 via fastembed Qdrant/bm25)
- **Embedding**: Qwen3-Embedding-0.6B (latest, top MTEB, but 0.6B params). Running natively via PyTorch cu128 on RTX 5060 Ti.
- **Reranker**: BAAI/bge-m3 as AutoModelForSequenceClassification (planned upgrade to bge-reranker-v2-m3)
- **LLM**: Qwen3-30B-A3B-Q4_K_M via llama-server on V100 SXM2 32GB
- **Corpus**: 13,124 points. Posts are 300-1500 chars, mix Russian/English, links, emojis, code. Chunking: posts <1500 chars = single chunk, >1500 = recursive split with overlap.

### The Core Problem We're Solving

**Recall@5 = 0.59** on a 10-question golden dataset (started at 0.15, improved through RRF fix and original query injection).

**Root cause**: Dense embeddings collapse on this domain. All AI-related posts score cosine 0.78-0.83 with ANY query. We call them "attractor documents" — generic popular posts (nanochat announcement, Tinker API, conference listings) that appear in top-10 for EVERY query regardless of actual topic.

Evidence from direct Qdrant search:
```
Query: "Meta купила Manus AI" (Meta bought Manus AI)

Dense search (cosine) top-3:
  0.82 | gonzo_ml — Tinker API announcement (IRRELEVANT)
  0.81 | gonzo_ml — nanochat repo (IRRELEVANT)
  0.81 | rybolos_channel — ACL 2025 papers (IRRELEVANT)

BM25 search top-3:
  21.3 | aioftheday — "Meta покупает Manus" (RELEVANT!)
  21.0 | ai_newz — "Meta купила Manus за $2 млрд" (RELEVANT!)
  18.2 | data_secrets — "Meta купила Manus" (RELEVANT!)
```

BM25 keyword matching works perfectly. Dense embeddings completely fail — they can't distinguish "Meta acquisition news" from "transformer course announcement" because both are "AI-related text."

### What We've Already Tried

| Approach | Recall@5 | Result |
|----------|----------|--------|
| RRF → dense re-score | 0.15 | Dense re-score kills BM25 signal |
| Pure RRF | 0.33 | Best fusion approach so far |
| RRF → MMR (lambda=0.7) | 0.11 | Cosine MMR re-promotes attractors |
| RRF → MMR (lambda=0.9) | 0.11 | Same problem with any lambda |
| + Original query in subqueries | **0.59** | Big win from BM25 keyword match |

Key finding: **cosine similarity is a broken signal** on this corpus. Any technique relying on cosine (MMR, dense re-scoring) makes things WORSE.

### Paper Reference: "Length-Induced Embedding Collapse"

arXiv:2410.24200 shows that texts of similar length cluster together in embedding space regardless of content. Our Telegram posts are ALL 300-1500 chars — they cluster by length, not by meaning. This is a fundamental limitation of single-vector embeddings on homogeneous corpora.

---

## Our Idea: Topic-Based Clustering

### The Intuition

Imagine a library where ALL books are shelved in one giant room. Looking for a book about "Meta's acquisition strategy" means scanning past thousands of AI textbooks, papers, and tutorials — they all look similar from the spine.

Now imagine the library organized into sections: "Industry M&A", "Model Releases", "Research Papers", "Tutorials", "Infrastructure". Suddenly, finding the Meta acquisition book is trivial — go to "Industry M&A", and it's right there. Within that section, books are actually different from each other.

**This is exactly our problem.** One Qdrant collection = one giant room. All AI posts have cosine 0.78-0.83. But within "Industry M&A" cluster, the Meta/Manus post would stand out clearly from posts about Nvidia/ARM acquisition.

### Proposed Architecture

```
OFFLINE (one-time, ~30 min):
  All 13K documents
    → TF-IDF vectorization (captures lexical topics, not broken like dense embeddings)
    → UMAP dimensionality reduction (50K sparse → 15-20 dense dims)
    → HDBSCAN clustering → 20-50 topic clusters
    → Each document gets cluster_id stored as Qdrant payload
    → Per-cluster: compute mean embedding vector for whitening

ONLINE (at search time):
  User query
    → Determine relevant clusters (3-5 out of 20-50)
    → Qdrant search with filter: cluster_id IN [relevant_clusters]
    → Per-cluster whitening applied to dense scores
    → RRF fusion (BM25 + whitened dense)
    → Reranker → final results
```

**Critical design choice**: ONE Qdrant collection, clusters via payload filtering. NOT 50 separate collections. Reason: BM25's IDF needs the full corpus breadth to compute meaningful term weights. Splitting into small collections would destroy BM25's statistical power.

### What This Gives Us

1. **Within-cluster cosine is meaningful**: posts about "model releases" compared to other posts about "model releases" — embeddings can differentiate
2. **Per-cluster whitening is statistically sound**: mean-centering from a topically homogeneous group is more principled than global mean
3. **Attractor documents are contained**: gonzo_ml:4121 (nanochat) stays in "open-source tools" cluster, doesn't pollute "M&A news" queries
4. **Natural diversity**: results from different clusters = different topics
5. **Scales to 50-100K**: adding documents just assigns to existing clusters (or triggers re-clustering periodically)

### Open Questions (What We Need Research On)

**Q1: Clustering method.** TF-IDF + HDBSCAN is the standard, but:
- Should we use BERTopic (adds LLM topic naming, c-TF-IDF, hierarchical)?
- Cluster on TF-IDF or on a different (non-collapsed) embedding model?
- What about clustering on BM25 sparse vectors directly (they work well for us)?

**Q2: How many clusters?** 13K docs / N clusters = avg cluster size.
- N=20 → 650 docs/cluster (decent for search, but topics might be too broad)
- N=50 → 260 docs/cluster (specific topics, but small for stats)
- Is there a principled method? Elbow, silhouette, HDBSCAN's natural clusters?

**Q3: Query → Cluster routing.** The hardest part. How does the system know which 3-5 clusters to search?
- Option A: Search ALL clusters, but apply per-cluster whitening → still helpful?
- Option B: Embed query → nearest cluster centroids (but centroids from collapsed space?)
- Option C: TF-IDF query → nearest cluster TF-IDF centroids (lexical, might work)
- Option D: LLM classifies query → top clusters (+5-8s latency, expensive)
- Option E: BM25 search cluster descriptions (fast, simple)
- Option F: Two-stage — coarse BM25 full-collection → see which clusters in top-50 → focused search
- Which is best for latency <30s and accuracy?

**Q4: Per-cluster whitening mechanics.**
- Mean-centering only, or full PCA whitening? (200-600 docs per cluster — enough for PCA?)
- How to apply at query time: transform query with EACH cluster's whitening matrix, or global?
- Does this require re-indexing vectors in Qdrant, or can we do it on-the-fly?

**Q5: Soft vs hard assignment.** Post about "Meta bought AI agent for Llama" = M&A + AI agents + model ecosystem.
- Duplicate document in multiple clusters?
- Store list of cluster_ids as payload?
- Fuzzy membership scores?

**Q6: Incremental updates.** Daily ingest of new posts.
- Assign to nearest existing cluster centroid?
- Re-cluster periodically (weekly)? Or only when cluster quality degrades?

**Q7: Is this worth the complexity?** Compared to simpler alternatives:
- Global embedding whitening (no clustering needed)
- Weighted RRF (just increase BM25 weight)
- Better reranker model
- ColBERT (per-token matching, fundamentally solves collapse)
- Would clustering give significantly more than these simpler approaches combined?

---

## What I'm Looking For

### Concrete Answers With Evidence

For each open question above, I need:
1. **Recommended approach** with justification (not "it depends")
2. **Evidence**: papers, benchmarks, production examples
3. **Code example** in Python (sklearn, umap-learn, hdbscan, qdrant-client)
4. **Expected impact** on recall metrics
5. **Gotchas** and failure modes

### Specific Deliverables

1. **Complete clustering pipeline code**: from raw texts to cluster_ids in Qdrant
2. **Query routing strategy**: with code and latency analysis
3. **Per-cluster whitening implementation**: including query-time application
4. **Comparison with alternatives**: is topic clustering worth it vs global whitening + weighted RRF?
5. **Incremental update strategy**: for daily ingest
6. **Evaluation methodology**: how to measure if clustering actually helped (A/B test design)

### Production Examples

Are there documented cases of:
- RAG systems using topic-clustered retrieval?
- Qdrant payload-based partitioning for improving retrieval quality?
- BERTopic + vector DB integration in production?
- Multi-index RAG architectures with routing?

## Constraints

- **Single Qdrant collection** (payload filtering, not multi-collection)
- **CPU-only for clustering** (GPU budget is for embedding/reranker/LLM)
- **13K documents now, will grow to 50-100K**
- **Russian/English mix**, short texts (300-1500 chars)
- **Latency budget**: <30s end-to-end per query
- **Python**: sklearn, umap-learn, hdbscan, qdrant-client, BERTopic available
- Already have: Qdrant with dense + sparse vectors, RRF fusion, BM25 working well

## Output Format

Structure the response as:
1. **Executive summary**: is topic clustering worth it for our case? Yes/no with evidence
2. **Recommended architecture**: step-by-step with code
3. **Answers to each open question** (Q1-Q7) with evidence
4. **Complete implementation guide**: from clustering to search integration
5. **Comparison table**: clustering vs simpler alternatives (whitening, weighted RRF, ColBERT)
6. **Risk assessment**: what could go wrong, how to detect and fix
