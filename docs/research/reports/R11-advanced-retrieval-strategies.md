# Fixing 0.33 recall@5 in a Russian Telegram RAG pipeline

**Your most likely bottleneck is a model confusion: you may be using BGE-M3 (a bi-encoder embedding model) as your "reranker" instead of the actual cross-encoder `bge-reranker-v2-m3`.** Combined with unweighted RRF that lets dense attractors dilute BM25's correct results, and an evaluation set too small for statistical significance, three quick fixes could double your recall@5 within a day. Qdrant v1.17 (released February 2026) now supports native weighted RRF, and inverting your pipeline to use BM25 as the primary retriever with dense as reranker completely eliminates the attractor problem. A realistic target after optimization is **0.55–0.70 recall@5**, achievable through the prioritized interventions below.

---

## The reranker misconfiguration that likely explains everything

The single most important finding from this research: **`BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` are two completely different models.** BAAI developer Shitao has confirmed on HuggingFace that bge-m3 is a bi-encoder — its `compute_score()` summarizes embedding similarity modes, not cross-encoder relevance. If you're using `BGEM3FlagModel` with `compute_score()`, you are *not* doing cross-encoder reranking.

The near-zero sigmoid scores you observe are actually **normal behavior for the real cross-encoder** `bge-reranker-v2-m3` on non-relevant pairs. The model's official examples show raw logits of -8.19 for irrelevant and +5.26 for relevant pairs, mapping to sigmoid scores of 0.0003 and 0.9948 respectively. **Use raw logits for ranking, not sigmoid thresholds.**

Correct usage of the actual reranker:

```python
# CORRECT: Cross-encoder reranker
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', devices=["cuda:0"], use_fp16=True)
pairs = [[query, doc.text] for doc in candidates]
scores = reranker.compute_score(pairs)  # raw logits for ranking

# WRONG: Bi-encoder embedding similarity (not reranking)
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3')  # This is NOT a reranker
```

**Expected impact:** If this misconfiguration exists, fixing it alone could push recall@5 from 0.33 to 0.50–0.60. **Effort: 1 hour.** Priority: **MUST-HAVE — do this first.**

---

## Weighted RRF and pipeline inversion defeat attractor documents

Qdrant v1.17 introduced native per-prefetch weights in RRF fusion. Since BM25 already finds correct documents but dense search dilutes them with attractors, **weighting BM25 3:1 over dense** directly suppresses the attractor effect. The weighted RRF formula adjusts rank contributions so a document ranked 3rd in BM25 scores equivalently to a document ranked 1st in dense search.

```python
client.query_points(
    collection_name="telegram_posts",
    prefetch=[
        models.Prefetch(query=bm25_query, using="sparse", limit=100),
        models.Prefetch(query=dense_query, using="dense", limit=20),
    ],
    query=models.RrfQuery(rrf=models.Rrf(weights=[3.0, 1.0], k=2)),
    limit=20,
)
```

An even more aggressive fix is **pipeline inversion** — use BM25 as the primary retriever, then rerank with dense embeddings. This completely eliminates attractors because documents not in BM25's top-50 never enter the pipeline:

```python
client.query_points(
    collection_name="telegram_posts",
    prefetch=models.Prefetch(query=bm25_query, using="sparse", limit=50),
    query=dense_query, using="dense",  # Dense only reranks BM25 candidates
    limit=20,
)
```

The risk with pipeline inversion is losing recall on vague semantic queries where BM25 fails. **The recommended approach is weighted RRF (BM25 3:1) with asymmetric prefetch limits (100 BM25, 20 dense), followed by cross-encoder reranking of the fused top-50.** Also test DBSF (Distribution-Based Score Fusion), available since Qdrant v1.11, which normalizes scores using mean ± 3σ and may better differentiate within the tight 0.78–0.83 cosine range.

Two additional Qdrant-native options are worth noting. The **formula query** (v1.14+) can penalize attractor documents if you pre-compute a "genericity score" for each document — count how many random queries each doc appears in top-10, store as payload, then apply a penalty. **Relevance Feedback Query** (v1.17) uses positive/negative example pairs to iteratively refine scoring.

**Expected impact:** Weighted RRF alone: recall@5 0.33 → 0.45+. Combined with pipeline inversion or proper reranking: → 0.55+. **Effort: 1–2 hours.** Priority: **MUST-HAVE.**

---

## ColBERT reranking fundamentally solves embedding collapse

ColBERT produces one vector per token rather than compressing the entire document into a single vector. At query time, it computes MaxSim: for each query token, find the maximum similarity with any document token, then sum. This fundamentally avoids the collapse problem — two posts about "transformers" vs "diffusion models" have similar single-vector embeddings but very different token-level profiles.

**jina-colbert-v2** (560M params, XLM-RoBERTa base) is the only production-ready multilingual ColBERT model with explicit Russian training. It covers **89 languages including Russian**, achieves +6.5% over ColBERTv2 on BEIR (nDCG@10 = 0.521), and outperforms BM25 on all MIRACL languages. At ~1.1 GB in FP16, it fits comfortably on the RTX 5060 Ti alongside the embedding model.

Qdrant supports ColBERT natively via multi-vector configuration since v1.10:

```python
# Collection config
"colbert": models.VectorParams(
    size=128, distance=models.Distance.COSINE,
    multivector_config=models.MultiVectorConfig(
        comparator=models.MultiVectorComparator.MAX_SIM),
    hnsw_config=models.HnswConfigDiff(m=0)  # Brute-force for reranking stage
)

# Multi-stage: BM25+Dense → RRF → ColBERT rerank
client.query_points(
    collection_name="telegram_posts",
    prefetch=models.Prefetch(
        prefetch=[
            models.Prefetch(query=bm25_query, using="sparse", limit=100),
            models.Prefetch(query=dense_query, using="dense", limit=100),
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[3.0, 1.0])),
        limit=30,
    ),
    query=colbert_vectors, using="colbert", limit=5,
)
```

For **13K docs** averaging ~100 tokens each, ColBERT storage is ~333–666 MB (manageable), and brute-force MaxSim on 30 prefetched candidates takes milliseconds. An alternative is **BGE-M3's ColBERT mode**, which provides dense+sparse+ColBERT from a single ~1.8 GB model — attractive for simplicity but lower quality than jina-colbert-v2 for Russian.

**Expected impact:** +6–10% nDCG@10 from ColBERT reranking stage. **Effort: 2–3 days** (encode all documents with jina-colbert-v2, add multi-vector config to Qdrant). Priority: **SHOULD-HAVE** — implement after the quick wins above.

---

## Embedding whitening and larger models widen the cosine range

A zero-cost fix for the 0.78–0.83 cosine collapse is **mean-centering and whitening** the embedding space:

```python
import numpy as np
from sklearn.decomposition import PCA
mean = np.mean(all_embeddings, axis=0)
centered = all_embeddings - mean
# Optional: full whitening
pca = PCA(whiten=True, n_components=1024)
whitened = pca.fit_transform(all_embeddings)
```

Research on embedding anisotropy (Liang et al.) confirms that pretrained language models produce vectors clustered in a narrow cone. Mean-centering spreads the distribution; whitening normalizes covariance to identity. **This can immediately widen the cosine similarity range with zero retraining cost.** Apply the same transform to query embeddings at search time.

Regarding larger models: **Qwen3-Embedding comes in 0.6B, 4B, and 8B sizes** (the 1.5B and 7B variants do not exist). The 8B model ranks #1 on MTEB multilingual (70.58 vs 64.64 for 0.6B), but at ~16 GB VRAM it leaves no room for the reranker. **Qwen3-Embedding-4B** (~8 GB VRAM) is the sweet spot — it fits alongside a reranker on 16 GB and provides a meaningful quality uplift. However, the "Length-Induced Embedding Collapse" paper (arXiv:2410.24200) shows that similar-length texts cluster together regardless of content, which is particularly relevant for Telegram posts of similar length. **Scaling alone won't fully solve domain-specific collapse** on a topically narrow corpus.

**Expected impact:** Whitening: variable but could spread cosine range from [0.78–0.83] to [0.5–0.9]. Model upgrade to 4B: +5–10% on retrieval benchmarks. **Effort:** Whitening: 2 hours. Model upgrade: 0.5–1 day. Priority: Whitening is **MUST-HAVE** (free win); model upgrade is **SHOULD-HAVE.**

---

## Fine-tuning the embedding model directly targets the collapse

Contrastive fine-tuning with hard negatives explicitly pushes similar-but-different documents apart, directly reducing anisotropy. The strategy for this corpus:

1. **Generate synthetic queries** using Qwen3-30B-A3B: for each of 13K posts, generate 3 search queries → ~39K positive pairs
2. **Mine hard negatives** from the current (broken) embedding space: for each query-post pair, the top-50 most similar non-positive posts become hard negatives — these *are* the attractor documents
3. **Train** with sentence-transformers MultipleNegativesRankingLoss (InfoNCE):

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
loss = MultipleNegativesRankingLoss(model)
# train_dataset columns: "anchor" (query), "positive" (relevant post), "negative" (attractor doc)
trainer = SentenceTransformerTrainer(model=model, train_dataset=train_dataset, loss=loss, ...)
trainer.train()
```

Aurelio AI benchmarks show fine-tuning with ~5K pairs on a domain-specific task brought a smaller model to parity with much larger ones. NV-Retriever showed hard negative mining improved nDCG@10 by **2–5 points** on BEIR. The InfoNCE loss optimizes for embedding uniformity (Wang & Isola, 2020), directly countering the collapse.

**Expected impact:** +5–15% recall depending on data quality. **Effort:** 3–5 days (1–2 days data generation, 1 day training, 1–2 days evaluation/iteration). Priority: **SHOULD-HAVE** — high impact but requires the most time investment.

---

## Diversity must use orthogonal signals since cosine is broken

MMR fails because it relies on the same collapsed cosine distances. The solution is layering **metadata-based**, **lexical**, and **structural** diversity signals:

**Channel-based dedup (MUST-HAVE, 1 hour):** Cap results at max 2 docs per Telegram channel. Qdrant's `group_by` parameter handles this natively. This is the single highest-ROI diversity intervention — prolific channels that produce attractor documents can't monopolize results.

**BM25-based novelty (MUST-HAVE, 1 day):** Replace cosine-based MMR with BM25 term-overlap diversity. Two AI posts about "трансформеры" vs "диффузионные модели" have similar embeddings but very different BM25 profiles. Implementation is a modified MMR loop using BM25 pairwise similarity instead of cosine:

```python
def bm25_diverse_rerank(candidates, k=5, lambda_=0.5):
    selected = [candidates[0]]
    for _ in range(k - 1):
        best = max(remaining, key=lambda doc:
            lambda_ * doc.score - (1 - lambda_) * max(bm25_sim(doc, s) for s in selected))
        selected.append(best)
    return selected
```

**Cluster-first diversity (SHOULD-HAVE, 2–3 days):** Pre-cluster documents using **UMAP dimensionality reduction → HDBSCAN**, then retrieve top doc from each cluster. Critical: cluster on **TF-IDF/BM25 features** rather than collapsed embeddings, or use BERTopic with UMAP+HDBSCAN. Store cluster IDs as Qdrant payload for runtime filtering.

**DPP (Determinantal Point Processes) (SHOULD-HAVE, 1–2 days):** Mathematically principled diversity selection using the `dppy` library. DPPs use the full kernel matrix structure rather than just pairwise comparisons, making them more sensitive to subtle differences. YouTube's production recommendation system uses DPP-based diversification. DPP consistently shows **5–15% improvements** in diversity metrics over MMR.

**Temporal spread (SHOULD-HAVE, 0.5 day):** Round-robin from time buckets (weekly/monthly) to ensure results span different periods. Qdrant payload filtering by `post_date` makes this trivial. Combined with temporal decay for recency-sensitive queries: `fused_score = 0.7 × semantic + 0.3 × 0.5^(age_days/14)`.

---

## Query classification and agentic self-reflection unlock adaptive retrieval

Since different query types need different retrieval strategies, a **query classifier** that routes to specialized pipelines delivers large gains. Enterprise deployments report **8% accuracy improvement and 35% latency reduction** from query routing.

```python
STRATEGIES = {
    "simple":     {"top_k": 3,  "diversity": "channel_dedup"},
    "temporal":   {"top_k": 10, "date_filter": "last_30d", "diversity": "temporal_buckets"},
    "comparative":{"top_k": 8,  "diversity": "bm25_mmr", "reranker": True},
    "multi_hop":  {"top_k": 5,  "diversity": "cluster_first", "iterative": True},
    "trend":      {"top_k": 15, "diversity": "cluster+temporal"},
}
```

The classification itself can be done by Qwen3-30B-A3B with a structured prompt returning JSON. For the ReAct agent, add **self-reflection retrieval**: after each search, the agent evaluates result quality (relevance, coverage, diversity), identifies gaps, and optionally reformulates the query — capped at 3 iterations. This pattern naturally fits ReAct's reasoning-acting loop. CRAG (Corrective RAG) by Yan et al. showed that adding a lightweight retrieval evaluator that can discard irrelevant results and retry with refined queries **significantly outperforms standard RAG and Self-RAG** across 4 datasets.

An implementation report showed **accuracy jumping from 58% to 83%** after adding retrieval evaluation gates with query rewriting. The key insight: if 20–30% of queries return poor answers, an evaluation-retry loop cuts that failure rate roughly in half.

**Expected impact:** Query routing: +5–10%. Self-reflection: +10–15% on complex queries. **Effort:** Query classifier: 1 day. Self-reflection: 1–2 days. Priority: both **MUST-HAVE** given the ReAct agent already supports iterative tool use.

---

## Social media and Telegram-specific optimizations

**Entity extraction with Natasha/Slovnet (MUST-HAVE, 1–2 days):** The Natasha library's Slovnet NER model is 27 MB, runs on CPU at 25 articles/sec, achieves F1 ~0.96 (only 1pp below SOTA DeepPavlov BERT NER). Extract PER/ORG/LOC entities from all 13K posts (~10 minutes processing), store as Qdrant payload, and apply entity-based filtering when queries mention specific people, companies, or tools. Expected: **+5–15% recall on entity-specific queries.**

**Contextual retrieval (SHOULD-HAVE, 2–4 days):** Anthropic's technique of prepending LLM-generated context before embedding. For Telegram posts, generate a 2–3 sentence prefix explaining the specific AI topic and distinguishing features. Anthropic measured **35–67% retrieval failure reduction**. The technique helps even for standalone short posts because it adds disambiguation signals — posts about "new model release" vs "model benchmarking" get explicitly different prefixes that push embeddings apart. Processing 13K posts with Qwen3-30B-A3B on V100: ~8–15 hours (one-time).

**Link expansion (SHOULD-HAVE, 2–3 days):** Many Telegram AI posts are "link + brief comment." Fetching and summarizing linked content, then indexing as expanded documents, could yield **+5–15% recall** on queries whose answer lives behind links rather than in post text.

**Channel authority scoring (SHOULD-HAVE, 0.5 day):** Manually assign 0–1 authority scores to each channel based on subscriber count, known expertise, and forward patterns. Store as Qdrant payload for score boosting. ReliabilityRAG (2025) demonstrated that explicit source reliability signals improve answer accuracy especially when documents conflict.

**Forward/reply chain awareness (SHOULD-HAVE, 1–2 days):** Store `reply_to_id` and `forwarded_from` as payload. When a relevant post is retrieved, also fetch its reply chain for expanded context. HyDE is deprioritized here — it adds 1–3s latency per query for LLM generation and **does not fix the underlying document-space collapse**, only changes the query representation. It's a **NICE-TO-HAVE** complementary technique.

---

## Your evaluation dataset is too small to measure anything reliably

With 10 questions and binary relevance, a single question flip changes recall@5 by **10 percentage points**. The 95% confidence interval for p=0.33 with n=10 spans roughly [0.10, 0.65] — your true recall could be anywhere in that range. Multiple evaluation guides recommend a **minimum of 50 questions** for regression testing and **200+ for statistical significance**.

The fastest path to a reliable evaluation set: use RAGAS TestsetGenerator or a custom pipeline with Qwen3-30B-A3B to generate 2–3 QA pairs per sampled post, filter with critique prompts (groundedness ≥4/5), then supplement with 30–50 manually crafted questions. **Estimated effort: 1–2 days for 200–500 quality QA pairs.**

For context on realistic targets: **0.33 recall@5 is below average but explainable** given the small embedding model (0.6B), short informal Russian text, narrow topic domain, and likely reranker misconfiguration. Production RAG systems typically achieve 0.70–0.85 recall@5 on well-structured corpora. For short social media posts in a non-English language, **0.55–0.70 recall@5 is a realistic post-optimization target**, with 0.75+ as a stretch goal.

SPLADE for Russian is not viable today — no production-ready multilingual SPLADE model exists, and the WSDM Cup 2026 results show learned sparse retrieval "struggled to remain competitive" on multilingual benchmarks. **Stick with BM25 via fastembed** for your sparse signal.

---

## Reranker alternatives that fit RTX 5060 Ti 16GB

| Model | Params | VRAM | MIRACL Avg | Best For |
|---|---|---|---|---|
| **bge-reranker-v2-m3** | 568M | ~1.2 GB | 69.32 | Best multilingual, production-proven |
| **jina-reranker-v2-base-multilingual** | 278M | ~0.6 GB | Competitive | Fastest (15× throughput), set `use_flash_attn=False` for sm_120 |
| **jina-reranker-v3** | 0.6B | ~1.2 GB | 66.50 | BEIR SOTA (61.94), Russian MIRACL 65.20 |
| **bge-reranker-v2-gemma** | 2.5B | ~5 GB | Higher | Best quality that fits alongside embedding model |

For **sm_120 (RTX 5060 Ti)**, Flash Attention is not supported. Use **PyTorch SDPA** (Scaled Dot-Product Attention) as the backend — install PyTorch nightly with cu128/cu129. For Jina models, set `use_flash_attn=False`. BGE rerankers use standard XLM-RoBERTa attention and work with SDPA automatically.

The optimal reranking pipeline: retrieve top-50 from BM25 + top-50 from dense, deduplicate (~70–100 unique candidates), **rerank entire pool with cross-encoder** → top-5. This uses the reranker *as* the fusion mechanism rather than RRF→rerank, which can be more effective since the cross-encoder makes content-aware relevance judgments. Latency for 70–100 candidates on RTX 5060 Ti: ~150–300ms for bge-reranker-v2-m3, well within the 30s budget.

---

## Prioritized implementation roadmap

The following is ordered by **impact/effort ratio**, with estimated cumulative recall@5 improvements:

| Phase | Technique | Effort | Δ Recall@5 | Cumulative | Priority |
|---|---|---|---|---|---|
| **Day 1** | Fix reranker model (use `bge-reranker-v2-m3` cross-encoder) | 1 hr | +0.15–0.25 | ~0.50 | MUST-HAVE |
| **Day 1** | Weighted RRF (BM25 3:1, k=2) + asymmetric prefetch | 1 hr | +0.05–0.10 | ~0.55 | MUST-HAVE |
| **Day 1** | Embedding whitening (mean-centering) | 2 hrs | +0.03–0.08 | ~0.60 | MUST-HAVE |
| **Day 1** | Channel-based dedup (max 2 per channel) | 1 hr | +0.02–0.05 | ~0.62 | MUST-HAVE |
| **Day 2** | Expand golden dataset to 200+ questions | 1 day | Enables measurement | — | MUST-HAVE |
| **Day 2** | Test DBSF fusion as alternative to RRF | 30 min | Variable | — | MUST-HAVE |
| **Day 3** | Entity extraction (Natasha) + payload filtering | 1 day | +0.03–0.08 | ~0.65 | MUST-HAVE |
| **Day 3** | Query classifier + strategy routing | 1 day | +0.03–0.05 | ~0.68 | MUST-HAVE |
| **Week 2** | Self-reflection retrieval in ReAct agent | 1–2 days | +0.03–0.07 | ~0.72 | SHOULD-HAVE |
| **Week 2** | ColBERT reranking (jina-colbert-v2) | 2–3 days | +0.03–0.06 | ~0.75 | SHOULD-HAVE |
| **Week 2** | BM25-based novelty reranking | 1 day | +0.02–0.04 | ~0.77 | SHOULD-HAVE |
| **Week 3** | Contextual retrieval (LLM-generated prefixes) | 2–4 days | +0.03–0.08 | ~0.80 | SHOULD-HAVE |
| **Week 3** | Fine-tune Qwen3-Embedding-0.6B on synthetic pairs | 3–5 days | +0.05–0.10 | ~0.85 | SHOULD-HAVE |
| **Week 4** | Link expansion, temporal decay, DPP diversity | 3–4 days | +0.03–0.07 | ~0.87 | NICE-TO-HAVE |
| **Week 4** | Upgrade to Qwen3-Embedding-4B | 0.5 day | +0.02–0.05 | ~0.89 | NICE-TO-HAVE |

## Conclusion

The path from 0.33 to 0.65+ recall@5 runs through three quick wins implementable on Day 1: **fixing the reranker model identity**, **weighting BM25 3:1 in RRF fusion**, and **whitening embeddings**. These require no new models, no reindexing, and under 4 hours of work. The attractor document problem stems from a confluence of dense embedding anisotropy, equal-weight RRF fusion, and likely incorrect reranking — not a fundamental limitation of the approach.

The deeper architectural insight is that **embedding collapse on a topically narrow corpus is expected behavior**, not a bug. Single-vector models compress all of "AI discourse" into a narrow manifold. The sustainable fixes — ColBERT's per-token matching, contrastive fine-tuning with hard negatives mined from attractor documents, and contextual retrieval's disambiguation prefixes — all attack different facets of this fundamental compression problem. Layer them incrementally, measuring against a properly sized golden dataset (200+ questions), and the system should converge on 0.75+ recall@5 within 2–3 weeks of focused engineering.