# Topic clustering for RAG: probably not your best first move

**For a 13K-post Qdrant collection suffering from dense embedding collapse (cosine 0.78–0.83), topic-based clustering is a defensible but over-engineered first intervention.** Global embedding whitening combined with a cross-encoder reranker will likely deliver **higher recall gains at one-fifth the implementation cost**. Clustering becomes valuable at scale (50K+ docs) or when simpler fixes plateau. The recommended path: start with global PCA whitening + weighted RRF tuning (4 hours, +5–15% recall), add bge-reranker-v2-m3 (8 hours, +15–30%), then evaluate whether clustering or a BGE-M3 model swap is needed. If clustering is pursued, BERTopic with a separate multilingual embedding model, soft assignment via `approximate_distribution()`, and ensemble query routing (centroid similarity + BM25 hit frequency) is the production-ready approach.

The core problem — all AI/ML posts scoring 0.78–0.83 cosine with any query — stems from two compounding effects. **Embedding anisotropy** causes all transformer embeddings to occupy a narrow cone in vector space (Ethayarajh 2019 showed average BERT pairwise cosine similarity of 0.99). **Domain homogeneity** amplifies this: AI/ML posts share vocabulary, concepts, and phrasing, compressing the already-narrow cone further. The Length-Induced Embedding Collapse paper (Zhou et al., ACL 2025) confirms this is a fundamental architectural property of self-attention acting as a low-pass filter, not a bug in Qwen3-Embedding-0.6B specifically.

---

## Q1: BERTopic beats raw TF-IDF for short multilingual clustering

**BERTopic with a dedicated multilingual embedding model is the clear winner** over raw TF-IDF + HDBSCAN. The critical insight: use a *different* embedding model for clustering than your retrieval model, since Qwen3-Embedding-0.6B exhibits the same collapse that makes retrieval fail.

TF-IDF clustering fails on short multilingual texts for two reasons. Sparse TF-IDF vectors from 300–1500 character posts produce extremely noisy representations — a post with 40–120 tokens yields maybe 20–60 unique terms. Cross-lingual bridging is impossible: Russian "трансформер" and English "transformer" become unrelated dimensions. BERTopic overcomes both by clustering in a dense semantic space where multilingual models map equivalent concepts together, then using c-TF-IDF (class-based TF-IDF) only for topic *representation*, not for clustering itself.

**Recommended clustering pipeline:**

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired

# Use a SEPARATE embedding model — not your retrieval model
cluster_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
embeddings = cluster_model.encode(docs, show_progress_bar=True, batch_size=64)

umap_model = UMAP(
    n_neighbors=15, n_components=5,
    min_dist=0.0, metric='cosine', random_state=42
)
hdbscan_model = HDBSCAN(
    min_cluster_size=150, min_samples=10,
    metric='euclidean', cluster_selection_method='eom',
    prediction_data=True
)
vectorizer_model = CountVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2))
ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

topic_model = BERTopic(
    embedding_model=cluster_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=KeyBERTInspired(),
    calculate_probabilities=True, verbose=True
)
topics, probs = topic_model.fit_transform(docs, embeddings)
```

The embedding model choice matters more than the clustering algorithm. `paraphrase-multilingual-mpnet-base-v2` (768-dim, 278M params) provides strong cross-lingual semantics without the cosine compression problem because UMAP projects into 5 dimensions first, re-scaling the compressed distance space. An ACM AICE 2024 study found that iterating `min_cluster_size` to maximize silhouette score improved BERTopic coherence by **12.24%** over defaults. Medvecki et al. (2024) demonstrated BERTopic on short Serbian text (morphologically rich, like Russian) with 23 informative topics and zero outliers using `min_topic_size=15`.

What about clustering on BM25 sparse vectors? This is viable — UMAP handles sparse input with the Hellinger metric, and BM25 vectors already exist in your Qdrant collection. However, BM25 vectors lack cross-lingual semantics, so Russian and English posts on the same topic would cluster separately. **Use BM25 vectors for routing (Method F below), not for clustering.**

---

## Q2: Target 25–40 clusters for 13K docs, scale with hierarchy

The ideal cluster count balances discrimination power against routing accuracy. **Too few clusters (<15) means each still contains 800+ diverse documents — minimal benefit. Too many (>100) means query routing becomes fragile.** For 13K documents, the sweet spot is **25–40 clusters** (325–520 docs each), achieved by setting `min_cluster_size=100–200` in HDBSCAN.

The principled approach is DBCV (Density-Based Cluster Validation) via HDBSCAN's `relative_validity_` score, not silhouette — silhouette assumes spherical clusters which HDBSCAN explicitly does not produce:

```python
from sklearn.metrics import silhouette_score
import hdbscan

results = []
for mcs in [50, 75, 100, 125, 150, 200, 250, 300]:
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=mcs, min_samples=10,
        cluster_selection_method='eom', prediction_data=True
    )
    labels = hdb.fit_predict(reduced_embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = (labels == -1).sum() / len(labels)
    dbcv = hdb.relative_validity_
    results.append({'mcs': mcs, 'n': n_clusters,
                    'noise': noise_ratio, 'dbcv': dbcv})
# Select: 25-40 clusters, noise <15%, highest DBCV
```

**For scaling to 50–100K documents**, use BERTopic's hierarchical topic modeling to create a two-level hierarchy: 8–12 broad categories for fast routing, then 40–80 fine-grained topics for precision. Increase `min_cluster_size` proportionally (300 at 50K, 500 at 100K). The Adaptive Cluster-Graph paper achieved perfect Recall@1 and Recall@5 on 50K CORD-19 documents with centroid-based hierarchical graphs, validating this approach at scale.

A critical anti-pattern: do **not** use BERTopic's `nr_topics="auto"` without inspection — it can over-merge semantically distinct topics. Instead, fit with natural HDBSCAN clusters, inspect the dendrogram via `topic_model.visualize_hierarchy()`, and manually choose a cut point.

---

## Q3: Ensemble routing with centroid similarity and BM25 hit frequency

**The recommended routing strategy combines two fast methods: embed the query and compare against cluster centroids (Method B), plus count which clusters appear most in the top-50 BM25 hits over the full corpus (Method F). Union the results for 3–5 clusters, then run filtered hybrid search in Qdrant.** Total latency: ~100–150ms on CPU.

Here is a comparative assessment of all six routing options:

| Method | Latency (CPU) | Accuracy | Best for |
|--------|-------------|----------|----------|
| **(A)** Search all clusters w/ whitening | 250–750ms | Highest (no routing error) | Small cluster count only |
| **(B)** Query embedding → centroids | <100ms | Good (semantic match) | Paraphrase-style queries |
| **(C)** TF-IDF query → cluster c-TF-IDF | <5ms | Moderate (keyword only) | Exact-term queries |
| **(D)** LLM zero-shot classification | 2–10s | Variable | Not recommended for CPU |
| **(E)** BM25 over cluster summaries | <5ms | Limited by summary quality | Simple topical routing |
| **(F)** BM25 full corpus → cluster frequency | ~60ms | High (leverages proven BM25) | Keyword-rich queries |

Method D (LLM classification) is explicitly **not recommended** — a 2025 RAG Router analysis found it produces "slow inference time, high operational costs, and poor accuracy on domain-specific topics." A fine-tuned BERT classifier achieves 94% accuracy at millisecond latency instead, but requires labeled training data you likely don't have.

**Production-ready ensemble routing implementation:**

```python
import numpy as np
from collections import Counter
from qdrant_client import models

def route_query(query, query_embedding, bm25_retriever, 
                centroid_matrix, cluster_labels, doc_to_cluster, 
                top_k_clusters=3):
    # Method B: centroid similarity
    sims = query_embedding @ centroid_matrix.T  # (n_clusters,)
    centroid_clusters = np.argsort(sims)[-top_k_clusters:][::-1].tolist()
    
    # Method F: BM25 hit frequency
    bm25_hits = bm25_retriever.retrieve(query, k=50)
    cluster_counts = Counter(doc_to_cluster[h.id] for h in bm25_hits)
    bm25_clusters = [c for c, _ in cluster_counts.most_common(top_k_clusters)]
    
    # Union (typically 3-5 unique clusters)
    return list(set(centroid_clusters + bm25_clusters))

# Qdrant filtered hybrid search
selected = route_query(query, q_emb, bm25, centroids, labels, d2c)
results = client.query_points(
    collection_name="posts",
    prefetch=[
        models.Prefetch(query=dense_vec, using="dense", limit=20),
        models.Prefetch(query=sparse_vec, using="bm25", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=models.Filter(must=[
        models.FieldCondition(
            key="cluster_ids",
            match=models.MatchAny(any=selected)
        )
    ]),
    limit=10
)
```

This pattern is validated by production systems. FAISS IVF uses centroid-based routing as its core algorithm. Coinbase's CAR system uses cluster analysis to dynamically determine retrieval scope, cutting LLM token usage by **60%** and latency by **22%**. RAPTOR (ICLR 2024) builds hierarchical cluster summaries achieving +20% absolute accuracy on the QuALITY benchmark.

---

## Q4: Global whitening first — per-cluster whitening is statistically dangerous

**Per-cluster PCA whitening with 200–600 documents in 1024 dimensions is mathematically unsound.** The covariance matrix is rank-deficient: with N=400 and d=1024, the sample covariance has rank ≤ 399, leaving 625 eigenvalues at exactly zero. Dividing by zero eigenvalues during whitening amplifies pure noise. **Global whitening is the correct approach** — with N=13,000 and d=1024, the ratio N/d ≈ 12.7 is comfortable for stable covariance estimation.

The whitening transform, from Su et al. (2021): given embeddings {x₁...xₙ}, compute mean μ, covariance Σ = UΛUᵀ, then transform x_white = Λ^(-1/2) Uᵀ (x − μ). Optionally keep only the top-k principal components for simultaneous quality improvement and dimensionality reduction.

**Benchmarked improvements from the literature:**

- Su et al. (2021) BERT-whitening: **+8–12 Spearman points** on STS tasks; 768→256 dim reduction often *improved* quality
- WhitenRec (2024): **+7–16% Recall@20** across recommendation datasets
- WhiteningBERT (Huang et al., EMNLP 2021): consistent +2–5 point boosts across all tested PLMs

```python
import numpy as np
from sklearn.decomposition import PCA

class EmbeddingWhitener:
    def __init__(self, n_components=512, epsilon=1e-4):
        self.n_components = n_components
        self.epsilon = epsilon
    
    def fit(self, embeddings):
        pca = PCA(n_components=self.n_components, whiten=False)
        pca.fit(embeddings)
        self.mean_ = pca.mean_
        self.components_ = pca.components_
        self.explained_variance_ = pca.explained_variance_
        return self
    
    def transform(self, x):
        centered = x - self.mean_
        projected = centered @ self.components_.T
        scale = 1.0 / np.sqrt(self.explained_variance_ + self.epsilon)
        return projected * scale

# Fit on all 13K embeddings, transform, re-index
whitener = EmbeddingWhitener(n_components=512)
whitener.fit(all_embeddings)  # (13000, 1024)
whitened = whitener.transform(all_embeddings)  # (13000, 512)
whitened /= np.linalg.norm(whitened, axis=1, keepdims=True)
# At query time: whitened_query = whitener.transform(query_emb)
```

**If global whitening is insufficient**, the safe per-cluster operation is mean-centering only (subtract cluster centroid) — this requires no covariance estimation and removes each cluster's dominant bias direction. Rajaee & Pilehvar (ACL 2021) showed this cluster-based isotropy enhancement outperforms global approaches because different topic clusters have different local dominant directions. **Never** attempt full PCA whitening per cluster without heavy regularization (Ledoit-Wolf shrinkage) or aggressive dimensionality reduction (keep ≤ N/3 components).

The Soft-ZCA approach (arXiv:2411.17538) offers a middle ground: W = U(Λ + εI)^(-1/2)Uᵀ where ε controls whitening strength. Full whitening (ε=0) can destroy useful signal in fine-tuned models. Start with ε=0.1 and tune on your evaluation set.

For Qdrant integration: pre-whiten all document vectors and create a new collection with reduced dimensions (512). Whitening parameters are tiny (~2MB) — load at startup, apply to queries in <0.1ms. This halves your vector storage and likely **improves** retrieval quality simultaneously.

---

## Q5: Soft assignment with BERTopic's approximate_distribution

**Store each document's top-3 cluster IDs as an integer array payload in Qdrant.** Hard single-cluster assignment loses recall for cross-cutting posts ("using transformers for financial NLP" belongs in both "transformers" and "finance" clusters). BERTopic's `approximate_distribution()` method is purpose-built for this — it uses sliding-window c-TF-IDF matching to compute a topic probability distribution per document, naturally handling multi-topic content.

```python
# After fitting BERTopic
topic_distr, _ = topic_model.approximate_distribution(
    docs, window=4, stride=1, min_similarity=0.01
)

THRESHOLD, TOP_K = 0.05, 3
for i in range(len(docs)):
    top_ids = np.argsort(topic_distr[i])[::-1]
    assigned = [int(t) for t in top_ids[:TOP_K] if topic_distr[i][t] > THRESHOLD]
    if not assigned:
        assigned = [int(top_ids[0])]
    client.set_payload(
        collection_name="posts",
        payload={"cluster_ids": assigned, "primary_cluster": assigned[0]},
        points=[point_ids[i]]
    )

# CRITICAL: create payload index
client.create_payload_index(
    "posts", "cluster_ids", models.PayloadSchemaType.INTEGER
)
```

Qdrant natively supports array-valued payloads — a filter condition succeeds if *any* element in the array matches. With a keyword/integer index, filtering by `MatchAny` across 3–5 clusters is efficient: Qdrant's query planner detects the filter cardinality (~10–25% of documents) and uses filterable HNSW, which traverses the graph while skipping non-matching nodes.

**Impact on index size:** Soft assignment with top-3 clusters does *not* duplicate vectors — only the payload grows slightly. Each document's payload increases by ~50–100 bytes (array of 3 integers + scores). At 13K docs, this adds ~1MB total. The alternative of duplicating points into multiple clusters would increase vector storage by 1.5–2×, which is unnecessary.

HDBSCAN's own `all_points_membership_vectors()` is an alternative but scales poorly on CPU: 50K docs takes 50 seconds, and 200K docs can take 1.5 hours. `approximate_distribution()` is faster and produces more interpretable multi-topic assignments for short texts.

---

## Q6: Daily nearest-centroid assignment, weekly model merging

**Use a two-tier incremental strategy:** daily new posts get assigned to the nearest existing cluster centroid (fast, zero retraining), and weekly BERTopic model merging incorporates accumulated posts into a properly re-fit model.

```python
from collections import deque

class IncrementalClusterManager:
    def __init__(self, topic_model, centroids, threshold=0.3):
        self.model = topic_model
        self.centroids = centroids  # (n_clusters, embed_dim)
        self.threshold = threshold
        self.orphan_buffer = []
        self.drift_monitor = deque(maxlen=200)
    
    def assign_new_doc(self, doc, embedding):
        sims = embedding @ self.centroids.T
        best_cluster = int(np.argmax(sims))
        confidence = float(sims[best_cluster])
        self.drift_monitor.append(1 - confidence)
        
        if confidence < self.threshold:
            self.orphan_buffer.append((doc, embedding))
        return best_cluster, confidence
    
    def should_recluster(self):
        if len(self.drift_monitor) < 50:
            return False
        avg_drift = np.mean(self.drift_monitor)
        return (avg_drift > 0.6 or len(self.orphan_buffer) > 500)
```

BERTopic's `merge_models()` (v0.16+) is the recommended approach for periodic re-clustering. The library author explicitly advises this over `partial_fit()`, which uses weaker algorithms (MiniBatchKMeans instead of HDBSCAN) and has stability issues. Merging trains a fresh BERTopic model on the new batch, then aligns and combines topic representations:

```python
# Weekly merge
new_model = BERTopic(nr_topics="auto")
new_model.fit_transform(weekly_new_docs, weekly_new_embeddings)
merged = BERTopic.merge_models([base_model, new_model])
# Update centroids, re-assign all docs if topics changed
```

**Re-clustering triggers:** (1) rolling average centroid distance exceeds threshold, (2) orphan buffer exceeds 500 documents, (3) weekly time-based schedule as safety net. At 100 new posts/day and weekly merges, this means ~700 new posts per merge cycle — manageable on CPU in under 5 minutes.

---

## Q7: The honest complexity-vs-impact comparison

This is where the research becomes most valuable. **The evidence strongly favors a simpler-first approach.** Here is the complete comparison, with expected recall deltas calibrated against published benchmarks:

| Intervention | Effort | Recall@5 Δ | Ops burden | Cumulative Recall@5 |
|---|---|---|---|---|
| **Weighted RRF tuning** (increase BM25 weight) | 1–2 hrs | +3–10% | None | 0.62–0.65 |
| **Global PCA whitening** (1024→512 dim) | 2–4 hrs | +5–15% | Near-zero | 0.67–0.73 |
| **bge-reranker-v2-m3** (rerank top-20→5) | 4–8 hrs | +15–30% | ~200ms latency | 0.75–0.82 |
| **BGE-M3 model swap** (dense+sparse+ColBERT) | 8–12 hrs | +25–40% | Medium | 0.80–0.88 |
| **Topic clustering + routing** | 20–40 hrs | +10–20% | High (ongoing) | 0.69–0.77 |

The key insight: **a cross-encoder reranker can compensate for embedding collapse by jointly attending to query-document pairs at scoring time**, sidestepping the cosine similarity floor entirely. The bge-reranker-v2-m3 (568M params, XLM-RoBERTa backbone) natively handles Russian + English and achieves **59% absolute MRR@5 improvement** in financial RAG benchmarks. At ~8ms per query-document pair on CPU, reranking top-20 candidates costs ~160ms — well within the 30s budget.

BGE-M3 deserves special attention as a potential single-model replacement. It produces dense, learned sparse (superior to traditional BM25), and ColBERT vectors from a single forward pass. On MIRACL (18-language retrieval), its combined mode achieves **nDCG@10 = 70.0** versus 65.4 for mE5-large. On MKQA (26 languages), **Recall@100 = 75.5%** versus 70.9% for the strongest baseline. This is a net complexity *reduction* — one model replacing two — while addressing the collapse problem through multi-signal retrieval.

**The recommended implementation sequence:**

- **Phase 1 (Day 1–2):** Tune RRF weights (sweep BM25 weight 0.5–3.0) + global PCA whitening with n_components=512. Re-index whitened vectors. Expected: Recall@5 → 0.67–0.73.
- **Phase 2 (Day 3–5):** Add bge-reranker-v2-m3 reranking top-20 → top-5. Expected: Recall@5 → 0.75–0.82.
- **Phase 3 (Week 2):** If still insufficient, swap to BGE-M3 (re-embed all 13K posts). Expected: Recall@5 → 0.80–0.88.
- **Phase 4 (Week 3+, optional):** Add BERTopic clustering as a metadata layer for topical filtering, browsing, and analytics — but at this point, it's an enhancement, not a fix.

---

## When clustering IS worth the complexity

Despite the above, there are three scenarios where topic clustering becomes the right architectural choice:

**Scale beyond 50K documents.** At 100K posts, even a good reranker struggles if the first-stage retrieval candidate pool is noisy. Cluster-based filtering reduces the HNSW search space by 90–95%, improving both latency and recall simultaneously. The Adaptive Cluster-Graph paper demonstrated perfect Recall@1 on 50K CORD-19 documents with centroid-based hierarchical routing.

**Heterogeneous domain expansion.** If the Telegram aggregator grows beyond AI/ML into crypto, biotech, finance, etc., topic clustering provides natural domain separation. The cosine similarity floor is a *within-domain* problem — cross-domain filtering via clusters becomes essential.

**Per-cluster whitening at scale.** With 50K+ docs and 500+ docs per cluster, per-cluster covariance estimation becomes statistically sound. At that point, the full pipeline — cluster-aware whitening removing local anisotropy directions — can outperform global whitening by 3–5% based on Rajaee & Pilehvar (ACL 2021).

---

## Complete recommended architecture

```
                        User Query
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
            [Embed Query]    [BM25 Full Corpus]
            (Qwen3 or BGE-M3)    (13K docs, ~10ms)
                   │                 │
                   ▼                 ▼
           [Global Whitening]  [Top-50 BM25 Hits]
           (PCA 1024→512)      [Count cluster freq]
                   │                 │
            ┌──────┘                 │
            ▼                        ▼
     [Centroid routing]      [Cluster frequency]
     (top-3 clusters)       (top-3 clusters)
            │                        │
            └──────┬─────────────────┘
                   ▼
          [Union: 3–5 clusters]
                   │
                   ▼
        [Qdrant Filtered Hybrid Search]
        filter: cluster_ids ∈ selected
        prefetch: dense(20) + BM25(20)
        fusion: weighted RRF
                   │
                   ▼
          [Top-20 Candidates]
                   │
                   ▼
        [bge-reranker-v2-m3]
        (cross-encoder, ~160ms CPU)
                   │
                   ▼
           [Top-5 Final Results]
```

**Latency budget:** Query embedding ~50ms + whitening <0.1ms + centroid routing <1ms + BM25 ~10ms + Qdrant filtered search ~20ms + reranking ~160ms = **~240ms total**. Well within the 30s constraint.

---

## Risk assessment and failure modes

**Clustering risks.** HDBSCAN can assign 20–74% of documents as noise (outlier topic -1) if `min_cluster_size` is too high. BERTopic's outlier reduction strategies (distributions, embeddings, c-TF-IDF) mitigate this but must be applied explicitly. Query routing errors cascade: if the correct cluster is not selected, recall drops to near-zero for that query. The ensemble routing approach (B+F) provides redundancy — empirically, one of the two methods usually identifies the correct cluster even when the other fails.

**Whitening risks.** Over-whitening amplifies noise in low-variance dimensions, destroying alignment (semantically similar items pushed apart). Mitigation: use dimensionality reduction (keep top-k components) which automatically drops noisy low-variance directions. Start with n_components=512, then try 256. Evaluate on your actual query set, not just cosine similarity spread.

**Reranker risks.** Cross-encoders add ~160ms latency per query (top-20 reranking on CPU). If reranking top-50 (for better recall), latency rises to ~400ms. Model size is 568M params — needs ~2GB RAM for FP16 inference. For a Telegram bot application, this is acceptable; for a high-QPS API, consider ONNX quantization or batching.

**BGE-M3 migration risk.** Re-embedding 13K documents takes ~2–4 hours on CPU. The model is 568M params (same as bge-reranker). ColBERT vectors increase storage by ~6× (one 1024-dim vector per token instead of one per document). For 13K posts averaging 50 tokens: 13K × 50 × 1024 × 4 bytes ≈ **2.7GB** for ColBERT vectors alone. Manageable, but plan for it.

**Scale risks (50–100K docs).** BERTopic's `merge_models()` is stable for weekly batches of ~5K docs but untested at 100K+ total. HDBSCAN soft clustering scales poorly on CPU (200K docs → 1.5 hours). At 100K docs, consider RAPIDS cuML for GPU-accelerated HDBSCAN (400K docs in <2 seconds) or switch to a simpler K-Means backbone.

## Conclusion

The evidence converges on a clear finding: **the embedding collapse problem has simpler, higher-ROI solutions than topic clustering**. Global PCA whitening directly addresses anisotropy at the mathematical level, cross-encoder reranking bypasses the cosine floor entirely by scoring query-document pairs jointly, and BGE-M3 provides multi-signal retrieval (dense + learned sparse + ColBERT) that is inherently more robust to single-vector collapse. Topic clustering remains valuable as an organizational metadata layer, a routing mechanism at scale, and a foundation for per-cluster whitening once cluster sizes exceed ~500 documents — but it should be Phase 4, not Phase 1. The expected trajectory from Recall@5 = 0.59 to 0.80+ is achievable in under two weeks with the phased approach, compared to 3–4 weeks for a full clustering pipeline delivering lower gains.