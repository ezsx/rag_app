# RAG coverage metrics that actually work in production

**The most reliable coverage signal combines raw cosine similarity statistics with lexical term overlap — not RRF scores, not document counts, and not LLM-as-judge calls.** Research from 2023–2025 reveals a critical insight: relevance and sufficiency are fundamentally different properties. Google's ICLR 2025 paper demonstrated that retrieved context can be highly relevant yet still insufficient to answer a query. This distinction matters because the current heuristic (document count ratio) conflates the two. The practical solution is a weighted composite of locally computable signals — max similarity, mean top-k quality, score distribution shape, and query term coverage — that approximates true sufficiency without requiring external LLM calls or training data.

The stakes are asymmetric: **false negatives (missing a needed search round) are far more harmful than false positives (one extra search)**. Google found models hallucinate 66.1% of the time with insufficient context versus 10.2% with no context at all. An unnecessary extra retrieval costs 200–500ms of latency; a missed retrieval produces confidently wrong answers that erode user trust.

---

## What "coverage" actually means — and why the naive metric fails

The term "coverage" in RAG systems has three distinct definitions that production systems routinely conflate. **Context recall** measures whether all information needed to answer a query was retrieved, typically computed as the fraction of ground-truth claims attributable to retrieved documents. **Context relevance** (or precision) measures how much of the retrieved context is actually useful, penalizing noise and off-topic content. **Context sufficiency**, the most rigorous definition introduced by Google Research, asks whether the context contains all necessary information to provide a definitive answer.

The current `citation_coverage` heuristic — likely a ratio of retrieved documents to some expected count — captures none of these properties. A system can retrieve 10 documents (high count ratio) that are all about a tangentially related topic (low relevance, zero sufficiency). Conversely, a single highly relevant passage might fully answer the query (coverage should be 1.0, but the count ratio would be 0.2 for target_k=5).

The RAGAS framework, now the industry-standard evaluation suite with integrations in LangChain, LlamaIndex, and Haystack, defines context recall as `claims_supported_by_context / total_claims_in_reference`. Context precision uses rank-weighted relevance: `Σ(precision@k × relevance_k) / total_relevant_items`. Both require either ground-truth references or LLM calls, making them unsuitable for real-time refinement decisions in a ReAct loop. **The practical challenge is approximating these metrics using only locally computable signals** — similarity scores, score distributions, and lexical overlap.

DeepEval sets a default threshold of **0.7** for contextual recall. TruLens uses **0.5** as its runtime guardrail threshold for context relevance filtering. Production RAG systems at scale (per CustomGPT benchmarks) target faithfulness above **0.85**, answer relevance above **0.8**, and context precision above **0.7**. These numbers provide calibration anchors for any coverage function.

---

## Seven approaches to adaptive retrieval, ranked by practicality

The 2023–2025 literature offers a spectrum of approaches for deciding when retrieval is sufficient. Each makes fundamentally different tradeoffs between accuracy, latency, and implementation complexity.

**Self-RAG** (Asai et al., ICLR 2024) is the most elegant solution for systems that can afford fine-tuning. It trains the LLM to emit special reflection tokens — `[Retrieve]` (yes/no), `[IsREL]` (relevant/irrelevant), `[IsSUP]` (fully/partially/not supported), and `[IsUSE]` (1–5 utility score). At inference, only the generator model runs; no separate critic is needed. **Self-RAG with Llama 2-7B outperforms ChatGPT on multiple benchmarks**, making it viable for local models. The adaptive retrieval threshold defaults to **0.2** probability for the `[Retrieve]=Yes` token, tunable without retraining.

**CRAG** (Corrective RAG, Yan et al., 2024) takes a post-retrieval approach: a fine-tuned **T5-Large evaluator** classifies retrieved documents into three confidence tiers. High confidence proceeds normally. Low confidence discards retrieved documents entirely and falls back to web search. Ambiguous confidence combines both. This plug-and-play design attaches to any existing RAG pipeline and improved accuracy by **+19% on PopQA**.

**Adaptive-RAG** (Jeong et al., NAACL 2024) uses a trained T5-Large classifier to route queries by complexity: simple queries skip retrieval, moderate queries use single-pass RAG, complex queries trigger multi-step iterative retrieval. The classifier requires a single forward pass, making it fast and practical. It is already integrated into LangChain and LlamaIndex.

**FLARE** (Jiang et al., EMNLP 2023) generates text iteratively, checking token-level probabilities. When any token probability drops below threshold θ, retrieval triggers. This requires logprobs access and multiple LLM forward passes per response, making it expensive but training-free. **DRAGIN** (Su et al., ACL 2024) improves on FLARE by combining token uncertainty with attention-based importance scores and semantic significance, producing better retrieval queries.

**SKR** (Self-Knowledge guided Retrieval, Wang et al., EMNLP 2023) is the most lightweight: a k-nearest-neighbor lookup against previously labeled query embeddings predicts whether the LLM needs retrieval. No model modification, no fine-tuning, just embedding similarity. Their key finding: **retrieved knowledge sometimes degrades performance**, making selective retrieval essential.

For the specific use case of a ReAct agent with Qdrant hybrid search, the most applicable approaches are a local composite scoring function (no external dependencies), CRAG-style post-retrieval evaluation (if T5-Large can run locally), or a simplified Self-RAG reflection if the local model supports it.

---

## Why raw cosine similarity beats RRF scores for coverage estimation

Qdrant's hybrid search produces RRF (Reciprocal Rank Fusion) scores computed as `score(d) = Σ 1/(k + rank_i)` across dense and sparse retrievers, with k=60 as the standard constant. These scores have critical limitations for coverage computation. The maximum possible RRF score for two retrievers is approximately **0.0328** (when a document ranks first in both). The scores are relative within a single query and not comparable across queries. Qdrant's own documentation warns that raw cosine and BM25 scores are **not linearly separable** for relevant versus non-relevant documents.

**Raw cosine similarity is the better signal for coverage** because it has an interpretable [0, 1] range, stable cross-query semantics, and well-studied threshold behavior. Practical thresholds for modern embedding models (OpenAI, BGE, E5):

- **0.85+**: Near-paraphrase match, very high relevance
- **0.75–0.85**: Strong topical match
- **0.60–0.75**: Moderate relevance, related content
- **0.45–0.60**: Tangentially related
- **Below 0.45**: Likely irrelevant

The recommended architecture: use RRF for ranking (it excels at combining incompatible score scales), but compute cosine similarity separately for coverage estimation. In Qdrant, request `with_vectors=True` on hybrid search results, then compute `dot(query_vector, doc_vector)` for each result since both vectors are L2-normalized.

Score distributions from information retrieval theory (Manmatha et al., 2001) follow predictable patterns: **relevant document scores approximate a Normal distribution** centered at higher values, while non-relevant scores follow an **Exponential distribution** concentrated at lower values. This means the shape of the score distribution — not just the mean — carries valuable coverage information.

---

## A production-ready coverage function in five signals

The following composite approach combines five locally computable signals into a single coverage float. No LLM calls, no training data, no external dependencies.

**Signal 1: Top-match quality** (`max_similarity`). The cosine similarity of the best-matching document. If the best document scores below 0.45, coverage is almost certainly insufficient regardless of other signals. Weight: **0.25**.

**Signal 2: Mean top-k quality** (`mean_top_k_similarity`). Average cosine similarity of the top-5 results. This captures whether the retrieval found a breadth of relevant content, not just one lucky match. Weight: **0.20**.

**Signal 3: Query term coverage** (`term_coverage`). Fraction of meaningful query terms (excluding stop words) that appear in the combined retrieved text. This lexical signal catches cases where embeddings find semantically similar but factually different content. Weight: **0.20**.

**Signal 4: Document count adequacy** (`doc_count_adequacy`). Number of documents scoring above the relevance threshold (0.55) divided by target_k (typically 5), capped at 1.0. Weight: **0.15**.

**Signal 5: Score concentration** (`score_gap`). The ratio `1 - (top1_score - topk_score) / top1_score`. A small gap means multiple strong results (broad coverage); a large gap means one dominant result with sparse supporting evidence. Weight: **0.15**.

An optional sixth signal, `above_threshold_ratio` (fraction of all retrieved documents above the relevance threshold), adds **0.05** weight for distinguishing clean retrievals from noisy ones.

```python
def calculate_coverage(query: str, retrieved_docs: list[dict],
                       relevance_threshold=0.55, target_k=5) -> float:
    if not retrieved_docs:
        return 0.0
    scores = sorted([d['cosine_sim'] for d in retrieved_docs], reverse=True)
    top_k = scores[:target_k]

    max_sim = scores[0]
    mean_top_k = sum(top_k) / len(top_k)
    relevant_count = sum(1 for s in scores if s >= relevance_threshold)
    count_adequacy = min(1.0, relevant_count / target_k)
    gap = 1.0 - (scores[0] - scores[min(target_k-1, len(scores)-1)]) / scores[0] if scores[0] > 0 else 0
    term_cov = _query_term_overlap(query, retrieved_docs)

    return min(1.0, 0.25*max_sim + 0.20*mean_top_k + 0.20*term_cov
                   + 0.15*count_adequacy + 0.15*gap + 0.05*(relevant_count/len(scores)))
```

The `_query_term_overlap` function extracts meaningful terms from the query (3+ character tokens minus stop words), checks which appear in the concatenated document text, and returns the covered fraction. This takes microseconds and provides a surprisingly strong signal — it catches the failure mode where semantic search returns topically similar but factually non-responsive documents.

**The refinement threshold should be set at 0.65–0.70, not 0.80.** Setting it at 0.8 with this composite metric will trigger unnecessary re-retrieval on most queries because the weighted combination naturally compresses scores. Calibrate on 30–50 labeled examples: for each, mark whether the retrieved context was sufficient, compute the coverage score, and find the threshold that maximizes F1 or minimizes the cost-weighted error rate (weighting false negatives ~3× higher than false positives).

---

## Calibration strategy and the asymmetric error tradeoff

The precision-recall tradeoff for refinement triggers is fundamentally asymmetric. Google's ICLR 2025 "Sufficient Context" paper demonstrated that **models hallucinate at 66.1% with insufficient context** — worse than the 10.2% hallucination rate when given no context at all. Irrelevant context actively harms output quality by increasing model confidence without providing correct information. This means a missed retrieval round (false negative) produces confidently wrong answers, while an unnecessary retrieval (false positive) adds only 200–500ms latency.

**Practical recommendation: bias the threshold toward more retrieval.** Set the threshold ~0.5 standard deviations below the empirical mean coverage score on your query distribution. In concrete terms, if your labeled queries produce coverage scores averaging 0.72 with standard deviation 0.12, set the threshold at approximately **0.66** rather than 0.72.

For calibration without extensive labeled data, use these empirical anchors from production systems:

- **Confidence < 0.50**: Always trigger additional retrieval (DBI Services adaptive RAG)
- **Confidence 0.50–0.70**: Trigger retrieval if latency budget permits
- **Confidence > 0.70**: Proceed to answer generation
- **Top cosine similarity < 0.30**: Abort and report "insufficient information" rather than hallucinate

Google's selective generation framework combines the sufficient-context signal with model self-confidence using logistic regression, achieving up to **10% accuracy improvement** on answered questions by abstaining on low-confidence cases. A simpler version: if coverage < 0.5 after two retrieval rounds, return a hedged answer with an explicit "limited information" disclaimer.

The FAIR-RAG paper (2025) found that iterative refinement shows diminishing returns: **F1 improves from 0.398 (1 iteration) to 0.447 (3 iterations)** on HotpotQA, plateauing thereafter. Cap refinement at **2–3 rounds maximum** with a hard stop rule regardless of coverage score.

---

## Conclusion

The path from a document-count heuristic to a meaningful coverage metric requires three changes. First, **pass the query into `compose_context`** and compute cosine similarity between the query embedding and each document's dense vector — this is the single highest-value improvement. Second, **replace the binary threshold with the five-signal composite** described above, which captures match quality, breadth, and lexical coverage without any external dependencies. Third, **lower the proceed threshold to ~0.65** while adding a hard cap of 2–3 retrieval rounds.

The most surprising finding from this research is that simple signal combinations — weighted averages of cosine statistics plus term overlap — approach the performance of LLM-as-judge methods for real-time refinement decisions. Google's autorater achieves 93% accuracy, but it requires a Gemini 1.5 Pro call per decision. The composite heuristic computes in microseconds and captures the most informative signals: whether the best match is strong, whether multiple matches exist, and whether the query's key concepts appear in the retrieved text. For offline evaluation, adopt RAGAS metrics (faithfulness > 0.85, context precision > 0.7) as quality gates. For runtime refinement decisions, the local composite function is the right tradeoff between accuracy and latency.