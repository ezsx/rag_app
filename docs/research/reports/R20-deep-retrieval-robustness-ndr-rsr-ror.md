# Adapting RAG robustness evaluation to production constraints

**The Cao et al. (2025) framework for measuring retrieval robustness — NDR, RSR, and ROR — can be scaled from its academic protocol of 1,500 questions and ~55,000 LLM calls per model down to a production-viable 50-question, ~650-call configuration that retains statistical power for detecting medium-to-large effects.** This adaptation requires careful choices about k-values, shuffle counts, scoring functions, and statistical tests. The key insight: with paired evaluation designs and appropriate non-parametric tests, N=50 questions achieves ~76% power for Cohen's d=0.5 effects, while N=100 reaches the 80% power standard at d=0.4. For Russian-language RAG specifically, xlm-roberta-large or ai-forever/ruBert-large deliver the best BERTScore performance, and GPT-4o is the most reliable LLM judge — though multilingual judge consistency remains low (Fleiss' κ ≈ 0.3), demanding calibration against human labels.

---

## The Cao et al. framework and its three robustness metrics

The paper "Evaluating the Retrieval Robustness of Large Language Models" (Cao et al., arXiv:2505.21870, May 2025) — authored by researchers from Bloomberg and University of Michigan — introduces three complementary metrics. Note that the paper's own terminology differs slightly from some descriptions: **NDR = No-Degradation Rate** (not "Noise Disturbance Ratio"), **RSR = Retrieval Size Robustness** (not "Retrieval Scalability Ratio"), and **ROR = Retrieval Order Robustness**.

**No-Degradation Rate (NDR)** measures how often RAG performance meets or exceeds the no-retrieval baseline:

$$\text{NDR} = \frac{1}{|Q| \cdot |K| \cdot |O|} \sum_{q \in Q} \sum_{k \in K} \sum_{o \in O} \mathbb{1}[f(q, k, o) \geq f(q, 0)]$$

where f(q, k, o) is the correctness score for query q with k retrieved documents in ordering o, and f(q, 0) is the non-RAG baseline. A high NDR means retrieval rarely degrades performance. Counterintuitively, **larger models can have lower NDR** because their strong non-RAG baseline creates a higher bar for retrieval to clear.

**Retrieval Size Robustness (RSR)** checks whether performance is monotonically non-decreasing as more documents are retrieved:

$$\text{RSR} = \frac{1}{|Q| \cdot (|K|-1) \cdot |O|} \sum_{q \in Q} \sum_{k_i \in K, i>1} \sum_{o \in O} \mathbb{1}\left[\bigwedge_{j < i} f(q, k_i, o) \geq f(q, k_j, o)\right]$$

This metric only counts a success when performance at k_i meets or exceeds performance at **all** smaller k values — a strict monotonicity requirement. It excludes the non-RAG baseline (k=0) to separate its signal from NDR.

**Retrieval Order Robustness (ROR)** quantifies sensitivity to document permutation:

$$\text{ROR} = \frac{1}{|Q| \cdot |K|} \sum_{q \in Q} \sum_{k \in K} \left(1 - 2 \cdot \sigma_{o \in O}[f(q, k, o)]\right)$$

Since f is bounded in [0,1], the standard deviation σ across orderings is bounded in [0, 0.5], so the factor of 2 normalizes ROR to [0, 1]. Higher values indicate greater consistency across orderings. The paper reports an overall **composite robustness score** as the geometric mean of NDR, RSR, and ROR.

### The academic protocol in full detail

Cao et al. used **1,500 questions** (500 each from Natural Questions, HotpotQA, and ASQA), retrieved from a June 2024 Wikipedia dump (~20M chunks) using both **BM25** (Apache Solr 9) and **BGE-large-en-v1.5** (dense retriever). They tested **k ∈ {5, 10, 25, 50, 75, 100}** — six retrieval sizes — with **three orderings** per configuration: original retriever ranking, reversed ranking, and a single random shuffle. Eleven LLMs were evaluated (Llama-3.x family, Command R/R+, Mistral-Nemo/Large, GPT-4o, o3-mini, Claude-3.5-Sonnet) across three prompting strategies (Vanilla, OwnKnow, S2A). Evaluation used **Llama-3.3-70B-Instruct as LLM judge** (93% agreement with GPT-4o on a 2,000-sample validation set), with greedy decoding on 8×H100 GPUs via vLLM.

The total compute per model per strategy: ~1,500 baseline calls + 54,000 RAG calls (1,500 questions × 6 k-values × 3 orderings × 2 retrievers), plus matching judge calls. Across all 11 models and 3 strategies, the full benchmark required roughly **1.8 million generation calls**.

### Expected metric ranges and benchmarks

All 11 models achieved **>80% geometric mean robustness** across the three metrics. GPT-4o and o3-mini surpassed **90%** on all individual metrics. The paper found that overall task performance generally increases with more documents, but models constantly trade off sample-level performance — helping some questions while hurting others. Most models (except GPT-4o and o3-mini) performed better with **reversed document order** (relevant documents closer to the question), connecting to the lost-in-the-middle literature. Under oracle (perfect) robustness, models could gain up to **12% absolute performance** from perfect NDR alone.

---

## Statistical power analysis: how small can your evaluation set be?

The foundational concern for production RAG evaluation is whether small test sets yield meaningful conclusions. Card et al. (2020, "With Little Power," EMNLP) demonstrated that **underpowered experiments are pervasive in NLP**: even 2,000-sentence MT test sets achieve only ~75% power for detecting 1-BLEU-point differences. Dror et al. (2018, "The Hitchhiker's Guide to Testing Statistical Significance in NLP," ACL) found that 65% of ACL 2017 papers with experiments did not report significance testing at all.

For paired comparisons (the natural design for RAG evaluation, where two system variants answer the same questions), the minimum detectable effect at standard parameters (α=0.05, power=0.80, two-sided) scales as follows:

| Cohen's d | N=30 | N=50 | N=100 |
|-----------|------|------|-------|
| 0.3 (small-medium) | 25% power | 37% power | 56% power |
| 0.5 (medium) | 57% power | 76% power | 94% power |
| 0.8 (large) | 86% power | 97% power | >99% power |

**N=50 is the practical minimum** for production RAG evaluation — achieving 76% power for medium effects (d=0.5) with 95% CI widths of ±11% for binary metrics (at p=0.80) and ±0.17 on a 0-2 scale (σ≈0.6). **N=100 is recommended** when detecting smaller effects matters, offering 80% power at d=0.4. With N=30, only large effects (d≥0.8) can be reliably detected, and binary CI widths balloon to ±14%.

### Choosing the right statistical test

Dror et al. (2018) provide a decision protocol that maps directly to RAG metrics. For **binary correctness** (correct/incorrect per question), use **McNemar's exact test** — but note that power depends on discordant pairs only, so two systems agreeing on 90% of questions leaves just 10% contributing to the test. For **ordinal quality scores** (0-2 or 0-3 scales), the **Wilcoxon signed-rank test** is appropriate since normality cannot be assumed. For **composite metrics** like NDR, RSR, ROR, or metrics computed over the full evaluation set (nDCG, MRR, BERTScore), **paired bootstrap** or **permutation tests** are the gold standard — they make no distributional assumptions and achieve power comparable to parametric tests. Use B ≥ 10,000 bootstrap resamples and apply **BCa (bias-corrected and accelerated)** confidence intervals for small samples.

Three strategies maximize power with constrained samples. First, **always use paired designs** — evaluating both systems on the same questions eliminates between-item variance, effectively doubling your sample when system correlation ρ≈0.5. Second, **use continuous/ordinal scores rather than binary** — a 0-3 scale captures more information per question, requiring 2-4× fewer samples than binary for equivalent precision. Third, **use one-sided tests** when the direction is pre-specified ("Is the new system better?"), gaining ~15% more power.

---

## The lost-in-the-middle effect: architecture, not training

Liu et al. (2024, "Lost in the Middle," TACL, 2,576+ citations) established that LLMs exhibit a **U-shaped performance curve**: accuracy is highest when relevant information appears at the beginning or end of the context, and drops by **>20% for middle placement**. GPT-3.5-Turbo's multi-document QA performance in the middle was **worse than closed-book performance** (i.e., retrieval actively hurt). The effect worsens with more documents and is not fixed by extended context windows.

A 2025 paper, "Lost in the Middle at Birth" (arXiv:2603.10123), proved mathematically that this bias exists at **random initialization** before any training, using a Qwen2-0.5B architecture. **Causal masking** guarantees primacy bias (geometric advantage for early tokens), while **residual connections** guarantee recency bias (direct shortcut for final tokens). Middle tokens are "starved" because they lack both advantages. This means the bias is an inherent architectural property of causal transformers — it cannot be fully eliminated by training.

### MoE models and Qwen specifically

No published study directly compares MoE versus dense models on positional bias in controlled conditions. However, since MoE architecture replaces only the FFN layers — **not the attention mechanism** where positional bias originates — MoE models are expected to show the **same fundamental U-shaped bias**. The router assigns experts per-token with no observable position-dependent pattern (Mistral team analysis). Models with smaller active parameters (e.g., 3B active out of 30B total) may have less capacity per token to aggregate distant information, potentially making them slightly *more* susceptible to attention dilution, but this is theoretical.

**Qwen 2.5 7B** was explicitly tested in Cuconasu et al. (EMNLP 2025, "Do RAG Systems Suffer From Positional Bias?") and showed the standard U-shaped pattern. Qwen2-0.5B served as the primary testbed for the "Lost in the Middle at Birth" proof. The Chroma "Context Rot" report (July 2025) tested Qwen3-32B among 18 models and found it, like all others, showed degradation with increasing input length beyond simple retrieval tasks.

### Document reordering: less effective than expected in practice

Five reordering strategies have been studied, but **the most important 2025 finding challenges the entire reordering paradigm**. Cuconasu et al. (EMNLP 2025) demonstrated through extensive experiments on PopQA, NQ, and TriviaQA with state-of-the-art retrieval pipelines that **random ordering yields statistically equivalent accuracy** to sophisticated reordering strategies in realistic RAG settings (Wilcoxon test, p=0.05). The reason: modern retrievers return both relevant passages and highly distracting passages (>60% of queries have at least one hard distractor in top-10), so strategic reordering that places relevant documents in "good" positions simultaneously places distractors there too, neutralizing the benefit.

That said, specific strategies show different characteristics:

- **Sandwich/sides** (best at beginning and end): Most benefit with large retrieval sets (>10 docs). Hsieh et al. (2024, "Found in the Middle," ACL) showed attention calibration on top of reordering provided **6-15 percentage point improvement** on middle-positioned information. LlamaIndex implements this natively via `LostInTheMiddleRanker`.
- **Reverse/ascending** (most relevant last): Wang et al. (2024, "Searching for Best Practices in RAG") found this **surprisingly outperformed** both descending and sandwich strategies, exploiting recency bias in decoder-only models.
- **Strict descending**: Simple baseline, works well with few documents but places low-relevance documents at the high-attention end position.

**For production RAG with modern LLMs (2024-2025)**, the evidence suggests **prioritizing retrieval quality over positioning**. Keep context lean (3-5 highly relevant documents), use two-stage retrieval (broad recall → cross-encoder reranking), and apply sandwich ordering as a low-cost default. But expect marginal rather than transformative gains from reordering alone.

---

## Scoring functions and Russian-language evaluation

### Binary versus continuous scoring

Databricks' empirical evaluation of grading scales recommends an **integer 0-3 scale** — binary (0/1) works for simple pass/fail but loses granularity, while fine-grained scales (0-10) create consistency problems for both human and LLM judges. For robustness metrics, the optimal approach is to **score continuously during evaluation, then binarize for computing NDR/RSR/ROR**. Standard binarization thresholds are:

- For 0-1 scale (RAGAS): **≥ 0.5** (natural midpoint) or **≥ 0.7** (strict)
- For 0-2 scale: **≥ 1.5** (75% threshold)
- For 0-3 scale (TruLens): **≥ 2** as "acceptable"

Report both continuous average scores and binary pass rates for maximum insight. Continuous scoring with binarization captures partial degradation while maintaining clean robustness metric computation.

### LLM-as-judge reliability, especially for Russian

Zheng et al. (2023, "Judging LLM-as-a-Judge," NeurIPS) showed GPT-4 achieves **>80% agreement** with human preferences — matching human-human agreement levels. However, Fu & Liu (2025, "How Reliable is Multilingual LLM-as-a-Judge?", arXiv:2505.12201) found **average cross-lingual Fleiss' κ of only ~0.3** across 25 languages, with European languages performing better than low-resource ones. Russian falls in the mid-tier — moderately supported but less reliable than English. **Neither training on multilingual data nor increasing model scale directly improves judgment consistency.**

For Russian-language RAG evaluation:
- Use **GPT-4o** as primary judge with **English-language prompts** specifying evaluation language as Russian
- Validate against **50+ human-labeled Russian examples** before trusting judge outputs
- Consider **ensemble judging** (2-3 models with majority voting) for critical evaluations
- Expect **~70-75% human-LLM agreement** for Russian (vs >80% for English)

### BERTScore model selection for Russian

The recommended ranking for Russian BERTScore:

1. **ai-forever/ruBert-large** — Best monolingual Russian BERT with superior tokenization. Requires manual layer tuning for BERTScore (no pre-tuned settings in the BERTScore library).
2. **xlm-roberta-large** — Best multilingual option with Pearson **r ≈ 0.83** on system-level human correlations (vs **r ≈ 0.25** for bert-base-multilingual-cased). Well-supported with pre-tuned layer settings.
3. **ai-forever/ruRoBERTa-large** — Strong alternative, generally outperforms BERT on Russian SuperGLUE.
4. **bert-base-multilingual-cased** — Significantly weaker; avoid as primary metric.

For sentence-level cosine similarity (faster alternative), **ai-forever/ru-en-RoSBERTa** (2024, fine-tuned on ~4M pairs) and **BGE-M3** provide strong Russian support. Note that all BERT-based models truncate at 512 tokens and have known issues with antonymy and numerical errors.

Traditional n-gram metrics (ROUGE, BLEU, exact match) perform **very poorly for Russian** due to its rich morphology — 6 grammatical cases × 2 numbers for nouns means identical meanings produce different surface forms, severely penalizing correct answers.

---

## Practical evaluation protocol for constrained compute

### Compute budget calculations

The total generation calls follow the formula: **N × (1 + |K| × |O|)**, doubled if using LLM-as-judge. Key configurations at ~100 LLM calls/hour:

| Configuration | N | K values | Shuffles | Gen calls | + Judge | Hours |
|---|---|---|---|---|---|---|
| Screening | 20 | [3, 10, 20] | 2 | 140 | 280 | 1.4-2.8 |
| Minimal viable | 50 | [3, 10, 20] | 3 | 500 | 1,000 | 5-10 |
| **Standard (recommended)** | **50** | **[3, 5, 10, 20]** | **3** | **650** | **1,300** | **6.5-13** |
| Full | 50 | [3, 5, 10, 20] | 5 | 1,050 | 2,100 | 10.5-21 |
| Large-N full | 100 | [3, 5, 10, 20] | 5 | 2,100 | 4,200 | 21-42 |

**k=[3, 5, 10, 20] is sufficient** for production. The Cao et al. paper uses k up to 100, but performance curves between k=10 and k=20 are typically monotonic, so k=15 adds ~25% more compute for marginal diagnostic value. Add k=15 only if initial results show anomalous behavior in that range.

**Three shuffles suffice** for screening and CI/CD; five shuffles provide tighter confidence intervals for definitive evaluation. The original paper uses only three orderings (canonical, reversed, one shuffle) — meaning even the academic benchmark uses a single random permutation per configuration.

### Two-stage sequential protocol

The most compute-efficient approach uses sequential testing (following Kharitonov et al., SIGIR 2015):

**Stage 1 (Screening):** Run N=20, K=[3,10,20], 2 orderings — 140 calls, ~1.5 hours. If |Δ| > 15%, declare significant and stop. If |Δ| < 3%, declare negligible and stop. Otherwise, proceed to Stage 2. **Stage 2 (Confirmation):** Run N=50, K=[3,5,10,20], 3 orderings — 650 calls, ~6.5 hours. Apply Bonferroni correction (α=0.025 per stage). In practice, **40-60% of evaluations resolve at Stage 1**, saving 6+ hours per evaluation cycle.

### Critical implementation optimizations

**Retrieve once at k_max and cache.** Retrieving at k=20 produces a superset of all smaller k configurations — simply take the top-3, top-5, or top-10 from the cached result. This eliminates redundant retrieval calls entirely. Generate all shuffled orderings offline from cached document sets at zero LLM cost. Use **async batched LLM calls** (10-20 concurrent requests) for near-linear speedup within rate limits.

For **combined k-sweeps with pipeline ablations**, use a two-stage fractional factorial design. A full factorial of 4 k-values × 4 binary component choices (retriever type, reranker on/off, prompt template, chunk size) requires 64 configurations × N × shuffles = 9,600 calls at N=50. A **Resolution IV 2^(4-1) fractional factorial** cuts this to 32 configurations (4,800 calls), while a two-stage screening approach (Resolution III with N=20 to identify important factors, then full factorial on 1-2 factors) reduces total compute to **~1,700-2,900 calls** — a 3-6× reduction.

### CI/CD integration tiers

- **Per-PR**: Screening config (140 calls, ~1.5 hours) with exact-match scoring
- **Weekly**: Standard config (650 calls, ~6.5 hours) with LLM judge
- **Monthly/Release**: Full config (1,050 calls, ~10.5 hours) with complete NDR+RSR+ROR + RAGChecker diagnostics

---

## Conclusion: a pragmatic path from academic rigor to production reality

The Cao et al. framework provides a principled decomposition of RAG robustness into three orthogonal dimensions — noise resistance (NDR), scalability (RSR), and order invariance (ROR) — that maps cleanly to production concerns. The most actionable finding is that **the academic protocol can be compressed by roughly 50×** (from ~55,000 to ~1,050 calls per system) while retaining statistical validity, by reducing to 50 questions, 4 k-values, and 5 orderings.

Three non-obvious insights emerge from synthesizing across all research areas. First, **continuous scoring with post-hoc binarization** (0-3 scale, threshold ≥ 2) maximizes both information density per question and clean robustness metric computation — this is strictly better than choosing binary or continuous alone. Second, **document reordering provides less benefit than the lost-in-the-middle literature suggests** in production RAG settings, because real retrievers co-locate distractors with relevant passages; retrieval quality improvements dominate positioning effects. Third, **for Russian-language evaluation, the weakest link is the LLM judge** (κ ≈ 0.3 cross-lingually), not the similarity metric — investing in a 50-100 example human calibration set for judge validation yields more reliability improvement than switching between BERTScore models.