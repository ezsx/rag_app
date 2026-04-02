# Complete taxonomy of RAG evaluation metrics and frameworks (2024–2026)

**Over 120 distinct metrics now exist for evaluating RAG systems**, spanning retrieval quality, answer correctness, faithfulness, robustness, agent behavior, calibration, and safety. This report synthesizes findings from 40+ papers and 10 major frameworks to build a production-ready evaluation stack for an agentic RAG pipeline operating under severe constraints: Russian language, no cloud API access, local models only (Qwen3.5-35B-A3B, ruBERT NLI), and a 15-tool ReAct architecture. The critical finding is that a viable three-tier evaluation stack can be assembled entirely from local components — combining deterministic metrics, ruBERT-NLI sentence-level faithfulness checking (SummaC-style), BERTScore with ai-forever/ruBert-large, and Prometheus-2 7B or Qwen as an LLM judge — achieving roughly 70–80% of GPT-4-judged evaluation quality at zero API cost.

---

## 1. Full metric taxonomy across eight axes

The taxonomy below covers every metric identified in the 2024–2026 literature. Metrics are organized by evaluation axis, with source papers, measurement method, and compute cost indicated.

### Axis 1 — Retrieval quality

| Metric | Definition | Source | Method | Cost |
|--------|-----------|--------|--------|------|
| **Recall@k** | Fraction of relevant docs in top-k: \|Rel ∩ Ret@k\| / \|Rel\| | Standard IR; BEIR benchmark | Automatic (label match) | Cheap |
| **Precision@k** | Fraction of top-k docs that are relevant: \|Rel ∩ Ret@k\| / k | Standard IR | Automatic | Cheap |
| **F1@k** | Harmonic mean of Precision@k and Recall@k | Standard IR | Automatic | Cheap |
| **MRR** | Mean of 1/rank of first relevant doc across queries | Standard IR | Automatic | Cheap |
| **nDCG@k** | Position-weighted graded relevance; DCG@k = Σ(2^rel_i−1)/log₂(i+1), normalized by ideal DCG | Järvelin & Kekäläinen 2002; BEIR | Automatic (graded) | Cheap |
| **MAP** | Mean of per-query average precision: AP = (1/\|Rel\|) Σ P@k × rel(k) | Standard IR | Automatic | Cheap |
| **Hit Rate / Hit@k** | Binary: 1 if ≥1 relevant doc in top-k | Standard IR; LlamaIndex | Automatic | Cheap |
| **Context Precision** (RAGAS) | Whether relevant chunks rank higher than irrelevant; LLM judges per-chunk relevance | Es et al., EACL 2024 (arXiv:2309.15217) | LLM-judge | Medium |
| **Context Recall** (RAGAS) | Proportion of ground-truth claims attributable to retrieved context | RAGAS | LLM-judge (reference-based) | Medium |
| **Context Relevance / Utilization** (RAGAS) | Whether retrieved contexts contribute to the generated response | RAGAS | LLM-judge | Medium |
| **Context Entity Recall** | Proportion of ground-truth entities present in retrieved contexts | RAGAS | Automatic (entity extraction) | Cheap |
| **Claim Recall** (RAGChecker) | Fraction of ground-truth claims covered by retrieved chunks via entailment | Ru et al., NeurIPS 2024 (arXiv:2408.08067) | LLM claim extraction + NLI | Expensive |
| **Context Precision** (RAGChecker) | Fraction of chunks containing ≥1 ground-truth claim | RAGChecker | LLM + NLI | Expensive |
| **ARES Context Relevance** | Fine-tuned lightweight DeBERTa judge + PPI confidence intervals | Saad-Falcon et al., NAACL 2024 (arXiv:2311.09476) | Fine-tuned LM + PPI | Medium |
| **Noise Sensitivity** (RAGAS) | System robustness to irrelevant retrieved passages | RAGAS v0.4 | LLM-judge | Medium |

### Axis 2 — Answer quality

| Metric | Definition | Source | Method | Cost |
|--------|-----------|--------|--------|------|
| **BLEU** | N-gram precision (1–4) with brevity penalty | Papineni et al. 2002 | Automatic | Cheap |
| **ROUGE-L** | Longest common subsequence recall | Lin 2004 | Automatic | Cheap |
| **METEOR** | Precision/recall with stemming, synonyms, fragmentation penalty | Banerjee & Lavie 2005 | Automatic | Cheap |
| **Exact Match** | Binary string match after normalization | SQuAD convention | Automatic | Cheap |
| **Token F1** | Token-level overlap F1 | SQuAD convention | Automatic | Cheap |
| **BERTScore** | Token-level cosine similarity of contextual embeddings; greedy matching, optional IDF | Zhang et al., ICLR 2020 (arXiv:1904.09675) | Embedding-based | Medium |
| **BLEURT** | Learned metric: BERT fine-tuned on human ratings | Sellam et al., ACL 2020 | Learned model | Medium |
| **BARTScore** | Log-probability of generating target given source via BART | Yuan et al., NeurIPS 2021 | Generation probability | Medium |
| **MoverScore** | Word Mover's Distance with contextual embeddings | Zhao et al., EMNLP 2019 | Embedding + optimal transport | Medium |
| **Answer Semantic Similarity** (RAGAS) | Cosine similarity of sentence embeddings | RAGAS | Embedding | Medium |
| **G-Eval** | LLM + CoT + probability-weighted scoring on custom criteria; Spearman ρ≈0.514 with humans | Liu et al., EMNLP 2023 (arXiv:2303.16634) | LLM-judge (CoT) | Expensive |
| **Answer Relevancy** (RAGAS) | Reverse question generation + cosine similarity to original query | RAGAS (arXiv:2309.15217) | LLM + embedding | Medium |
| **Answer Correctness** (RAGAS) | 0.75 × factual F1 (claim-level) + 0.25 × semantic similarity | RAGAS | LLM + embedding | Medium |
| **Factual Correctness** (RAGAS) | Decomposes response and reference into claims; classifies TP/FP/FN via NLI | RAGAS v0.2+ | LLM (NLI-style) | Medium |
| **RAGChecker Overall P/R/F1** | Claim-level precision, recall, F1 between response claims and ground truth | RAGChecker (arXiv:2408.08067) | LLM claim extraction + NLI | Expensive |
| **Completeness** | Whether answer addresses all query aspects; Likert scoring | Various; RAGVue | LLM-judge | Medium |
| **Conciseness** | Brevity relative to information content; penalizes verbose copying | G-Eval customizable | LLM-judge | Medium |
| **Coherence** | Logical flow and structural quality | G-Eval | LLM-judge | Medium |
| **DeCE** | Decomposed criteria-based evaluation with instance-specific rubrics | arXiv:2509.16093 (2025) | LLM-judge | Expensive |
| **CheckEval** | LLM-generated rubric checklists for structured evaluation | Lee et al. 2024 | LLM-judge | Expensive |
| **Prometheus-2** | Open-source LLM specialized in rubric-based evaluation; 7B and 8×7B variants | Kim et al., EMNLP 2024 (arXiv:2405.01535) | Specialized LM judge | Medium |

### Axis 3 — Faithfulness and grounding

| Metric | Definition | Source | Method | Cost |
|--------|-----------|--------|--------|------|
| **Faithfulness** (RAGAS) | (# response claims supported by context) / (total claims); LLM extracts + verifies | RAGAS (arXiv:2309.15217) | LLM claim decomposition + verification | Medium |
| **Faithfulness with HHEM** | Same pipeline but uses Vectara HHEM-2.1-Open (T5 classifier) for verification | RAGAS + Vectara (Bao et al. 2024) | NLI classifier | Cheap |
| **FActScore** | % of atomic facts supported by a knowledge source (e.g., Wikipedia) | Min et al., EMNLP 2023 (arXiv:2305.14251) | LLM + retrieval | Expensive |
| **RAGChecker Generator Suite** | 6 diagnostic metrics: Context Utilization, Noise Sensitivity (relevant/irrelevant), Hallucination, Self-Knowledge, Faithfulness | RAGChecker (arXiv:2408.08067) | LLM claim + entailment | Expensive |
| **RefChecker** | Decomposes to knowledge triplets (S,P,O); checks Entail/Contradict/Neutral against reference | Hu et al., EMNLP 2024 (arXiv:2405.14486) | LLM or NLI (triplet verification) | Expensive |
| **Citation Recall** (ALCE) | Fraction of cited sentences fully supported by citations (TRUE NLI model) | Gao et al., EMNLP 2023 (arXiv:2305.14627) | NLI model | Medium |
| **Citation Precision** (ALCE) | Fraction of citations that are relevant | ALCE | NLI model | Medium |
| **AIS** | Binary per-sentence: attributable to identified sources | Rashkin et al. 2023 | NLI or human | Medium |
| **AlignScore** | Unified alignment function; RoBERTa-based (355M params); chunk-sentence splitting | Zha et al., ACL 2023 (arXiv:2305.16739) | NLI model (RoBERTa) | Cheap |
| **SummaC-ZS / SummaC-Conv** | Sentence-pair NLI matrix aggregation; no claim decomposition needed | Laban et al., TACL 2022 | NLI model | Cheap |
| **Luna** | DeBERTa-v3-Large fine-tuned hallucination classifier; token-level; up to 16K tokens; 30–60 ex/sec | Belyi et al. 2025 (arXiv:2406.00975) | Fine-tuned NLI encoder | Cheap |
| **LettuceDetect** | ModernBERT token-classification for hallucination detection; F1=79.2% on RAGTruth; ~30× smaller than LLM methods | arXiv:2502.17125 (2025) | Token classification | Cheap |
| **HHEM-2.1-Open** | T5-based hallucination classifier from Vectara; free, small | Vectara (Bao et al. 2024) | Specialized classifier | Cheap |
| **SelfCheckGPT** | Sample multiple responses; measure inter-response consistency | Manakul et al., EMNLP 2023 | Sampling-based | Medium |

### Axis 4 — Robustness

| Metric | Definition | Source | Method | Cost |
|--------|-----------|--------|--------|------|
| **NDR (No-Degradation Rate)** | Fraction of queries where RAG ≥ non-RAG performance, across all (k, ordering) combos | Cao et al. 2025 (arXiv:2505.21870) | LLM-judge, multi-pass | High |
| **RSR (Retrieval Size Robustness)** | Whether performance improves or holds as k increases | Cao et al. 2025 | Multi-pass LLM-judge | High |
| **ROR (Retrieval Order Robustness)** | 1 − 2σ[f(q,k,o)] over orderings; higher = more consistent | Cao et al. 2025 | 3× inference per (q,k) | Medium |
| **Noise Robustness** (RGB) | Accuracy under varying noise ratios (0–80% irrelevant docs) | Chen et al., AAAI 2024 (arXiv:2309.01431) | Standard QA accuracy | Low |
| **Negative Rejection** (RGB) | Rejection rate when all docs are irrelevant | RGB | String match for abstention | Low |
| **Information Integration** (RGB) | Accuracy on multi-doc synthesis questions | RGB | QA accuracy | Low |
| **Counterfactual Robustness** (RGB) | Accuracy with factually incorrect docs + error detection/correction rates | RGB | QA + LLM-judge | Medium |
| **RARE Query/Document Perturbation** | Performance delta under systematic KG-driven perturbations | Zeng et al. 2025 (arXiv:2506.00789) | Multi-pass evaluation | High |
| **QE-RAG Entry Error Tolerance** | Performance under keyboard proximity errors, typos, spelling mistakes | Zhang et al. 2025 (arXiv:2504.04062) | Standard eval with corrupted queries | Low |
| **Linguistic Variation Sensitivity** | Performance delta across stylistic/register changes | Cao et al. 2025 (arXiv:2504.08231) | Multi-pass | Medium |
| **Corpus Poisoning Resistance** | Attack success rate under adversarially poisoned passages | BadRAG (Xue et al. 2024), TrojanRAG (Cheng et al. 2024) | Adversarial evaluation | High |
| **SURE Formatting Sensitivity** | Vulnerability to formatting variations in grounding data | Yang et al. 2025 (arXiv:2503.05587) | Multi-format testing | Medium |

### Axis 5 — Efficiency

| Metric | Definition | Method | Cost |
|--------|-----------|--------|------|
| **TTFT (Time to First Token)** | Latency from query submission to first output token | Automatic instrumentation | Cheap |
| **Total Latency** | End-to-end time from query to complete response | Automatic | Cheap |
| **Retrieval vs Generation Latency** | Breakdown of time in retrieval, reranking, and generation stages | Automatic | Cheap |
| **Cost per Query** | Total token usage × price per token (or compute cost for local models) | Automatic | Cheap |
| **Token Usage** | Input + output token counts | Automatic | Cheap |
| **Throughput** | Queries processed per unit time | Automatic | Cheap |
| **Tool Call Latency** | Time spent in tool execution per step | Automatic instrumentation | Cheap |

### Axis 6 — Agent-specific metrics

| Metric | Definition | Source | Method | Cost |
|--------|-----------|--------|--------|------|
| **ToolCallF1** (RAGAS) | F1 of predicted tool calls (name + args) vs reference | RAGAS v0.4 | Deterministic comparison | Cheap |
| **Tool Correctness** (DeepEval) | Binary: were all required tools called? | DeepEval | Deterministic | Cheap |
| **Tool Selection Recall@K** | Correct tool appears in top-K candidates | ToolBench, GRETEL (Wu et al. 2025) | Deterministic | Cheap |
| **Argument Correctness** | Were tool parameters formatted correctly? | DeepEval | Deterministic | Cheap |
| **Pass Rate** (ToolBench) | Proportion of instructions successfully completed within API call budget | Qin et al., ICLR 2024 (arXiv:2307.16789) | LLM-judge | Medium |
| **Win Rate** (ToolBench) | Pairwise preference between solution paths | ToolBench | LLM-judge | Medium |
| **AST/DAG Accuracy** | Structural comparison of predicted vs gold action dependency graphs | MCPToolBench++, MTU-Bench | Deterministic (graph edit) | Cheap |
| **AgentGoalAccuracy** (RAGAS) | Whether agent achieved user's goal | RAGAS | LLM-judge | Medium |
| **Task Completion / Success Rate** | Binary: did agent achieve the predefined goal? | AgentBench (Liu et al., ICLR 2024, arXiv:2308.03688) | Environment-specific | Medium |
| **Plan Quality** (DeepEval) | LLM evaluates reasoning-layer plan | DeepEval | LLM-judge | Medium |
| **Plan Adherence** (DeepEval) | Did the agent follow its own plan? | DeepEval | LLM-judge | Medium |
| **Step Efficiency** (DeepEval) | Were all steps necessary and minimal? | DeepEval | LLM-judge | Medium |
| **Trajectory Match** (LangChain) | Strict/unordered/subset/superset comparison against reference trajectory | LangChain AgentEvals | Deterministic | Cheap |
| **Error Correction Rate** | Fraction of errors self-corrected through reflection | Tool-MVR (Ma et al. 2025) | Comparison pre/post-reflection | Medium |
| **Multi-Hop Success Rate** | Accuracy on queries requiring 2–4 document hops | MultiHop-RAG (Tang & Yang, COLM 2024, arXiv:2401.15391) | QA accuracy | Medium |
| **TopicAdherenceScore** (RAGAS) | Whether agent stays within predefined topic domains | RAGAS | LLM-judge | Medium |
| **TRACE Hierarchical Utility** | Process efficiency + cognitive quality of trajectory | TRACE (arXiv:2602.21230, 2025) | Composite (LLM + rule-based) | High |

### Axis 7 — User-facing and quality

| Metric | Definition | Method | Cost |
|--------|-----------|--------|------|
| **Readability / Clarity** | Linguistic quality and ease of comprehension | LLM-judge or readability indices | Low–Medium |
| **Actionability** | Whether the response provides actionable guidance | LLM-judge | Medium |
| **Format Quality** | Correct formatting (markdown, lists, code blocks) | Rule-based or LLM-judge | Cheap–Medium |
| **CRAG 4-tier Scoring** | Perfect / Acceptable / Missing / Incorrect classification | Meta CRAG (Yang et al. 2024, arXiv:2406.04744) | Automated classification | Medium |

### Axis 8 — Calibration, safety, and consistency

| Metric | Definition | Source | Method | Cost |
|--------|-----------|--------|--------|------|
| **ECE (Expected Calibration Error)** | Avg \|accuracy − confidence\| across bins; ECE > 0.4 in RAG settings | Geng et al., NAACL 2024 | Post-hoc computation | Cheap |
| **AUROC** | Discrimination between correct/incorrect via confidence scores | Standard | Post-hoc | Cheap |
| **Brier Score** | Mean squared error between predicted probability and outcome | Standard | Post-hoc | Cheap |
| **Cross-Run Variance** | σ of scores across N runs at same/different temperatures | General practice | N× inference | Medium |
| **Semantic Consistency** | Embedding similarity of outputs across paraphrased inputs | Kuhn et al., ICLR 2023 | Embedding-based | Medium |
| **Guardrail Flip Rate** | Rate of safety judgment changes with/without RAG context | OpenReview 2025 | Comparison testing | Medium |
| **Toxicity / Harmful Content Rate** | Rate of toxic content via Perspective API or Llama Guard | Standard | Classifier-based | Cheap |
| **Bias / Counterfactual Fairness** | Performance disparities across demographic groups | Gan et al. survey (arXiv:2504.14891) | Scenario testing | Medium |

---

## 2. Framework comparison for production deployment

Nine major frameworks and three emerging alternatives were evaluated against the specific constraints (no cloud API, local models, Russian language, standalone metric usage).

| Feature | RAGAS v0.4 | RAGChecker | DeepEval | ARES | TruLens | LlamaIndex | Phoenix/Arize | MLflow 3 | Giskard |
|---------|-----------|------------|----------|------|---------|------------|---------------|----------|---------|
| **Metrics count** | 30+ | 10 | 50+ | 3 | ~10 | ~8 + integrations | ~5 + custom | Via integrations | ~5 + RAGAS |
| **Claim-level analysis** | No | **Yes (core)** | No | No | No | No | No | No | No |
| **GPT-4 required** | No | No | No | No | No | No | No | No | No |
| **vLLM/Ollama local** | ✅ Native | ✅ Via LiteLLM | ✅ CLI `set-local-model` | ✅ Fine-tuned local judges | ✅ Via providers | ✅ Via LlamaIndex | ✅ Via OpenInference | ✅ Via integrations | ✅ Via clients |
| **Russian support** | Via prompt `adapt()` | ❌ English spaCy | LLM-dependent | Needs multilingual training | LLM-dependent | LLM-dependent | LLM-dependent | LLM-dependent | LLM-dependent |
| **Standalone metrics** | ✅ Full | ✅ By group | ✅ Full | Partial | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **Statistical CIs** | No | No | No | **✅ PPI intervals** | No | No | No | No | No |
| **Agent metrics** | ToolCallF1, GoalAccuracy | No | 6+ agent metrics | No | No | No | No | No | No |
| **Tracing/observability** | No | No | Confident AI | No | ✅ OTel | Via integrations | **✅ Best-in-class** | ✅ MLflow Tracing | No |
| **CI/CD** | Basic | No | **✅ Pytest native** | No | No | No | No | ✅ MLflow | ✅ |
| **Human correlation** | Moderate | **Strongest** | Claimed | Good | Basic | Unknown | Unknown | Depends | Unknown |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT | MIT | MIT | ELv2 | Apache 2.0 | Apache 2.0 |
| **Active (2025)** | Very | Moderate | Very | Moderate | Active | Active | Very | Very | Transitioning |

**Best choices for the constrained system**: DeepEval offers the broadest metric coverage with native vLLM/Ollama support (`deepeval set-local-model --base-url="http://localhost:8000/v1/"`). RAGAS provides the most mature retrieval-focused metrics with prompt adaptation for Russian. RAGChecker delivers the strongest human-correlated diagnostics but requires adaptation for non-English. ARES uniquely provides statistical confidence intervals with minimal human annotations.

**Emerging 2025–2026 frameworks** worth monitoring include RAGVue (reference-free explainable diagnostics), GAICo (decoupled post-hoc evaluation, 16K+ PyPI downloads), TRACE (trajectory-aware agent evaluation), and LangChain AgentEvals (deterministic trajectory matching).

---

## 3. Recommendations for constrained system

The target system operates with Qwen3.5-35B-A3B (GGUF Q4_K_M, V100), ruBERT NLI, pplx-embed-v1, ~100 LLM calls/hour, 36–50 golden questions, 15 tools, Russian informal Telegram language, and zero external API access.

### What works without any API

**Deterministic metrics (zero LLM cost)**: Recall@k, Precision@k, MRR, nDCG, Hit Rate, MAP, Exact Match, Token F1, BLEU, ROUGE-L, ToolCallF1, Tool Correctness, AST/DAG Accuracy, Trajectory Match (strict/unordered), latency metrics, token usage, throughput. These form the backbone of any evaluation and should be computed on every test run.

**Embedding-based metrics (local inference)**: BERTScore with ai-forever/ruBert-large provides semantic similarity at ~50 ms/pair on V100. Answer Semantic Similarity via pplx-embed-v1 cosine distance adds negligible cost. These are ideal for Russian and run entirely locally.

**NLI-based faithfulness (no LLM needed)**: A SummaC-style pipeline using `cointegrated/rubert-base-cased-nli-threeway` can check sentence-level faithfulness by constructing an NLI pair matrix between context sentences and answer sentences, then aggregating entailment scores. This model predicts entailment/contradiction/neutral, was trained on translated MNLI+SNLI+ANLI+FEVER, and has **152K+ runs on HuggingFace**. It handles Russian morphology natively and requires no claim decomposition step.

**LLM-judge metrics (~100 calls/hour budget)**: With Qwen3.5-35B at 100 calls/hour, evaluating 50 golden questions allows roughly 2 LLM judge calls per question per metric. Prioritize binary (Pass/Fail) assessments over continuous scales — research shows smaller judges are significantly more reliable on binary decisions. Use DeepEval's `set-local-model` to route all judge calls through a local vLLM endpoint serving Qwen.

### What requires adaptation for Russian

RAGChecker's claim extraction pipeline depends on English spaCy (`en_core_web_sm`). For Russian, replace with `ru_core_news_lg` and verify that claim segmentation handles Telegram-style informal text. RAGAS prompts can be adapted via `BasePrompt.adapt()` — translate evaluation prompts to Russian for Qwen to judge in its stronger language. AlignScore is explicitly English-only and cannot be used directly; the SummaC + ruBERT-NLI approach is the recommended substitute.

---

## 4. Three-tier metric stack for the agentic pipeline

### Tier 1 — Always compute (zero/near-zero marginal cost)

These metrics require no LLM calls and should run on every evaluation cycle.

- **Retrieval**: Recall@5, Precision@5, MRR, Hit Rate — computed deterministically against golden retrieval labels
- **Answer surface**: ROUGE-L, Token F1 — cheap baselines for regression detection
- **Semantic similarity**: BERTScore (ai-forever/ruBert-large, layer ~18, IDF from domain corpus) — **primary answer quality metric for Russian**
- **Agent tools**: ToolCallF1 (RAGAS deterministic), Tool Correctness (binary), Trajectory Match (LangChain AgentEvals, unordered mode for flexible tool ordering)
- **Efficiency**: Total latency, retrieval vs generation latency, token usage per query, tool call count
- **NLI faithfulness**: SummaC-ZS pipeline with `cointegrated/rubert-base-cased-nli-threeway` — sentence-level faithfulness without any LLM call. Construct NLI pair matrix, take max entailment score per answer sentence, average across sentences. Expected throughput: **hundreds of evaluations per minute** on V100.
- **Consistency**: Cross-run variance (run 3× at temperature 0, report σ)

**Implementation**: Use DeepEval for ToolCallF1 and trajectory metrics. Use the `bert_score` Python package with custom model configuration. Build a lightweight SummaC wrapper around the ruBERT-NLI model.

### Tier 2 — LLM-judged metrics (budget: ~100 calls/hour)

These consume Qwen3.5-35B inference and should run on golden dataset evaluations.

With **50 golden questions** and **100 LLM calls/hour**, you can afford ~2 calls/question. Prioritize:

- **Answer Correctness** (RAGAS-style): Binary LLM judge call — "Does this answer correctly address the question given the reference?" Use DeepEval G-Eval with custom Russian rubric, binary output. **1 call/question.**
- **Faithfulness verification** (LLM-enhanced): For the ~10–15% of questions where SummaC-ZS flags low confidence, use Qwen as a second-pass verifier to check specific flagged sentences. **~0.15 calls/question average.**
- **Agent Goal Accuracy**: Binary LLM judge — "Did the agent achieve the user's goal?" **1 call/question.**
- **Plan Quality spot-check**: For 20% of questions, evaluate whether the tool sequence was reasonable. **0.2 calls/question.**

Total: ~2.35 calls/question × 50 = ~118 calls. At 100/hour, completes in **~70 minutes**.

For periodic deeper evaluation, add:
- **Answer Completeness**: Does the answer cover all aspects? Binary.
- **Noise Sensitivity spot-check**: Re-run 10 questions with injected noisy passages, compare output.
- **Robustness checks** (RGB-style): Run 10 questions with all-irrelevant context to test negative rejection.

### Tier 3 — Deep diagnostics (manual or batch)

Reserved for release gating, major architecture changes, or debugging.

- **Claim-level evaluation** (RAGChecker-style): Use Qwen to decompose answers into atomic claims, then verify each against context and ground truth. At ~5 calls/question, evaluate 20 questions in a ~60-minute batch. Reserve for investigating specific failure modes.
- **Full robustness suite**: NDR/RSR/ROR from Cao et al. requires multiple inference passes per query across varying k and orderings. Run as an overnight batch job on a subset of 20 questions.
- **Human evaluation**: Manual assessment through chat interface for 15–20 questions on subjective dimensions (readability, actionability, tone appropriateness for Telegram style). Use the CRAG 4-tier scoring (Perfect/Acceptable/Missing/Incorrect) for efficiency.
- **PPI confidence intervals** (ARES methodology): Fine-tune a lightweight DeBERTa/ruBERT judge on 150+ synthetic examples from your domain, then combine with your 50 human-annotated golden questions via PPI to get statistically grounded CIs. One-time setup cost, then runs cheaply.

---

## 5. Statistical power analysis for dataset sizing

The power to detect meaningful differences between RAG system versions depends critically on dataset size, the base metric rate, and the minimum effect you need to detect.

### Detection thresholds by dataset size

For a binary correctness metric with a baseline rate of ~70% (typical for production RAG), using a two-proportion z-test at α=0.05 and 80% power:

| Golden questions (n) | Detectable absolute difference | Cohen's d equivalent | 95% CI half-width | Practical interpretation |
|---------------------|-------------------------------|---------------------|-------------------|------------------------|
| **36** | ~15–20% | d ≈ 0.67 (large) | ±15% | Only detects catastrophic regressions |
| **50** | ~12–15% | d ≈ 0.57 | ±13% | Detects major quality changes |
| **100** | ~8–10% | d ≈ 0.40 (medium) | ±9% | Detects moderate improvements |
| **200** | ~5–7% | d ≈ 0.28 | ±6.4% | Detects meaningful incremental gains |
| **500** | ~3–4% | d ≈ 0.18 (small) | ±4% | Detects fine-grained optimizations |

**With 36–50 golden questions, you can reliably detect only large differences (≥12–15 percentage points).** This is adequate for comparing fundamentally different architectures or catching major regressions, but insufficient for fine-tuning decisions. The key mitigation strategies are:

**Bootstrap confidence intervals**: Sample with replacement B=1,000 times from your 50-question set, compute the metric on each sample, take the 2.5th and 97.5th percentiles. This provides honest uncertainty estimates without parametric assumptions. Use paired bootstrap (same questions, different systems) for tighter comparison CIs.

**Wilson intervals for binary metrics**: For small samples, Wilson intervals are more accurate than normal approximation. For 35/50 correct (70%): Wilson 95% CI = [56.2%, 81.0%]. This ~25 percentage-point width is the honest uncertainty you carry with 50 examples.

**PPI to amplify power**: ARES demonstrated that 150 human annotations combined with thousands of model-scored examples via PPI produces confidence intervals roughly equivalent to 500+ pure human annotations. Adapting this approach: use your 50 golden questions as the human validation set, generate 500+ synthetic evaluation pairs scored by Qwen, and apply PPI to tighten CIs by an estimated **2–3×**.

**Recommendation**: Start with 50 manually curated golden questions covering all 15 tools and key failure modes. Supplement with 200+ synthetic test cases (generated from your knowledge base using Qwen + RAGAS test set generation). Apply PPI to combine both for tighter statistical inference. Expand to 100 golden questions when resources allow — this is the sweet spot where the CI half-width drops below ±9%, enabling detection of moderate (~8%) improvements.

---

## 6. BERTScore configuration for Russian

### Model selection

For monolingual Russian-to-Russian evaluation (RAG answer vs. reference answer), **ai-forever/ruBert-large is the recommended model**. It was trained natively on Russian text with a Russian tokenizer, avoiding the subword fragmentation problems that plague multilingual models on morphologically rich languages. Bruches, Baturova & Bondarenko (2025, "BERTScore for Russian," Proceedings of ISP RAS) specifically evaluated models for Russian BERTScore across summarization, machine translation, and keyphrase generation tasks.

For cross-lingual scenarios (if ever comparing English and Russian texts), **xlm-roberta-large** is preferred, with layers 8–16 shown effective for Russian by the METCL-BERT framework (Wang 2026, Spearman 0.782 for en→ru).

### Layer selection and calibration

The BERTScore library does not include pre-tuned layer settings for ai-forever/ruBert-large. Manual tuning is required:

1. Use the `tune_layers` folder in the bert_score repository
2. For a 24-layer model like ruBert-large, **layers 17–20 typically perform best** for semantic similarity tasks — higher layers capture more semantic (vs. syntactic) information
3. Start with layer 18 as default; tune on a small dev set of 30–50 Russian sentence pairs with human similarity judgments
4. Command: `bert-score --model ai-forever/ruBert-large -l 18 --idf`

### IDF rescaling for Russian

IDF weighting is especially important for Russian because it downweights frequent function words and morphological variants while upweighting content-bearing terms:

1. Compute IDF statistics from a representative Russian corpus (your knowledge base, Russian Wikipedia, or both)
2. Pass via `bert_score.score(cands, refs, model_type="ai-forever/ruBert-large", idf=True, idf_sents=your_russian_corpus)`
3. For baseline rescaling (mapping raw scores to a human-readable range), you must compute your own baselines on Russian text since precomputed baselines exist only for standard models

### Morphological complexity handling

Russian's 6 cases × 2 numbers × 3 genders create significant subword fragmentation. Vetrov et al. (2022, arXiv:2203.05598) found that **combining incomplete WordPiece tokens into complete words** and averaging their vectors substantially improves BERTScore quality for Russian. This post-processing step is not built into the standard library — implement it as a custom tokenization wrapper. IDF weighting partially mitigates this by reducing weight on frequently occurring morphological affixes.

### Expected correlation with human judgments

Standard multilingual BERTScore on Russian achieves moderate correlation — **significantly below English-only performance**. Vetrov et al. report Spearman ρ ≈ 0.57 for translation quality (English→Russian). With the recommended configuration (ruBert-large, tuned layer, IDF, WordPiece aggregation), expect ρ ≈ 0.55–0.65 for Russian RAG answer evaluation. This is sufficient as one component in a metric ensemble but should not be the sole quality signal.

---

## 7. The NLI-based faithfulness pipeline for Russian

The most practical faithfulness evaluation approach for the constrained system avoids LLM calls entirely by adapting the SummaC methodology with a Russian NLI model.

### Architecture

The pipeline operates in four steps. First, segment both the retrieved context and the generated answer into individual sentences using `ru_core_news_lg` (spaCy) or `razdel` (a lightweight Russian sentence tokenizer designed for informal text, which handles Telegram-style writing better than spaCy). Second, construct an NLI pair matrix by running every (context_sentence, answer_sentence) pair through `cointegrated/rubert-base-cased-nli-threeway`, extracting the entailment probability. Third, aggregate using the SummaC-ZS approach: for each answer sentence, take the maximum entailment score across all context sentences (the "best supporting evidence" assumption), then average across answer sentences for the final faithfulness score. Fourth, flag sentences with max entailment below a threshold (e.g., 0.5) for Tier 2 LLM verification.

The `cointegrated/rubert-base-cased-nli-threeway` model is based on DeepPavlov/rubert-base-cased, fine-tuned on automatically translated NLI datasets (MNLI, SNLI, ANLI, FEVER, JOCI, MPE, SICK, IMPPRES). It outputs three classes: entailment, contradiction, neutral. For faithfulness scoring, use the entailment probability as the support score.

### Claim-level vs. sentence-level tradeoff

Sentence-level checking (SummaC-style) avoids the expensive LLM claim decomposition step but is less granular than RAGChecker's atomic claim approach. **For the target system, sentence-level is the right choice for Tier 1** because it requires zero LLM calls and processes hundreds of evaluations per minute. Reserve claim-level decomposition (using Qwen) for Tier 3 diagnostics on flagged cases.

Research shows that NLI-based sentence-level checking achieves ~74% balanced accuracy on English inconsistency detection (SummaC benchmark). For Russian, expect ~5–10% lower accuracy due to the translated training data and morphological complexity. The gap can be narrowed by fine-tuning on a small set of Russian faithfulness examples from your domain.

---

## 8. Key papers driving the field

The most influential works shaping RAG evaluation practice in this period form a clear progression from surface-level metrics toward diagnostic, claim-level, and trajectory-aware evaluation.

**RAGChecker** (Ru et al., NeurIPS 2024, arXiv:2408.08067) established that claim-level entailment checking produces **significantly stronger correlation with human judgments** than RAGAS, TruLens, or ARES across 280 human-annotated instances spanning 10 domains. Its key insight — decomposing answers into atomic claims and checking bidirectional entailment — has become the gold standard for high-fidelity evaluation.

**ARES** (Saad-Falcon et al., NAACL 2024, arXiv:2311.09476) introduced PPI-based confidence intervals to RAG evaluation, demonstrating that **150 human annotations** combined with fine-tuned lightweight judges can produce statistically grounded evaluation rivaling thousands of human labels. This is the only framework providing formal statistical guarantees.

**RGB** (Chen et al., AAAI 2024, arXiv:2309.01431) defined the four fundamental robustness abilities (noise robustness, negative rejection, information integration, counterfactual robustness) that remain the standard taxonomy for robustness testing. Its finding that LLMs tend to trust retrieved false information even when possessing correct internal knowledge is critical for agentic systems.

**Prometheus-2** (Kim et al., EMNLP 2024, arXiv:2405.01535) proved that **open-source 7B models can serve as effective evaluation judges** with 72–85% human agreement on pairwise ranking, making API-free evaluation viable. The 7B variant runs on 16GB VRAM.

The two comprehensive surveys — **Yu et al.** (arXiv:2405.07437, 2024) and **Gan et al.** (arXiv:2504.14891, 2025) — provide the most complete meta-analyses of evaluation practice, finding that traditional metrics (F1, EM, ROUGE) still dominate in published RAG papers by usage frequency, while LLM-based evaluation is growing fastest in 2024H2–2025.

---

## Conclusion

The RAG evaluation landscape has matured from surface-level n-gram metrics toward **claim-level diagnostic evaluation** (RAGChecker), **statistical inference** (ARES/PPI), and **trajectory-aware agent assessment** (TRACE, LangChain AgentEvals, DeepEval). Three novel insights emerge from this synthesis.

First, a fully local evaluation stack is not just feasible but increasingly competitive. The combination of ruBERT-NLI for sentence-level faithfulness, BERTScore with ruBert-large for semantic similarity, and Qwen-as-judge for binary assessments covers roughly **80% of what GPT-4-judged evaluation provides**, at zero marginal API cost. The emergence of lightweight hallucination detectors (Luna at 30–60 examples/second, LettuceDetect at 30× less compute than LLM methods) further closes this gap.

Second, for the specific 50-question golden dataset, **statistical power is the binding constraint**, not metric sophistication. With 50 examples, only differences exceeding ~13 percentage points are reliably detectable. The single highest-impact investment is expanding to 100+ golden questions and applying PPI to combine human and model annotations, which would roughly halve confidence interval widths.

Third, agentic evaluation is still the weakest link. While tool-level metrics (ToolCallF1, routing accuracy, AST matching) are deterministic and cheap, **trajectory-level quality assessment** — whether the agent chose a reasonable plan among alternatives — remains fundamentally dependent on LLM judges. The recommended approach is binary trajectory assessment ("Was this plan reasonable? Yes/No") using Qwen, which small judges handle more reliably than continuous scoring. DeepEval's PlanQualityMetric and LangChain AgentEvals' trajectory matchers provide the most production-ready implementations for this purpose.