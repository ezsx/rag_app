# rag_app — Production RAG with Agentic ReAct Pipeline

> Self-hosted RAG system over 36 Russian-language AI/ML Telegram channels.
> No managed APIs. No frameworks. Custom retrieval pipeline on local hardware.

**Factual: 0.84** | **Faithfulness: 0.91** | **Robustness: 0.954** | **Recall@3: 0.97** | **15 tools** | **36 channels, 13K docs**

---

## What it does

User asks a question about AI/ML news. ReAct agent plans sub-queries, runs hybrid retrieval (BM25 + dense + ColBERT) over 13K documents, filters with cross-encoder, produces a grounded answer with citations via SSE streaming.

```
Query → query_plan → search (BM25+Dense → RRF → ColBERT) → CE filter → compose_context → answer
```

15 LLM tools with phase-based dynamic visibility. Analytics tools (entity tracking, trend digests, channel expertise) short-circuit the search path when appropriate.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Windows Host                                                     │
│   llama-server.exe → V100 SXM2 32GB (TCC)                       │
│   Qwen3.5-35B-A3B MoE (Q4_K_M), port 8080                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│ WSL2 Native               │                                     │
│   gpu_server.py → RTX 5060 Ti 16GB                               │
│   pplx-embed-v1 + Qwen3-Reranker + jina-colbert-v2 + ruBERT-NLI │
│   port 8082                                                      │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│ Docker (CPU only)         │                                     │
│   FastAPI + Web UI (:8001)   Qdrant (:6333)                      │
│   Langfuse v3 (:3100) — observability                            │
└─────────────────────────────────────────────────────────────────┘
```

V100 in TCC mode poisons NVML in WSL2 — Docker GPU unavailable. All GPU workloads run natively. See [decision log](docs/architecture/11-decisions/decision-log.md).

---

## Eval Results

Claude judge (0.0-1.0 granular scale) + independent NLI faithfulness verification via ruBERT. Full metric definitions in [project scope](docs/progress/project_scope.md#автоматические-proxy-метрики-evaluate_agentpy-без-llm-judge).

**LLM judge metrics** (36 Qs golden_v2, [eval results](results/raw/eval_results_20260401-091242.json)):

| Metric | Value | Details |
|--------|-------|---------|
| **Factual correctness** | **0.842** | 36 Qs, Claude judge with 7 calibration examples |
| **Usefulness** | **1.778 / 2** | 36 Qs |
| **Key Tool Accuracy** | **1.000** | 36/36 correct tool selection |
| **Faithfulness** | **0.91** | 17 retrieval Qs, 171 claims verified, **0 hallucinations** ([analysis](results/reports/nli_faithfulness_analysis_20260401.md)) |
| **Retrieval recall@3** | **0.97** | 100 hand-crafted queries ([dataset](datasets/eval_retrieval_calibration.json)) |
| **Mean latency** | **24.4s** | Full pipeline including LLM inference, p95=65.6s |

**Automatic proxy metrics** (same run, no judge needed, [report](results/reports/eval_report_20260401-091242.json)):

| Metric | Value | Scope |
|--------|-------|-------|
| acceptable_set_hit | 0.471 | 17 retrieval Qs — at least one correct doc found |
| strict_anchor_recall | 0.588 (9 full, 6 zero) | 17 retrieval Qs — exact document ID match |
| coverage (LANCER) | 0.414 | 36 Qs — nugget-based query coverage |
| failure_breakdown | refusal_wrong: 2 | 36 Qs — agent refused when it shouldn't |

### Why standard proxy metrics fail (and why we still implement them)

We compute BERTScore F1, SummaC, Precision@5, MRR, nDCG@5 automatically on every eval run ([SPEC-RAG-22](docs/specifications/completed/SPEC-RAG-22-comprehensive-eval-metrics.md), [R26](docs/research/reports/R26-deep-comprehensive-rag-eval-metrics.md)). These are standard RAG metrics used across the industry. Our finding: **they are poor proxies for actual answer quality** on Russian-language agent outputs.

**Evidence: BERTScore vs Claude judge on the same 151 answers** ([BERTScore raw](results/robustness/ndr_rsr_ror_raw_20260402-082135.json), [Claude judge](results/robustness/judge_ndr_rsr_ror_final.json)):

| What we measured | BERTScore F1 | Claude Judge | Gap |
|-----------------|:---:|:---:|-----|
| NDR (retrieval helps?) | 0.818 | **0.963** | BERTScore underestimates by 0.145 |
| RSR violations (quality drops at higher k?) | 5 false violations | **1 real** | 4 ghost violations from semantic similarity noise |
| RSR rate | 0.706 | **0.941** | |
| ROR (order matters?) | 0.974 | **0.959** | Roughly agrees |
| Mean factual at k=20 | ~0.45 | **0.63** | BERTScore compresses the scale |
| Mean factual at k=0 | ~0.40 | **0.10** | BERTScore can't tell refusal from answer |

Root cause: BERTScore measures semantic similarity, not factual correctness. A "confident refusal" (`Я не могу ответить на этот вопрос`) gets high BERTScore against expected answer because both are fluent Russian text about the same topic. Claude judge correctly scores it 0.0.

**Standard IR proxy metrics** (agent eval batches, [reports](results/reports/)):

| Metric | Value | Scope | vs Claude judge |
|--------|-------|-------|-----------------|
| BERTScore F1 (ruBERT-large) | **0.52** | 36 Qs | Judge factual = 0.84. BERTScore shows ~0.5 for correct and incorrect answers alike |
| SummaC faithfulness | **0.37** | retrieval Qs | Judge faithfulness = 0.91. SummaC sentence-level NLI misses cross-lingual paraphrases |
| Precision@5 | **0.10** | retrieval Qs | Underestimates: agent cites 5-8 docs, many relevant but not in narrow expected set |
| MRR | **0.50** | retrieval Qs | Expected doc not always at rank 1, but agent compensates via multi-query |
| nDCG@5 | **0.50** | retrieval Qs | Same limitation as MRR: narrow expected set vs broad retrieval |

These metrics are standard in RAG evaluation literature but **systematically underestimate** our pipeline quality on Russian-language agent outputs. We implement them for completeness ([SPEC-RAG-22](docs/specifications/completed/SPEC-RAG-22-comprehensive-eval-metrics.md), [evaluate_agent.py](scripts/evaluate_agent.py)) and to demonstrate the gap.

**Evidence: NLI faithfulness on full 36Q eval** ([nli_scores](results/raw/nli_scores_20260401_full.json), 1977 NLI pairs):

| Metric | Value |
|--------|-------|
| faithfulness (raw) | 0.792 |
| faithfulness (corrected) | **~0.91** |
| citation_precision | 0.509 |
| claims supported | 133 / 171 (78%) |
| contradictions (raw / real) | 19 / **0** |

19 raw contradictions manually reviewed ([analysis](results/reports/nli_faithfulness_analysis_20260401.md)): 12 ruBERT false positives on Russian paraphrases (e.g. "Дженсен Хуанг (Nvidia)" not matched to "гендиректор NVIDIA"), 5 wrong-doc matches, 2 borderline. Zero actual hallucinations. SummaC sentence-level faithfulness available as a lighter alternative ([src/services/eval/summac.py](src/services/eval/summac.py)).

**Conclusion**: LLM judge (Claude) remains the only reliable scoring method for our domain. Automatic metrics serve as diagnostic signals, not primary quality measures. We implement them for completeness and to demonstrate the gap — a finding consistent with recent literature on RAG evaluation limitations.

### Robustness ([Cao et al. 2025](docs/progress/experiment_log.md#методология-наша-vs-cao-et-al-2025-arxiv2505-21870) adapted)

Bypass pipeline: direct Qdrant + LLM, controlled k and ordering. 151 answers scored ([raw data](results/robustness/ndr_rsr_ror_raw_20260402-082135.json)).

| Metric | BERTScore (proxy) | Claude Judge (final) | Finding |
|--------|:-:|:-:|---------|
| **NDR** | 0.818 | **0.963** (26/27) | BERTScore underestimated by 0.145 |
| **RSR** | 0.706 | **0.941** (16/17) | BERTScore showed false violations |
| **ROR** | 0.974 | **0.959** | Roughly correct |
| **Composite** | 0.826 | **0.954** | |

**BERTScore F1 failed as a robustness proxy**: semantic similarity doesn't capture factual correctness — a "confident refusal" scores high similarity to the expected answer. Claude judge is required for final numbers. [Judge scores](results/robustness/judge_ndr_rsr_ror_final.json).

Retrieval adds **+0.53** absolute factual improvement (k=0: 0.10, k=20: 0.63). RSR monotonicity confirmed: k=3 (0.52) < k=5 (0.59) < k=10 (0.60) < k=20 (0.63).

57 eval runs across development. Full [experiment log](docs/progress/experiment_log.md) with per-question analysis.

### Custom vs LlamaIndex Benchmark

Built the same pipeline in LlamaIndex (best-effort) and measured against our custom implementation.
4 pipelines, same LLM, same data, same questions. [Full spec](docs/specifications/completed/SPEC-RAG-29-framework-comparison-benchmark.md). Research: [R27](docs/research/reports/R27-framework-benchmark-methodology.md).

**Agent E2E** (17 questions, judge: Claude Opus 4.6):

| Pipeline | Factual | Usefulness | Grounding | Latency |
|----------|:---:|:---:|:---:|:---:|
| Naive (dense + LLM) | 0.55 | 1.04 | 0.28 | ~4s |
| LlamaIndex stock | 0.51 | 1.13 | 0.46 | ~9s |
| LlamaIndex maxed (weighted RRF + CE) | 0.54 | 1.21 | 0.48 | ~11s |
| **Custom pipeline** | **0.84** | **1.77** | **0.88** | **~30s** |

Custom wins by **+0.30 factual**, **+0.56 usefulness**, **+0.40 grounding** vs best framework config.

**Retrieval-only** (100 auto-generated queries — exact text fragments from posts):

| Pipeline | Recall@1 | Recall@5 | MRR | Latency |
|----------|:---:|:---:|:---:|:---:|
| Naive (dense only) | 0.820 | 0.920 | 0.861 | 0.1s |
| LlamaIndex stock | 0.820 | 0.920 | 0.861 | 0.1s |
| LlamaIndex maxed | 0.880 | 0.940 | 0.907 | 1.4s |
| **Custom (RRF + ColBERT)** | **0.939** | **0.949** | **0.944** | **0.2s** |

Custom wins clearly: **+12% Recall@1**, **+8% MRR** vs LlamaIndex maxed. ColBERT token-level matching shines on exact term queries (LLM, MoE, SSM).

**Retrieval-only** (100 hand-crafted natural-language queries):

| Pipeline | Recall@1 | Recall@5 | MRR | Latency |
|----------|:---:|:---:|:---:|:---:|
| Naive (dense only) | 0.730 | 0.940 | 0.825 | 0.1s |
| LlamaIndex stock | 0.730 | 0.940 | 0.825 | 0.1s |
| LlamaIndex maxed | 0.780 | 0.980 | 0.865 | 1.4s |
| **Custom (RRF + ColBERT)** | **0.780** | **0.970** | **0.866** | **0.2s** |

On natural-language queries the retrieval gap narrows — dense embedding already captures semantics well. ColBERT ≈ cross-encoder here. The real differentiation comes from the agent layer (query planning, LANCER, specialized tools), not retrieval tuning alone.

Key findings:
- **LlamaIndex stock = naive**: default hybrid fusion adds zero gain on both datasets
- **Reranker is not the differentiator**: li_maxed ≈ li_stock on agent E2E (+0.03 factual)
- **Multi-query planning + LANCER + specialized tools = main gain source** (not retrieval tuning)
- **Grounding 0.88 vs 0.48**: inline `[1][2][3]` citations via `compose_context` → `final_answer`
- **Custom 7x faster** than LlamaIndex maxed on retrieval (direct HTTP vs framework abstraction)
- LlamaIndex adds ~70 transitive dependencies vs our ~12

Full per-question breakdown in [judge_scores.md](benchmarks/results/judge_scores.md). Agent answers: [custom](benchmarks/results/agent_answers.json), [naive + LI](benchmarks/results/agent_naive_listock.json), [LI-maxed](benchmarks/results/agent_limaxed.json). Retrieval: [auto-generated](benchmarks/results/retrieval_auto_generated.json), [calibration](benchmarks/results/retrieval_calibration.json).

### What didn't work (with evidence)

All rejected with measured evidence. Details in [experiment log](docs/progress/experiment_log.md#протестировано-и-отклонено-с-evidence).

| Technique | Result | Why rejected |
|-----------|--------|--------------|
| Cosine MMR | recall 0.70 → 0.11 | Re-promotes attractor documents |
| Dense re-score after RRF | recall 0.33 → 0.15 | Erases BM25 contribution |
| PCA whitening 1024→512 | recall 0.70 → 0.56 | Too aggressive dimensionality cut |
| DBSF fusion | 0.72 vs RRF 0.73 | RRF slightly better |
| CE reranking after ColBERT | r@3: 0.97 → 0.94 | Degrades top-3, replaced with [CE filter](docs/architecture/11-decisions/decision-log.md) (DEC-0045) |
| Pipeline v2 (RRF→CE→ColBERT) | +0.02 r@2 only | Not worth complexity |
| BERTScore as robustness proxy | NDR off by 0.145 | Doesn't capture factual correctness |
| XLM-RoBERTa for Russian NLI | ent=0.006 on obvious pairs | ruBERT 150x better on Russian |
| Cosine-based coverage | 45% false refinements | Replaced with [LANCER nugget coverage](docs/architecture/11-decisions/decision-log.md) (DEC-0044) |

---

## Key Components

| Component | Model / Tech | Size | Where |
|-----------|-------------|------|-------|
| **LLM** | Qwen3.5-35B-A3B MoE (3B active) | Q4_K_M | V100 via llama-server |
| **Embedding** | pplx-embed-v1-0.6B (1024-dim) | bf16 | RTX 5060 Ti |
| **Reranker** | Qwen3-Reranker-0.6B (CRAG-style filter) | fp16 | RTX 5060 Ti |
| **ColBERT** | jina-colbert-v2 (128-dim MaxSim) | fp16 | RTX 5060 Ti |
| **NLI** | rubert-base-cased-nli-threeway | fp16, 0.36 GB | RTX 5060 Ti |
| **Vector store** | Qdrant (dense + sparse BM25 + ColBERT) | — | Docker |
| **Observability** | Langfuse v3 (self-hosted) | — | Docker |
| **Data** | 36 Telegram channels, 13K docs ([channel selection](docs/research/reports/R09-telegram-channels-collection.md)) | Jul 2025 - Mar 2026 | Qdrant |

## Retrieval Pipeline

```
BM25 top-100 ──┐
               ├── Weighted RRF (3:1) ── ColBERT rerank ── CE confidence filter ── Channel dedup
Dense top-20 ──┘
```

- **Multi-query**: LLM generates 3-5 sub-queries, each runs independent hybrid retrieval, round-robin merge
- **Original query injection**: user query always in subqueries for BM25 keyword match
- **LANCER nugget coverage**: query_plan subqueries as nuggets, targeted refinement on uncovered
- **CE filter** (not reranker): documents with score < 0 removed, ColBERT order preserved

## Agent Tools (15)

Phase-based dynamic visibility (max 5 visible), data-driven keyword routing from `datasets/tool_keywords.json`.

| Category | Tools |
|----------|-------|
| **Search** | `search`, `temporal_search`, `channel_search`, `cross_channel_compare`, `summarize_channel` |
| **Analytics** | `entity_tracker` (top/timeline/compare/co-occurrence), `arxiv_tracker` (top/lookup) |
| **Topics** | `hot_topics` (BERTopic weekly digest), `channel_expertise` (per-channel profiles) |
| **Planning** | `query_plan`, `list_channels` |
| **Enrichment** | `rerank`, `related_posts`, `compose_context` |
| **Synthesis** | `final_answer` |

---

## Observability

Self-hosted Langfuse v3. Every agent request produces a trace tree:

```
agent_request (root)
+-- llm_step_1 -> llm_chat_completion
+-- tool:query_plan -> query_planner
+-- llm_step_2 -> llm_chat_completion
+-- tool:search -> hybrid_retrieval (per subquery)
+-- tool:rerank -> CE confidence filter
+-- tool:compose_context
+-- llm_step_N_final -> llm_chat_completion
+-- tool:final_answer
+-- tool[system]:verify -> hybrid_retrieval
```

Rich output per span: hits_count, coverage, prompt_len, token usage. Error marking for failed tools. Root trace: plan, strategy, tokens, coverage, citations_count.

---

## Development Workflow

Built with AI coding agents (Claude Code, Codex) following a structured process:

```
Research (29 reports) → Specification (35 specs) → Implementation → Evaluation → Documentation
```

- **Research-driven**: every decision backed by deep research ([R00-R27](docs/research/reports/), [46 prompts](docs/research/prompts/))
- **Evaluation-first**: 57 eval runs, every change measured against [golden dataset](datasets/eval_golden_v2.json) ([experiment log](docs/progress/experiment_log.md))
- **Specs before code**: concrete acceptance criteria, reviewed by Codex ([35 completed specs](docs/specifications/completed/))
- **Architecture docs**: mirror current codebase ([system overview](docs/architecture/04-system/overview.md), [flows](docs/architecture/05-flows/))
- **Decision log**: [45 ADR entries](docs/architecture/11-decisions/decision-log.md) documenting every choice and why

---

## How We Run It

Our hardware: V100 SXM2 32GB (LLM inference), RTX 5060 Ti 16GB (embedding + reranker + ColBERT), Docker Desktop (CPU services).

```bash
# 1. LLM on V100 (PowerShell)
llama-server.exe -m models/Qwen3.5-35B-A3B-Q4_K_M.gguf --jinja --reasoning-budget 0 -c 32768

# 2. Embedding + Reranker + ColBERT + NLI on RTX 5060 Ti (WSL2)
CUDA_VISIBLE_DEVICES=0 python scripts/gpu_server.py --with-nli

# 3. Infrastructure (Docker)
docker compose -f deploy/compose/compose.langfuse.yml up -d  # Langfuse
docker compose -f deploy/compose/compose.dev.yml up -d        # API + Qdrant

# Web UI: http://localhost:8001
# Langfuse: http://localhost:3100
```

## Project Structure

```
src/
  adapters/             Qdrant, LLM (llama-server), TEI, hybrid retriever
  api/                  FastAPI endpoints + SSE streaming
  services/             Agent service (decomposed), 15 tools, query planner
  services/agent/       State, coverage, executor, routing, formatting
  services/eval/        NLI faithfulness verification (eval-only)
  core/                 Settings, DI, observability (Langfuse)
scripts/                GPU server, evaluation, NLI, ingestion, calibration
docs/
  architecture/         Source of truth (45 decisions, flows, data model)
  research/             46 prompts + 29 reports (R00-R27) + audio transcripts
  specifications/       35 completed specs
  progress/             Project scope + experiment log (57 runs)
benchmarks/             Framework comparison (LlamaIndex vs custom, 4 pipelines)
datasets/               Golden dataset (36 Qs), calibration (100 Qs), prompts, entity dictionary
deploy/                 Docker compose (dev, langfuse, test, benchmark)
```

## License

Private project. Not open-sourced.
