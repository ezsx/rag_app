# rag_app — Production RAG with Agentic ReAct Pipeline

> Most RAG portfolios show a LangChain demo. This project shows:
> 1. A custom pipeline that **beats LlamaIndex by +0.30 factual** on identical data and model
> 2. An evaluation framework with **57 runs, cross-family judge consensus** (Claude Opus + GPT-5.4), and NLI-verified faithfulness
> 3. **39 ablation experiments** with honest reporting of what didn't work
>
> Self-hosted on two GPUs. No managed APIs. No frameworks.

![Factual](https://img.shields.io/badge/Factual-0.858-brightgreen) ![Useful](https://img.shields.io/badge/Useful-1.71%2F2-brightgreen) ![Faithfulness](https://img.shields.io/badge/Faithfulness-0.91-brightgreen) ![Robustness](https://img.shields.io/badge/Robustness-0.954-brightgreen) ![Recall@5](https://img.shields.io/badge/Recall%405-0.900-blue) ![Tools](https://img.shields.io/badge/Tools-15-blue) ![Docs](https://img.shields.io/badge/Docs-13K-blue) ![License](https://img.shields.io/badge/License-Apache%202.0-orange)

---

## What it does

User asks a question about AI/ML news. ReAct agent plans sub-queries, runs hybrid retrieval (BM25 + dense + ColBERT) over 13K documents from 36 Russian-language Telegram channels, filters with cross-encoder + cosine recall guard, produces a grounded answer with citations via SSE streaming.

```
Query → query_plan → multi-query search (BM25+Dense → RRF → ColBERT) → MMR merge → CE re-sort + adaptive filter → compose_context → answer
```

15 LLM tools with phase-based dynamic visibility. Analytics tools (entity tracking, trend digests, channel expertise) short-circuit the search path when appropriate.

<!-- TODO: GIF of live SSE-streaming response (10-15 sec) -->
<!-- TODO: place demo.gif here -->

<details>
<summary><b>Example: real query trace</b> (from eval RUN-008)</summary>

```
User: "Кого Financial Times назвала человеком года в 2025?"
→ query_plan: 3 subqueries (k=10, fusion=rrf)                    3.7s
→ search: 28 docs retrieved (hybrid BM25+dense, 3 subqueries)    2.5s
→ rerank: CE scores [6.6, 5.2, 3.5, 2.5, 2.4] → 5 docs kept    1.9s
→ compose_context: 5 citations from 3 channels, coverage 0.70    0.002s
→ final_answer: 845 chars with inline [1][2][3] citations
Total: 31.7s (LLM inference ~25s = 84%, retrieval+rerank ~8s)

Answer: "Financial Times назвала «Человеком года» в 2025 году Дженсена Хуанга,
основателя и генерального директора NVIDIA [1][2]. Издание отметило ключевую
роль Хуанга в трансформации полупроводниковой индустрии..."
Sources: techsparks, ai_machinelearning_big_data, data_secrets
```

</details>

**Why 13K docs, not millions?** Small corpus is harder for retrieval — no redundancy to compensate for weak search. If the relevant document is one of three, not one of thousands, you need precision. HNSW search is sublinear; scaling to 1M+ docs adds single-digit milliseconds to retrieval, not orders of magnitude. The Telegram domain (mixed languages, emoji, slang, forwards, code snippets) is harder than clean Wikipedia/arxiv text — no ready benchmarks exist.

---

## Architecture

```mermaid
graph TB
    subgraph WIN["Windows Host"]
        LLAMA["llama-server.exe<br/>Qwen3.5-35B-A3B MoE Q4_K_M<br/>V100 SXM2 32GB (TCC)<br/>:8080"]
    end

    subgraph WSL["WSL2 Native"]
        GPU["gpu_server.py<br/>RTX 5060 Ti 16GB<br/>:8082"]
        EMB["pplx-embed-v1-0.6B"]
        RR["Qwen3-Reranker-0.6B"]
        COL["jina-colbert-v2"]
        NLI["ruBERT-NLI"]
        GPU --- EMB & RR & COL & NLI
    end

    subgraph DOCKER["Docker (CPU only)"]
        API["FastAPI + Web UI<br/>:8001"]
        QD["Qdrant<br/>dense + sparse + ColBERT<br/>:6333"]
        LF["Langfuse v3<br/>:3100"]
    end

    API -- "LLM inference" --> LLAMA
    API -- "embed / rerank / colbert" --> GPU
    API -- "vector search" --> QD
    API -. "traces" .-> LF
```

V100 in TCC mode poisons NVML in WSL2 — Docker GPU unavailable. All GPU workloads run natively. See [decision log](docs/architecture/11-decisions/decision-log.md).

---

## Eval Results

**Judge methodology**: cross-family consensus — Claude Opus 4.6 (Anthropic) + GPT-5.4 via Codex CLI (OpenAI) + manual calibration on hard cases. Eliminates single-judge bias. Granular scale: factual 0.0-1.0 (step 0.1), useful 0.0-2.0 (step 0.1). Independent NLI faithfulness verification via ruBERT.

**LLM judge metrics** (36 Qs golden_v2, [latest run](experiments/runs/RUN-008/results.yaml)):

| Metric | Value | Details |
|--------|-------|---------|
| **Factual correctness** | **0.858** | 95% CI **[0.792, 0.917]**, n=36, cross-family judge consensus |
| **Usefulness** | **1.71 / 2** | 95% CI **[1.606, 1.803]**, n=36 |
| **Key Tool Accuracy** | **1.000** | 36/36 correct tool selection |
| **Faithfulness** | **0.91** | 17 retrieval Qs, 171 claims verified, **0 hallucinations** ([analysis](experiments/legacy/reports/nli_faithfulness_analysis_20260401.md)) |
| **Retrieval R@5** | **0.900** | 120 natural-language queries, 6 categories |
| **Correct refusal** | **3/3** | Agent correctly refuses out-of-scope queries |
| **Mean latency** | **~30s** | LLM inference 84% (self-hosted Qwen3.5-35B on V100), retrieval ~2.5s, CE rerank ~2s. With managed API: ~6-8s estimated |

Bootstrap confidence via [`scripts/compute_confidence.py`](scripts/compute_confidence.py): retrieval factual **0.888** (95% CI [0.782, 0.965], n=17), analytics factual **0.793** (95% CI [0.679, 0.893], n=14). Intervals are intentionally wide at n=36; expanding the golden set to 100+ questions is the next step.

### Why standard proxy metrics fail (and why we still implement them)

We compute BERTScore F1, SummaC, Precision@5, MRR, nDCG@5 on every eval run ([SPEC-RAG-22](docs/specifications/completed/SPEC-RAG-22-comprehensive-eval-metrics.md)). Finding: **they systematically underestimate pipeline quality** on Russian-language agent outputs.

| Proxy metric | Value | LLM Judge | Gap |
|-------------|:---:|:---:|-----|
| BERTScore F1 | 0.52 | Factual **0.858** | Can't distinguish refusal from answer — semantic similarity ≠ factual correctness |
| SummaC faithfulness | 0.37 | **0.91** | Misses cross-lingual paraphrases ("Хуанг (Nvidia)" ≠ "гендиректор NVIDIA") |
| Precision@5 | 0.10 | acceptable_set_hit 0.47 | Agent cites 5-8 docs, many relevant but not in narrow expected set |

19 raw NLI contradictions manually reviewed → **0 actual hallucinations** (12 ruBERT false positives on Russian paraphrases, 5 wrong-doc matches, 2 borderline). [Full analysis](experiments/legacy/reports/nli_faithfulness_analysis_20260401.md).

**Conclusion**: LLM judge remains the only reliable scoring method for our domain. Automatic metrics serve as diagnostic signals. We implement them to demonstrate the gap — consistent with recent literature on RAG evaluation limitations. Details: [R26](docs/research/reports/R26-deep-comprehensive-rag-eval-metrics.md), [robustness data](experiments/legacy/robustness/).

### Robustness ([Cao et al. 2025](docs/progress/experiment_log.md#методология-наша-vs-cao-et-al-2025-arxiv2505-21870) adapted)

Bypass pipeline: direct Qdrant + LLM, controlled k and ordering. 151 answers scored ([raw data](experiments/legacy/robustness/ndr_rsr_ror_raw_20260402-082135.json)).

| Metric | BERTScore (proxy) | Claude Judge (final) | Finding |
|--------|:-:|:-:|---------|
| **NDR** | 0.818 | **0.963** (26/27) | BERTScore underestimated by 0.145 |
| **RSR** | 0.706 | **0.941** (16/17) | BERTScore showed false violations |
| **ROR** | 0.974 | **0.959** | Roughly correct |
| **Composite** | 0.826 | **0.954** | |

**BERTScore F1 failed as a robustness proxy**: semantic similarity doesn't capture factual correctness — a "confident refusal" scores high similarity to the expected answer. Claude judge is required for final numbers. [Judge scores](experiments/legacy/robustness/judge_ndr_rsr_ror_final.json).

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

**Key insights** (applicable beyond this project):
- **Framework defaults add zero value**: LlamaIndex stock = naive on both datasets. Default hybrid fusion doesn't help without tuning
- **Agent layer is the differentiator, not retrieval tuning**: multi-query planning + LANCER coverage + specialized tools = main gain source. li_maxed reranker gave only +0.03 factual over stock
- **Grounding requires architecture, not prompting**: 0.88 vs 0.48 grounding comes from `compose_context` → `final_answer` citation pipeline, not from asking the LLM to "cite sources"
- **Custom 7x faster** on retrieval (direct HTTP vs framework abstraction), ~70 vs ~12 transitive dependencies

Full per-question breakdown in [judge_scores.md](experiments/legacy/benchmarks/judge_scores.md). Agent answers: [custom](experiments/legacy/benchmarks/agent_answers.json), [naive + LI](experiments/legacy/benchmarks/agent_naive_listock.json), [LI-maxed](experiments/legacy/benchmarks/agent_limaxed.json). Retrieval: [auto-generated](experiments/legacy/benchmarks/retrieval_auto_generated.json), [calibration](experiments/legacy/benchmarks/retrieval_calibration.json).

### Retrieval Ablation Study (39+ experiments)

Systematic 5-day study: parameter sweep, diagnosis, new retrieval tracks, orchestration improvements. Full results: [ablation_study.md](docs/progress/ablation_study.md), artifacts: [experiments/](experiments/).

**Progress**: R@5 0.833 → **0.900** (+8%), factual 0.842 → **0.858**, 8 formal experiment runs (RUN-001 through RUN-008).

| Phase | Experiments | Key result |
|-------|:-:|-----------|
| **Parameter sweep** | 24 | no-prefix +5.8% R@5, dense 20→40 +3.4%, ColBERT critical (−10% without) |
| **Diagnosis** | — | CE sigmoid bug fixed, stage attribution (semantic gap > truncation) |
| **New tracks** | 10 | Sparse lexicon norm +0.009 R@1. PRF, HyDE, normalize-all rejected |
| **Orchestration** | 3 traces | MMR merge, CE re-sort, adaptive filter, planner language fix |
| **Validation** (RUN-001) | 1 | Full pipeline R@5 = Retrieval-only (0.900), **context quality better** (judge 6:1:8) |
| **Post-protocol** (RUN-004–008) | 5 | Channel dedup 2→3 (adopted), cosine recall guard (adopted), dual scoring (rejected), negative intent fix |

**Answer comparison** (15 queries, both contexts → Qwen3.5, Claude judge): full pipeline produced better LLM answers in 6/15, worse in 1/15, tie in 8/15. Subqueries find additional relevant documents; CE re-sort puts best docs first.

**Experiment protocol** developed after two config-drift bugs wasted compute: spec-before-run, config parity checks, structured artifacts, early checkpoints. [Protocol](experiments/PROTOCOL.md).

### What didn't work (with evidence)

All rejected with measured evidence. Details in [experiment log](docs/progress/experiment_log.md#протестировано-и-отклонено-с-evidence).

| Technique | Result | Why rejected |
|-----------|--------|--------------|
| Cosine MMR | recall 0.70 → 0.11 | Re-promotes attractor documents |
| Dense re-score after RRF | recall 0.33 → 0.15 | Erases BM25 contribution |
| PCA whitening 1024→512 | recall 0.70 → 0.56 | Too aggressive dimensionality cut |
| DBSF fusion | 0.72 vs RRF 0.73 | RRF slightly better (ablation phase 1) |
| Dense+BM25 normalize-all | R@5 −4.2% | Synonyms in dense query dilute embeddings (ablation R1) |
| BM25 PRF expansion | R@5 −4.2% | Top terms from initial hits = noise (ablation R5) |
| HyDE pseudo-document | No improvement | Hypothetical doc doesn't help on hard subset (ablation D2) |
| Funnel expansion (RRF/ColBERT limits) | 0% change | Problem is semantic gap, not truncation (ablation A1-A3) |
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

```mermaid
graph LR
    Q["User Query"] --> QP["Query Planner<br/>3-5 subqueries"]
    QP --> OQ["+ Original query<br/>injection"]

    subgraph PER_Q["Per subquery"]
        BM25["BM25 top-100<br/>(lexicon norm)"]
        DENSE["Dense top-40<br/>(pplx-embed)"]
        BM25 & DENSE --> RRF["Weighted RRF<br/>3:1 BM25 weight"]
        RRF --> CB["ColBERT MaxSim<br/>rerank top-20"]
    end

    OQ --> PER_Q
    CB --> MMR["MMR merge<br/>λ=0.7"]
    MMR --> CE["CE re-sort +<br/>adaptive filter"]
    CE --> COS["Cosine recall<br/>guard"]
    COS --> DD["Channel dedup<br/>max 3/ch"]
    DD --> CTX["compose_context<br/>4000 tokens"]
```

- **Multi-query**: LLM generates 3-5 sub-queries, each runs independent hybrid retrieval, **MMR merge** (balances relevance + diversity)
- **Original query injection**: user query always in subqueries for BM25 keyword match
- **Sparse normalization** (R2): BM25 query normalized via lexicon (slang/aliases), dense query stays raw
- **LANCER nugget coverage**: query_plan subqueries as nuggets, targeted refinement on uncovered
- **CE re-sort + adaptive filter + cosine recall guard**: cross-encoder re-orders by relevance, gap detection (>2.0) + top-K guarantee (min 5). Cosine recall guard saves docs killed by CE but ranked high by bi-encoder — CE struggles with digest/bullet-list format ([sentence-transformers#2874](https://github.com/UKPLab/sentence-transformers/issues/2874)), cosine catches semantic relevance CE misses

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

### Agent ReAct Flow

```mermaid
graph TD
    START["User Query"] --> PHASE1

    subgraph PHASE1["Phase 1: Pre-Search"]
        direction LR
        P1_VIS["Visible: query_plan, search"]
        P1_ACT["LLM plans subqueries<br/>or picks analytics tool"]
    end

    PHASE1 -->|search / temporal| PHASE2
    PHASE1 -->|analytics tool| PHASE3
    PHASE1 -->|list_channels| PHASE4

    subgraph PHASE2["Phase 2: Post-Search"]
        direction LR
        P2_VIS["Visible: rerank, compose_context,<br/>final_answer, related_posts"]
        P2_ACT["LLM reranks, composes context,<br/>checks coverage (LANCER)"]
    end

    PHASE2 -->|coverage < 0.75| REF["Targeted refinement<br/>(max 1 round)"]
    REF --> PHASE2
    PHASE2 -->|coverage OK| FA

    subgraph PHASE3["Analytics Complete"]
        direction LR
        P3_VIS["Visible: final_answer only"]
        P3_ACT["Skip forced search"]
    end
    PHASE3 --> FA

    subgraph PHASE4["Navigation Complete"]
        direction LR
        P4_VIS["Visible: final_answer only"]
        P4_ACT["Skip forced search"]
    end
    PHASE4 --> FA

    FA["final_answer + verify"] --> SSE["SSE Stream<br/>citations + answer"]
```

---

## Observability

Self-hosted Langfuse v3. Every agent request produces a trace tree with per-span timing, token usage, and tool outputs.

<!-- TODO: screenshot of Langfuse trace tree with real query (replace ASCII below) -->
<!-- Langfuse UI: http://localhost:3100, pick a retrieval_evidence trace like q01 or q02 -->

```
agent_request (root)                          30.5s total
├── llm_step_1 → llm_chat_completion         1.7s  (1144 in → 38 out tokens)
├── tool:query_plan                           2.7s
├── llm_step_2 → llm_chat_completion         1.4s
├── tool:search → hybrid_retrieval            2.5s  (30 docs, 3 subqueries)
├── llm_step_3 → llm_chat_completion         11.0s
├── tool:rerank → CE filter                   2.1s  (scores: 9.4, 9.3, 9.1...)
├── tool:compose_context                      0.002s (10 citations, coverage 0.79)
├── llm_step_4_final → llm_chat_completion    8.0s  (7453 in → 305 out tokens)
├── tool:final_answer
└── tool[system]:verify → hybrid_retrieval    0.4s
```

Rich output per span: hits_count, coverage, prompt_len, token usage. Error marking for failed tools. Root trace: plan, strategy, tokens, coverage, citations_count.

---

## Development Workflow

Built with AI coding agents (Claude Code, Codex) following a structured process:

```
Research (29 reports) → Specification (35 specs) → Implementation → Evaluation → Documentation
```

- **Research-driven**: every decision backed by deep research ([R00-R27](docs/research/reports/), [46 prompts](docs/research/prompts/))
- **Evaluation-first**: 57+ eval runs + 39 ablation experiments, every change measured against [golden dataset](datasets/eval_golden_v2.json) ([experiment log](docs/progress/experiment_log.md), [ablation study](docs/progress/ablation_study.md))
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
experiments/            Experiment protocol, baseline config, run artifacts (spec + data + results)
benchmarks/             Framework comparison (LlamaIndex vs custom, 4 pipelines)
datasets/               Golden dataset (36 Qs), calibration (100 Qs), prompts, entity dictionary
deploy/                 Docker compose (dev, langfuse, test, benchmark)
```

## Security Considerations

15 LLM tools with dynamic visibility = attack surface. Current mitigations: `SecurityManager` sanitizes all external input before tool execution, `refusal_policy` with deterministic trim blocks out-of-scope queries, tool visibility is phase-gated (agent cannot call `final_answer` before search, cannot call `search` after analytics short-circuit). Rate limiting middleware with per-client tracking and exponential backoff. Formal adversarial eval (prompt injection, tool abuse scenarios) is planned for the expanded 200Q golden dataset — current refusal rate is 3/3 on known out-of-scope queries.

## What I'd do with more resources

**Fine-tuning the reranker** on domain-specific hard negatives. Qwen3-Reranker-0.6B struggles with digest/bullet-list format (CE=-5.8 on clearly relevant docs). LoRA fine-tuning on ~500 hard negative pairs from our eval logs — expected +3-7% on domain based on literature. ~1 hour on H100, ~$3 on cloud. The hard negatives already exist in our experiment artifacts.

**Production deployment**: failure mode handling (gpu_server crash → graceful degradation to dense-only), cost model per query, multi-user rate limiting beyond current middleware. Architecture is ready — FastAPI + Docker + Qdrant all containerized, GPU processes are the only native components.

**Scaling to 1M+ docs**: add news wire collections alongside Telegram. HNSW is sublinear — retrieval latency stays flat. Main work: incremental reindex pipeline (already implemented for Telegram), Qdrant sharding config, stale document detection.

## License & Usage

Code: [Apache 2.0](LICENSE) — free to use, modify, and learn from.

This is a research/portfolio project, not a product. The methodology, experiment logs, ablation results, and decision rationale in `docs/` and `experiments/` are the main deliverable — the code is an implementation of that methodology. If you're building your own RAG system, the writeups are probably more valuable than copying the code: your domain and data will require different choices, and the point of this project is showing how to make those choices systematically.

Questions, discussions, and methodological critiques welcome via [Issues](../../issues).
