# rag_app — Production RAG with Agentic ReAct Pipeline

> Self-hosted RAG system over 36 Russian-language AI/ML Telegram channels.
> No managed APIs. No frameworks. Custom retrieval pipeline on local hardware.

**Factual: 0.84** | **Faithfulness: 0.91** | **Robustness: 0.954** | **Recall@3: 0.97** | **15 tools** | **36 channels, 200K+ docs**

---

## What it does

User asks a question about AI/ML news. ReAct agent plans sub-queries, runs hybrid retrieval (BM25 + dense + ColBERT) over 200K+ documents, filters with cross-encoder, produces a grounded answer with citations via SSE streaming.

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

Claude judge (0.0-1.0 granular scale) + independent NLI faithfulness verification via ruBERT.

| Metric | Value | Details |
|--------|-------|---------|
| **Factual correctness** | **0.842** | 36 Qs, Claude judge with 7 calibration examples |
| **Usefulness** | **1.778 / 2** | 36 Qs |
| **Key Tool Accuracy** | **1.000** | 36/36 correct tool selection |
| **Faithfulness** | **0.91** | 17 retrieval Qs, 171 claims verified, **0 hallucinations** |
| **Retrieval recall@3** | **0.97** | 100 hand-crafted calibration queries |
| **Mean latency** | **24.4s** | Full pipeline including LLM inference |

### Robustness (Cao et al. 2025 adapted)

Bypass pipeline: direct Qdrant + LLM, controlled k and ordering. Claude judge on 151 answers.

| Metric | Value | What it shows |
|--------|-------|---------------|
| **NDR** | **0.963** (26/27) | Retrieval helps in 96% of cases |
| **RSR** | **0.941** (16/17) | Quality monotonically increases with more docs |
| **ROR** | **0.959** | Document order has no effect on answers |
| **Composite** | **0.954** | |

Retrieval adds +0.53 absolute factual improvement (k=0: 0.10, k=20: 0.63). Full [robustness analysis](docs/planning/robustness_experiments.md).

57 eval runs across development. Full [experiment history](docs/planning/experiment_history.md) with per-question analysis. [NLI faithfulness analysis](results/reports/nli_faithfulness_analysis_20260401.md). [Retrieval playbook](docs/planning/retrieval_improvement_playbook.md).

### Custom vs LlamaIndex Benchmark

Built the same pipeline in LlamaIndex (best-effort) and measured against our custom implementation.
4 pipelines, same LLM, same data, same questions. [Full spec](docs/specifications/active/SPEC-RAG-29-framework-comparison-benchmark.md).

**Agent E2E** (17 questions, judge: Claude Opus 4.6):

| Pipeline | Factual | Usefulness | Grounding | Latency |
|----------|:---:|:---:|:---:|:---:|
| Naive (dense + LLM) | 0.55 | 1.04 | 0.28 | ~4s |
| LlamaIndex stock | 0.51 | 1.13 | 0.46 | ~9s |
| LlamaIndex maxed (weighted RRF + CE) | 0.54 | 1.21 | 0.48 | ~11s |
| **Custom pipeline** | **0.84** | **1.77** | **0.88** | **~30s** |

Custom wins by **+0.30 factual**, **+0.56 usefulness**, **+0.40 grounding** vs best framework config.

**Retrieval-only** (100 hand-crafted queries):

| Pipeline | Recall@1 | Recall@5 | MRR | Latency |
|----------|:---:|:---:|:---:|:---:|
| Naive (dense only) | 0.730 | 0.940 | 0.825 | 0.1s |
| LlamaIndex stock | 0.730 | 0.940 | 0.825 | 0.1s |
| LlamaIndex maxed | 0.780 | 0.980 | 0.865 | 1.4s |
| **Custom (RRF + ColBERT)** | **0.780** | **0.970** | **0.866** | **0.2s** |

Key findings:
- **LlamaIndex stock = naive**: default hybrid fusion adds zero gain
- **Reranker is not the differentiator**: li_maxed ≈ li_stock on agent E2E (+0.03 factual)
- **Multi-query planning + LANCER + specialized tools = main gain source** (not retrieval tuning)
- **Grounding 0.88 vs 0.48**: inline `[1][2][3]` citations via `compose_context` → `final_answer`
- **Custom 7x faster** than LlamaIndex maxed on retrieval (direct HTTP vs framework abstraction)
- LlamaIndex adds ~70 transitive dependencies vs our ~12

Full per-question breakdown in [judge_scores.md](benchmarks/results/judge_scores.md).

### What didn't work (with evidence)

| Technique | Result | Why rejected |
|-----------|--------|--------------|
| Cosine MMR | recall 0.70 -> 0.11 | Re-promotes attractor documents |
| Dense re-score after RRF | recall 0.33 -> 0.15 | Erases BM25 contribution |
| PCA whitening 1024->512 | recall 0.70 -> 0.56 | Too aggressive dimensionality cut |
| DBSF fusion | 0.72 vs RRF 0.73 | RRF slightly better |
| CE reranking after ColBERT | r@3: 0.97 -> 0.94 | Degrades top-3, replaced with filter |
| Pipeline v2 (RRF->CE->ColBERT) | +0.02 r@2 only | Not worth complexity |
| XLM-RoBERTa for Russian NLI | ent=0.006 on obvious pairs | ruBERT 150x better on Russian |

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
| **Data** | 36 Telegram channels, 200K+ docs | Jul 2025 - Mar 2026 | Qdrant |

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
Research (28 reports) -> Specification (21 specs) -> Implementation -> Evaluation -> Documentation
```

- **Research-driven**: every architectural decision backed by deep research with full project context
- **Evaluation-first**: 57 eval runs, every change measured against golden dataset
- **Specs before code**: concrete acceptance criteria, reviewed by Codex before implementation
- **Architecture docs**: mirror current codebase, not aspirational
- **Decision log**: [45 ADR entries](docs/architecture/11-decisions/decision-log.md) documenting every choice and why

---

## Quick Start

Requires: V100 (or compatible GPU for LLM), RTX 5060 Ti (or 8GB+ GPU for embedding/reranker), Docker Desktop.

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
  research/             34 prompts + 28 reports
  specifications/       21 specs (active + completed)
  planning/             Roadmap, playbook, experiment history
benchmarks/             Framework comparison (LlamaIndex vs custom, 4 pipelines)
datasets/               Golden dataset (36 Qs), calibration (100 Qs), prompts, entity dictionary
deploy/                 Docker compose (dev, langfuse, test, benchmark)
```

## License

Private project. Not open-sourced.
