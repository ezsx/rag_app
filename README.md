# rag_app

Self-hosted RAG system with an agentic ReAct pipeline over 36 Russian-language Telegram channels about AI/ML. No managed APIs, no frameworks — custom retrieval pipeline on local hardware.

## What it does

User asks a question. ReAct agent plans sub-queries, runs hybrid retrieval (BM25 + dense + ColBERT) over 13K+ documents, reranks with cross-encoder, produces a grounded answer with citations via SSE streaming.

```
Query → query_plan (3-5 subqueries) → search (BM25+Dense → RRF → ColBERT) → rerank → compose_context → answer
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Windows Host                                                │
│   llama-server.exe → V100 SXM2 32GB (TCC)                  │
│   Qwen3.5-35B-A3B (Q4_K_M), port 8080                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│ WSL2 Native              │                                  │
│   gpu_server.py → RTX 5060 Ti 16GB                          │
│   Qwen3-Embedding-0.6B + bge-reranker-v2-m3                │
│   + jina-colbert-v2 (560M), port 8082                       │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│ Docker (CPU only)        │                                  │
│   FastAPI + Web UI (:8001)  Qdrant (:16333)                 │
│   Langfuse v3 (:3100) — observability                       │
└─────────────────────────────────────────────────────────────┘
```

V100 in TCC mode poisons NVML in WSL2 — Docker GPU unavailable. All GPU workloads run natively. See [decision log](docs/architecture/11-decisions/decision-log.md).

## Key components

| Component | What | Where |
|-----------|------|-------|
| **LLM** | Qwen3.5-35B-A3B MoE (3B active, Gated Delta Networks) | V100 via llama-server |
| **Embedding** | pplx-embed-v1-0.6B (1024-dim, mean pooling) | RTX 5060 Ti via gpu_server.py |
| **Reranker** | Qwen3-Reranker-0.6B-seq-cls (cross-encoder, CRAG-style filter) | RTX 5060 Ti via gpu_server.py |
| **ColBERT** | jina-colbert-v2 (128-dim per-token MaxSim) | RTX 5060 Ti via gpu_server.py |
| **NLI** | rubert-base-cased-nli-threeway (faithfulness verification) | RTX 5060 Ti via gpu_server.py |
| **Vector store** | Qdrant — 3 named vectors: dense, sparse (BM25), ColBERT | Docker |
| **Fusion** | Weighted RRF (BM25 weight=3, dense weight=1) | — |
| **Agent** | ReAct loop, native function calling, 15 tools, dynamic visibility | — |
| **Analytics** | Entity/arxiv tracking via Qdrant Facet API, BERTopic trends | — |
| **Observability** | Langfuse v3 — per-request traces with parent-child spans | Docker |
| **Data** | 36 Telegram channels, 13K+ documents, Jul 2025 — Mar 2026 | — |

## Retrieval pipeline

```
BM25 top-100 ──┐
               ├── Weighted RRF (3:1) ── ColBERT rerank ── Cross-encoder rerank ── Channel dedup (max 2/ch)
Dense top-20 ──┘
```

Multi-query: LLM generates 3-5 sub-queries via query planner, each runs independent hybrid retrieval, results merged via round-robin. Original user query always included for BM25 keyword match.

## Agent tools (15)

Phase-based dynamic visibility (max 5 visible at a time), data-driven keyword routing.

| Category | Tools |
|----------|-------|
| **Search** | `search`, `temporal_search`, `channel_search`, `cross_channel_compare`, `summarize_channel` |
| **Analytics** | `entity_tracker` (top/timeline/compare/co-occurrence), `arxiv_tracker` (top/lookup) |
| **Topics** | `hot_topics` (weekly/monthly trend digest), `channel_expertise` (per-channel profiles) |
| **Planning** | `query_plan`, `list_channels` |
| **Enrichment** | `rerank`, `related_posts`, `compose_context` |
| **Synthesis** | `final_answer` |

## Eval metrics

Claude judge (0.0-1.0 granular scale) + independent NLI faithfulness verification (ruBERT).

| Metric | Value | Scope |
|--------|-------|-------|
| **Factual correctness** | **0.842** | All 36 Qs |
| **Usefulness** | **1.778 / 2** | All 36 Qs |
| **Key Tool Accuracy** | **1.000** (36/36) | All 36 Qs |
| **Faithfulness (NLI)** | **0.91** (corrected), 0 hallucinations | 17 retrieval Qs |
| **Retrieval recall@3** | **0.97** (100 calibration queries) | Retrieval-only |
| **Mean latency** | 24.4s | All 36 Qs |
| **Eval dataset** | 36 questions, 4 eval modes (retrieval, analytics, navigation, refusal) | — |

57 eval runs, ~30 experiments tracked in the [playbook](docs/planning/retrieval_improvement_playbook.md). Full [experiment history](docs/planning/experiment_history.md) with per-question analysis. [NLI faithfulness analysis](results/reports/nli_faithfulness_analysis_20260401.md).

## Observability

Self-hosted Langfuse v3 (SPEC-RAG-19). Every agent request produces a trace tree:

```
agent_request (root)
├── llm_step_1 → llm_chat_completion
├── tool:query_plan → query_planner → llm_completion
├── llm_step_2 → llm_chat_completion
├── tool:search → hybrid_retrieval (per subquery)
├── tool:rerank → rerank (cross-encoder)
├── tool:compose_context
├── llm_step_N_final → llm_chat_completion
├── tool:final_answer
└── tool[system]:verify → hybrid_retrieval
```

Session grouping, tags, per-question eval trace naming (`agent_request_q01..q36`).

## What didn't work (with evidence)

| Technique | Result | Why |
|-----------|--------|-----|
| Cosine MMR | recall 0.70 → 0.11 | Re-promotes attractor documents |
| Dense re-score after RRF | recall 0.33 → 0.15 | Erases BM25 contribution |
| PCA whitening 1024→512 | recall 0.70 → 0.56 | Too aggressive dimensionality cut |
| DBSF fusion | 0.72 vs RRF 0.73 | RRF slightly better |
| CE reranking after ColBERT | r@3: 0.97 → 0.94 | Degrades top-3, replaced with filter |
| Pipeline v2 (RRF→CE→ColBERT) | +0.02 r@2 only | Not worth complexity |
| XLM-RoBERTa-large-xnli for NLI | ent=0.006 on obvious pairs | ruBERT 150x better on Russian |

## Development workflow

Built with AI coding agents (Claude Code, Codex) following a structured process.

```
Research (28 reports) → Specification (19 specs) → Implementation → Evaluation → Documentation
```

- **Research**: deep research prompts with full project context, numbered reports (R01-R28)
- **Specifications**: concrete acceptance criteria, move to `completed/` after implementation (21 specs)
- **Evaluation-driven**: every change measured against golden dataset before committing (57 eval runs)
- **Architecture docs**: mirror current codebase, not aspirational
- **Decision log**: 45 ADR entries documenting every architectural choice

## Quick start

```bash
# 1. LLM on V100 (PowerShell)
$env:CUDA_VISIBLE_DEVICES = "0"
llama-server.exe -m models/Qwen3.5-35B-A3B-Q4_K_M.gguf --jinja --reasoning-budget 0 --cache-type-k q8_0 --cache-type-v q8_0 -c 32768

# 2. Embedding + Reranker on RTX 5060 Ti (WSL2)
source /home/ezsx/infinity-env/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/gpu_server.py

# 3. Langfuse observability (Docker)
docker compose -f deploy/compose/compose.langfuse.yml up -d

# 4. API + Qdrant (Docker)
docker compose -f deploy/compose/compose.dev.yml up -d

# UI: http://localhost:8001
# Langfuse: http://localhost:3100 (admin@local.dev / Admin123!@#)
```

## Project structure

```
src/
  adapters/           Qdrant, LLM (llama-server), TEI, hybrid retriever
  api/                FastAPI endpoints + SSE streaming
  services/           Agent service, 15 tools, query planner, reranker
  core/               Settings, DI, observability (Langfuse)
  schemas/            Pydantic models
scripts/              GPU server, evaluation, ingestion, WSL networking helpers
docs/
  architecture/       Source of truth — current system state (41 decisions)
  research/           30 prompts + 28 reports
  specifications/     19 specs (active + completed)
  planning/           Roadmap, playbook
datasets/             Golden dataset (36 Qs), entity dictionary, tool routing config
deploy/               Docker compose (dev, langfuse, test)
```

## License

Private project. Not open-sourced.
