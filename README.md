# rag_app

> **Work in progress.** Active development — architecture, retrieval pipeline, and evaluation are evolving rapidly.

Self-hosted RAG system with an agentic ReAct pipeline over a curated collection of 36 Russian-language Telegram channels about AI/ML.

No managed APIs. No frameworks. Custom retrieval pipeline built from scratch — hybrid search, ColBERT reranking, adaptive tool selection, all running on local hardware.

## What it does

User asks a question → ReAct agent plans sub-queries → hybrid retrieval (BM25 + dense + ColBERT) over 13K documents → cross-encoder reranking → grounded answer with citations via SSE streaming.

```
Query → query_plan (LLM) → search (BM25+Dense → RRF → ColBERT) → rerank → compose_context → answer
```
http://localhost:8001/
## Architecture

Full architecture document: [Architecture_ru.md](docs/Architecture_ru.md) (auto-generated from modular sources in `docs/architecture/`).

```
┌─────────────────────────────────────────────────────────┐
│ Windows Host                                            │
│   llama-server.exe → V100 SXM2 32GB                     │
│   Qwen3-30B-A3B (Q4_K_M), port 8080                     │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────┐
│ WSL2 Native         │                                   │
│   gpu_server.py → RTX 5060 Ti 16GB                      │
│   Qwen3-Embedding-0.6B + bge-reranker-v2-m3             │
│   + jina-colbert-v2 (560M), port 8082                   │
└─────────────────────┼───────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────┐
│ Docker (CPU only)   │                                   │
│   FastAPI + Web UI  │  Qdrant (dense + sparse + ColBERT)│
│   port 8001         │  port 6333                        │
└─────────────────────┴───────────────────────────────────┘
```

**Why no Docker GPU?** V100 in TCC mode poisons NVML in WSL2 — nvidia-container-cli crashes for all GPUs. GPU workloads run natively via PyTorch in WSL2. See [decision log](docs/architecture/11-decisions/decision-log.md).

## Key components

| Component | What | Where |
|-----------|------|-------|
| **LLM** | Qwen3-30B-A3B MoE (3B active) | V100 via llama-server |
| **Embedding** | Qwen3-Embedding-0.6B (1024-dim) | RTX 5060 Ti via gpu_server.py |
| **Reranker** | bge-reranker-v2-m3 (cross-encoder) | RTX 5060 Ti via gpu_server.py |
| **ColBERT** | jina-colbert-v2 (128-dim per-token MaxSim) | RTX 5060 Ti via gpu_server.py |
| **Vector store** | Qdrant — 3 named vectors: dense, sparse (BM25), ColBERT | Docker |
| **Fusion** | Weighted RRF (BM25 weight=3, dense weight=1) | — |
| **Agent** | ReAct loop with native function calling, 13 LLM tools, dynamic visibility | — |
| **Analytics** | Entity tracker + Arxiv tracker via Qdrant Facet API | — |
| **Data** | 36 Telegram channels, 13K+ documents, Jul 2025 — Mar 2026 | — |

## Retrieval pipeline

```
BM25 top-100 ──┐
               ├── Weighted RRF (3:1) ── ColBERT rerank ── Cross-encoder rerank ── Channel dedup
Dense top-20 ──┘
```

Multi-query search: LLM generates sub-queries, all executed via round-robin merge. Original user query always included for BM25 keyword match.

## Current metrics

| Eval | Metric | Value | Questions |
|------|--------|-------|-----------|
| Golden v1 (post SPEC-RAG-15) | **Factual correctness** | **1.79/2** | 30 |
| Golden v1 (post SPEC-RAG-15) | **Usefulness** | **1.72/2** | 30 |
| Golden v1 (post SPEC-RAG-15) | Key tool accuracy | 0.926 | 30 |
| Agent v1 (legacy) | Recall@5 | 0.76 | 10 |
| Retrieval-only (direct Qdrant) | Recall@5 | 0.73 | 100 |

Manual judge (Claude Opus 4.6 + Codex GPT-5.4) is the primary metric — strict recall@5 is unreliable for analytics/temporal queries. Pipeline optimization history: 24 experiments across 5 eval datasets. Details in [playbook](docs/planning/retrieval_improvement_playbook.md).

## What didn't work (with evidence)

| Technique | Result | Why |
|-----------|--------|-----|
| Cosine MMR | recall 0.70 → 0.11 | Re-promotes attractor documents |
| Dense re-score after RRF | recall 0.33 → 0.15 | Erases BM25 contribution |
| PCA whitening 1024→512 | recall 0.70 → 0.56 | Too aggressive dimensionality cut |
| Whitening 1024→1024 | parity | Dense isn't bottleneck at BM25 3:1 |
| DBSF fusion | 0.72 vs RRF 0.73 | RRF slightly better |

## Development workflow

This project is built with AI coding agents (Claude Code, Codex) following a structured engineering process — not prompt-and-pray vibe coding.

```
Research → Specification → Implementation → Evaluation → Documentation
```

**Research phase.** Every non-trivial decision starts with a Deep Research prompt that includes full project context — hardware constraints, current metrics, what was already tried. Produces numbered reports (R01–R18 so far) that become permanent reference. No throwaway ChatGPT threads.

**Specification phase.** Research findings are distilled into specs with concrete acceptance criteria (recall targets, latency budgets, specific test queries). Specs live in `docs/specifications/active/` and move to `completed/` after implementation.

**Evaluation-driven development.** Every pipeline change is measured against golden datasets before committing. 22 experiments tracked in the [playbook](docs/planning/retrieval_improvement_playbook.md) with per-question analysis. Single-query verification before full eval runs (7 min each).

**Documentation as source of truth.** `docs/architecture/` mirrors the current codebase — not aspirational, not historical. Agent context files (`CLAUDE.md`, `AGENTS.md`, `agent_context/`) enforce consistency across sessions. Governance rules in [02-documentation-governance.md](docs/architecture/00-meta/02-documentation-governance.md) prevent doc sprawl.

**What this avoids:** orphaned docs nobody reads, "works on my machine" knowledge, regression from agents that don't know project history, accumulating technical debt from undocumented decisions.

## Current: 13 Tools with Dynamic Visibility + Analytics (Phase 3.4)

Agent has 13 tools with phase-based dynamic visibility (max 5 visible at a time):
- **Search**: `search`, `temporal_search`, `channel_search`, `cross_channel_compare`, `summarize_channel`
- **Analytics**: `entity_tracker` (top/timeline/compare/co-occurrence), `arxiv_tracker` (top/lookup)
- **Planning**: `query_plan`, `list_channels`
- **Enrichment**: `rerank`, `related_posts`, `compose_context`
- **Synthesis**: `final_answer`

Data-driven routing: tool keywords + agent policies loaded from `datasets/tool_keywords.json`. Enriched payload (SPEC-RAG-12): 16 indexed fields including NER entities (91 AI/ML entities), arxiv IDs, year_week. Collection `news_colbert_v2` with 13K+ points.

## Project structure

```
src/                        Application code
  adapters/                   Qdrant, LLM, TEI, hybrid retriever
  api/                        FastAPI endpoints + SSE streaming
  services/                   Agent service, tools, query planner, reranker
  core/                       Settings, DI, auth
  schemas/                    Pydantic models
scripts/                    GPU server, evaluation, ingestion
docs/
  architecture/               Source of truth — current system state
  research/                   20 research prompts + 18 reports
  specifications/             Implementation specs (active + completed)
  planning/                   Roadmap, playbook, implementation plans
datasets/                   Golden dataset (30 Qs), entity dictionary, tool routing config
deploy/                     Docker compose files
```

## License

Private project. Not open-sourced.
