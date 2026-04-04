# Growth Ceiling Context — `rag_app`

> Snapshot date: **2026-03-29**
> Purpose: attach this document to Deep Research so the investigator starts from a dense, evidence-backed project snapshot instead of reconstructing the repo from zero.

## Author Context

- Backend Python engineer, 24 y.o.
- Goal: transition into **Applied LLM Engineer** roles, target roughly **$2-3k/month**, remote.
- This repo is intentionally **overbuilt** as a portfolio flagship, not a toy pet project.
- Other background: production backend, shipped CV system in hospitals, built multi-service infra, strong Python systems base.

## Project Goal

`rag_app` is a **self-hosted FastAPI RAG platform** for searching and aggregating AI/ML news from Telegram channels.

High-level pipeline:

```text
Telegram channels
  -> ingest
  -> Qdrant (dense + sparse + ColBERT named vectors)
  -> hybrid retrieval
  -> ReAct agent with native function calling
  -> SSE answer
```

This is meant to demonstrate:

- retrieval engineering
- agent orchestration without frameworks
- self-hosted LLM infra
- production constraints and trade-off reasoning
- evidence-driven iteration

## Hardware and Infra Constraints

- **LLM**: Qwen3-30B-A3B GGUF via `llama-server.exe`
- **Host GPU**: V100 SXM2 32GB on Windows host
- **Embedding / reranker / ColBERT**: `gpu_server.py` in WSL2 on RTX 5060 Ti 16GB
- **Vector DB**: Qdrant in Docker
- **Constraint**: Docker GPU path is intentionally not used; mixed Windows + WSL + Docker setup is a real project constraint, not an accident
- **Principle**: self-hosted first, no managed inference dependency for the main system

## Current Implemented System

### Retrieval

Current retrieval pipeline:

```text
BM25 top-100 + dense top-20
  -> weighted RRF (BM25 3 : dense 1)
  -> ColBERT MaxSim rerank
  -> cross-encoder rerank
  -> channel dedup
```

Important implementation traits:

- original user query is always injected into retrieval
- multi-query search uses round-robin merge
- retrieval is strongly tuned around Russian Telegram AI/ML corpus behavior

### Agent

The ReAct loop is custom, not LangChain/LlamaIndex orchestration.

Implemented LLM-visible tools: **15**

1. `query_plan`
2. `search`
3. `temporal_search`
4. `channel_search`
5. `cross_channel_compare`
6. `summarize_channel`
7. `list_channels`
8. `rerank`
9. `related_posts`
10. `compose_context`
11. `final_answer`
12. `entity_tracker`
13. `arxiv_tracker`
14. `hot_topics`
15. `channel_expertise`

Important orchestration choices:

- dynamic tool visibility by phase
- max 5 visible tools at once
- keyword routing driven by `datasets/tool_keywords.json`
- forced search if the model skips tools
- bypasses for analytics/navigation paths
- SSE contract is part of the production surface

### Precomputed Analytics

Recent additions:

- `hot_topics`: weekly BERTopic-based digest/trend tool
- `channel_expertise`: per-channel profile / ranking tool

These rely on cron-like precomputation, not live retrieval only.

## Production Hardening Already Done

Recent hardening work includes:

- per-request isolation via `RequestContext` + `ContextVar`
- auth hardening
- rate limit fix
- CORS tightening
- visible-tool whitelist
- cooperative request deadline
- demo auth path

Interpretation:

- this repo is no longer just experimenting with retrieval quality
- it is also trying to look like a production-aware service

## Evidence-Backed Strengths

### Evaluation

Latest strong canonical eval artifact currently available:

- `results/reports/eval_judge_20260325_spec15.md`

Key values from that report:

- golden dataset: **30 questions**
- manual judge: **Claude Opus 4.6 + Codex GPT-5.4**
- factual correctness: **1.79 / 2**
- usefulness: **1.72 / 2**
- key tool accuracy: **0.926**
- strict Recall@5: **0.342**

Interpretation:

- the system is not just “it seems good”
- there is already a real eval narrative with human review
- but the canonical report is still pre-`SPEC-RAG-16/17`

### Research Depth

Research archive has grown beyond early experimentation:

- prompts exist through `26-*`
- reports exist through `R21-*`

Notable late-stage research directions already covered:

- NLI citation faithfulness
- retrieval robustness
- RAG necessity classifier
- domain-specific tools
- evaluation methodology

This means the repo already contains more design reasoning than a typical demo project.

## Evidence-Backed Weaknesses / Trust Gaps

These are the most important weaknesses for portfolio credibility.

### 1. Canonical Eval Is Behind the Current System

The strongest published eval artifact is still:

- `results/reports/eval_judge_20260325_spec15.md`

That means the repo does **not yet have a fresh canonical measurement** after the recent:

- BERTopic analytics tools
- production hardening changes
- auth / timeout / request-isolation changes

### 2. Tests Are Stale and Cannot Currently Serve as Proof

Container-based pytest evidence collected during independent review:

- `108 collected, 1 error` with `RAG_ENV=dev`
- the immediate collection error comes from `src/tests/test_new_tools.py`
- ignoring that stale file produced:
  - `46 failed`
  - `61 passed`
  - `1 skipped`

Concrete stale examples:

- `src/tests/test_new_tools.py` imports removed modules like `math_eval` and `time_now`
- `src/tests/test_agent_service.py` still references pre-`RequestContext` internals and outdated tool counts
- endpoint tests still expect old auth behavior

Interpretation:

- the system may be improving faster than its proof layer
- current tests do not yet certify the present codebase

### 3. Docs / Packaging Drift

Repo documentation is partially inconsistent with the current state.

Examples:

- `README.md` still says `Work in progress`
- `README.md` contains inconsistent experiment/research counts
- `docs/progress/project_scope.md` still mixes old and new states
- some counts mention older research/tool totals even after recent expansions

Interpretation:

- implementation quality is currently ahead of portfolio packaging quality

### 4. Data Freshness / Coverage Gaps

Known operational gaps:

- corpus freshness is behind current date
- `hot_topics` currently has partial digest coverage rather than full historical weekly coverage
- BERTopic labels still need cleanup / humanization

These do not negate the architecture, but they weaken demo reliability.

## What This Research Should Actually Decide

The central question is:

> What would it take to make `rag_app` a genuinely **production-grade, complete** system — not a demo, not a proof-of-concept, but something an engineer would be proud to hand over as a finished product?

This is NOT about:
- “is this enough for an interview?” (cosmetics and packaging are trivially solvable with AI in half a day)
- “should we stop?” (we're not looking for permission to stop — we're looking for real gaps)

This IS about:
- **What practices, patterns, or capabilities do real production RAG/agent systems have that this one doesn't?**
- **What would a senior engineer reviewing this repo flag as missing, incomplete, or naive?**
- **How does this compare to actual production tools and systems on the market — not toy portfolio projects, but real products?**

## Specific Gaps To Investigate

The investigator should go beyond surface-level feature checklists and examine:

### vs Market tools
- How does this retrieval pipeline compare to what Perplexity, Glean, Danswer/Onyx, Langdock, or similar production RAG products actually do?
- What architectural patterns do they use that we don't?
- Where do they invest engineering effort that we skipped?

### vs Best practices
- Are there well-known RAG/agent production patterns we haven't applied?
- Observability, testing strategies, error recovery, graceful degradation?
- Evaluation methodology — are 30 questions with manual judge actually credible, or is this below the bar for serious work?
- Citation faithfulness — can we credibly claim the agent doesn't hallucinate without NLI or equivalent?

### vs State of the art (2025-2026)
- What has changed in RAG/agent architecture since the project started?
- Are there newer techniques (CRAG, Self-RAG, Adaptive-RAG, Graph RAG, late interaction models, etc.) that would meaningfully improve the system?
- Is the embedding model choice (Qwen3-Embedding-0.6B) still competitive, or has the field moved on?

## What Is NOT A Research Question

The following are trivially solvable and should NOT be investigated:

- UI cosmetics (half a day with AI)
- README/packaging/diagrams (half a day)
- Refactoring large files into modules (mechanical)
- Running fresh eval (just do it)
- Fixing stale tests (mechanical)
- Updating docs to match code (mechanical)

These are execution tasks, not research questions. The research should focus on **what we don't know we're missing**.

## Suggested Artifacts To Verify

The investigator should verify the context against the repo, not trust this file blindly.

Priority artifacts:

- `README.md`
- `docs/progress/project_scope.md`
- `docs/specifications/active/SPEC-RAG-16-hot-topics-channel-expertise.md`
- `docs/specifications/active/SPEC-RAG-17-production-hardening.md`
- `results/reports/eval_judge_20260325_spec15.md`
- `src/services/agent_service.py`
- `src/tests/`

## No Pre-Determined Conclusion

This document does NOT have a working hypothesis. The research should arrive at its own conclusion based on evidence. Possible outcomes include:

- “The system is genuinely production-grade, remaining work is mechanical”
- “There are 2-3 fundamental gaps that separate this from production quality”
- “The architecture is sound but the evaluation/proof story is weak”
- “There are industry patterns you completely missed”
- “You're solving the wrong problem — the real gap is X”

Any of these is a valid answer. The worst answer is a vague “it's good enough” or “here's 20 more features to add”.
