# Framework Comparison Benchmark Artifacts

SPEC-RAG-29: Custom pipeline vs LlamaIndex (2026-04-03). Spec: `docs/specifications/completed/SPEC-RAG-29-framework-comparison-benchmark.md`.

4 pipelines (naive, LI stock, LI maxed, custom), same LLM, same data.
Agent E2E: 17 questions, Claude Opus 4.6 judge.
Retrieval-only: 100 auto-generated + 100 calibration queries.

**Results**: Custom wins +0.30 factual, +0.56 usefulness, +0.40 grounding vs best framework.

**Files**: agent answers (4 pipeline JSONs), judge scores, retrieval results, Langfuse traces.
**Total**: 13 files, all committed.
