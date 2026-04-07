# Agent Evaluation Artifacts

57+ eval runs (2026-03-17..04-01). Full history: `docs/progress/experiment_log.md`.

Прогоны agent E2E через `scripts/evaluate_agent.py` (36 Qs golden dataset).
Claude judge scoring: factual correctness, usefulness, key tool accuracy.

**Key milestones**:
- Golden v1 (30 Qs): factual ~0.79, useful ~1.72
- Golden v2 baseline (36 Qs): factual ~0.80, useful ~1.53, KTA 1.000
- Final (36 Qs + NLI): factual 0.842, useful 1.778, faithfulness 0.91

NLI scores, claims files, calibration data также здесь.

**Total local files**: 88. **Committed sample**: eval_results_20260401-091242.json (final run).
