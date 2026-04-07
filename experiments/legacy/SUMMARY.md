# Legacy Experiment Artifacts

Артефакты экспериментов проведённых до введения experiment protocol (2026-04-08).
Полные данные хранятся локально, в git — summary файлы и representative samples.

| Категория | Прогонов | Период | Описание |
|-----------|:---:|--------|----------|
| **ablation/** | 39+ | 2026-04-04..08 | Retrieval ablation study: parameter sweep, diagnosis, new tracks, orchestration |
| **agent_eval/** | 57+ | 2026-03-17..04-01 | Agent E2E evaluation (36 Qs golden dataset, Claude judge) |
| **reports/** | 15+ | 2026-03-25..04-02 | Eval reports, NLI analysis, judge artifacts |
| **robustness/** | 5 | 2026-04-02 | NDR/RSR/ROR robustness testing (151 LLM calls, BERTScore vs Claude judge) |
| **judge_batches/** | 13 | 2026-03-25..30 | Claude + Codex judge verdicts per batch |
| **eval_\*/** | 8 | 2026-03-17..30 | Intermediate eval runs (smoke tests, evidence tests, stack iterations) |

Начиная с RUN-001 все эксперименты проводятся по `experiments/PROTOCOL.md`.
