# Ablation Study Artifacts

39+ экспериментов за 5 дней (2026-04-04..08). Полное описание: `docs/progress/ablation_study.md`.

## Phase 1 — Parameter Sweep (24 experiments)
24 JSON файла: baseline, no_prefix, dense_10/40, rrf_1_1..1_5, dbsf, 11 combos.
Winner: no-prefix + dense=40 + RRF [1:3]. R@5: 0.833 → 0.900.

## Phase 2 — Diagnosis + New Tracks (15 experiments)
`phase2/`: prod-parity eval, query-plan ablation, 10 new retrieval tracks.
Stage attribution, CE score distribution. R2 sparse normalization — лучший.

## Phase 3 — Orchestration
`phase3/`: full pipeline runs, MMR merge, CE re-sort, adaptive filter.
RUN-001 артефакты: `experiments/runs/RUN-001/`.

**Total local files**: 80. **Committed samples**: baseline.json, combo_noprefix_d40_rrf13.json (winner), _summary.json, stage_attribution.json.
