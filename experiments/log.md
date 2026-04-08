# Experiment Log

> Summary всех experiment runs. Одна строка на run. Детали — в runs/RUN-NNN/.
> Обновляется после каждого завершённого run (adopt или reject).

| Run | Date | Hypothesis | Key metric | Baseline | Result | Decision |
|-----|------|-----------|------------|----------|--------|----------|
| — | 2026-04-04..07 | Ablation phase 1-3 (pre-protocol) | R@5 | 0.833 | 0.900 | adopted → baseline |
| — | 2026-04-07 | CE URL fix + prefix fix | R@5 FP | 0.783 | 0.883 | adopted → baseline |

> Runs выше проведены ДО введения протокола. Начиная с RUN-001 — по протоколу.

| RUN-001 | 2026-04-08 | Baseline: RO vs FP после prefix fix | R@5 RO=FP=0.900, judge FP 6:1:8 | — | FP quality > RO | **adopted** |
| RUN-002 | 2026-04-08 | k=15 + floor=0.0 + dedup=3 | ce_neg 4.2→6.4, mean_ce 3.2→2.9 | RUN-001 | больше docs но хуже quality | **rejected** |
| RUN-003 | 2026-04-08 | MMR λ=0.85 (было 0.7) | MRR 0.638=0.638, 0 diffs | RUN-001 | CE re-sort доминирует, λ не влияет | **rejected** |
| RUN-004 | 2026-04-08 | compose_context 1800→4000 | factual 0.83 vs 0.842 | RUN-001 | в рамках погрешности, retrieval bottleneck не в budget | **rejected** |
| RUN-005 | 2026-04-08 | channel dedup 2→3 | q08: boris:3749 + techsparks:5439 вернулись | RUN-001 | dedup=2 убивал 3й doc из канала с ключевыми фактами | **adopted** |
| RUN-006 | 2026-04-08 | dual scoring (norm_linear + rrf_ranks) | q08 деградация: 4 cit vs 10 | RUN-005 | пересортировка ломает CE gap detection | **rejected** |
| RUN-007 | 2026-04-08 | cosine recall guard (CE precision + bi-encoder recall) | q08 10 cit, control stable | RUN-005 | CE фильтрует, cosine спасает false negatives | **adopted** |
