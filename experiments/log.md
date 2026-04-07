# Experiment Log

> Summary всех experiment runs. Одна строка на run. Детали — в runs/RUN-NNN/.
> Обновляется после каждого завершённого run (adopt или reject).

| Run | Date | Hypothesis | Key metric | Baseline | Result | Decision |
|-----|------|-----------|------------|----------|--------|----------|
| — | 2026-04-04..07 | Ablation phase 1-3 (pre-protocol) | R@5 | 0.833 | 0.900 | adopted → baseline |
| — | 2026-04-07 | CE URL fix + prefix fix | R@5 FP | 0.783 | 0.883 | adopted → baseline |

> Runs выше проведены ДО введения протокола. Начиная с RUN-001 — по протоколу.

| RUN-001 | 2026-04-08 | Baseline: RO vs FP после prefix fix | R@5 RO=FP=0.900, judge FP 6:1:8 | — | FP quality > RO | **adopted** |
