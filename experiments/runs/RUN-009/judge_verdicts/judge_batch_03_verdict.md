# RUN-009 Judge Verdict — Batch 03

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_03.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id   | factual | useful | evidence_support | retrieval_sufficiency | correct_refusal | short_reason                                                                                                        |
| ---------- | ------: | -----: | ---------------: | --------------------: | --------------: | ------------------------------------------------------------------------------------------------------------------- |
| golden_q21 |       — |    1.8 |                — |                     — |               1 | Корректный out-of-range refusal; только лишнее предупреждение про неточность.                                       |
| golden_q22 |     1.0 |    1.9 |                — |                     — |               — | Правильно использован `entity_tracker`; ответ соответствует real 3-month timeline, а reference выглядит устаревшим. |
| golden_q23 |     0.9 |    1.8 |                — |                     — |               — | Топ arXiv papers передан верно; не проговорена sparsity или число уникальных papers.                                |
| golden_q24 |     0.9 |    1.8 |                — |                     — |               — | Назван релевантный лидер topic-mode; `expected answer` уже не выглядит лучшим ориентиром.                           |
| golden_q25 |     0.4 |    0.7 |              0.8 |                   0.6 |               — | Уходит в SGR vs Tools, но не покрывает `boris_again` / `any2json` и ломает формат raw JSON + tool markup.          |
| golden_q26 |     0.9 |    1.9 |                — |                     — |               — | Топ компаний передан хорошо по `entity_tracker`; часть expected-list не проговорена явно.                           |
| golden_q27 |     0.8 |    1.8 |                — |                     — |               — | Core co-occurrence с NVIDIA верный, но объяснения причин и хвост списка выглядят более вольными, чем tool preview. |
| golden_q28 |     0.9 |    1.8 |                — |                     — |               — | Сравнение OpenAI vs DeepSeek верно по counts и peaks; не раскрыт явный контекст V3 для пика DeepSeek.               |
| golden_q29 |     0.9 |    1.8 |                — |                     — |               — | Топ papers назван корректно; не проговорена sparsity корпуса как отдельный вывод.                                   |
| golden_q30 |     0.9 |    1.8 |                — |                     — |               — | Верно указан канал `gonzo_ml` и дан полезный контекст двух упоминаний.                                               |

## Flagged For Manual Review

- `golden_q22` — хороший случай, где answer модели выглядит лучше weak reference: вопрос про последние 3 месяца, tool output дал 470 упоминаний за 14 недель, а reference или required claims тянут старый long-range baseline про `~1600`. Я бы не штрафовал.
- `golden_q24` — похожий кейс: `expected answer` указывает на `techsparks`, но analytics tool вывел `xor_journal` как top authority по теме робототехники или роботакси; ответ модели здесь выглядит допустимо и, возможно, лучше reference.
- `golden_q25` — слабый ответ: не покрыт `boris_again`, пропущены `any2json` и `Technical AI Safety`, плюс сильный format break с raw JSON или tool markup. Retrieval по второй половине вопроса тоже выглядит неполным.

## Summary

Самые сильные ответы здесь: `golden_q22`, `golden_q26`, а также ровные analytics-кейсы `golden_q23`, `golden_q24`, `golden_q28`, `golden_q29`, `golden_q30`.

Самые слабые: `golden_q25`, затем `golden_q27`.

Повторяющиеся паттерны ошибок в батче: weak или stale `expected answers`, тематически близкий, но мимо `required claims` ответ, и format leakage через raw JSON или tool markup.
