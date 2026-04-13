# RUN-009 Judge Verdict — Batch 04

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_04.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id       | factual | useful | evidence_support | retrieval_sufficiency | correct_refusal | short_reason                                                                                               |
| -------------- | ------: | -----: | ---------------: | --------------------: | --------------: | ---------------------------------------------------------------------------------------------------------- |
| golden_v2_q31  |     1.0 |    1.9 |                — |                     — |               — | `hot_topics` использован правильно; названы несколько реальных тем недели.                                 |
| golden_v2_q32  |     0.9 |    1.8 |                — |                     — |               — | Реальные weekly topics перечислены, но часть формулировок шире и менее точно привязана к digest.         |
| golden_v2_q33  |     0.9 |    1.5 |                — |                     — |               — | Month aggregation по `hot_topics` в целом верна, но ответ содержит утечку `</think>` и слегка шумный хвост. |
| golden_v2_q34  |     1.0 |    1.9 |                — |                     — |               — | Корректный topic-mode ranking по NLP с несколькими релевантными каналами.                                  |
| golden_v2_q35  |     0.9 |    1.5 |                — |                     — |               — | Профиль `gonzo_ml` описан по tool output, но финал сломан утечкой `</think>` и голым `[1]`.                |
| golden_v2_q36  |     1.0 |    1.9 |                — |                     — |               — | Хороший topic-mode ответ по робототехнике или роботакси с релевантными каналами.                           |
| golden_v3_q038 |     1.0 |    1.9 |              0.9 |                   1.0 |               — | Все required claims про Atlas покрыты; есть полезный обзор ограничений и фич.                              |
| golden_v3_q039 |     1.0 |    2.0 |              1.0 |                   1.0 |               — | Точное сравнение AI Trader vs шизотрейдинг, required claims закрыты полностью.                             |
| golden_v3_q040 |     1.0 |    1.9 |              1.0 |                   1.0 |               — | Метрики `13 игр / 68% / <15%` переданы чисто и прямо по документу.                                         |
| golden_v3_q041 |     0.6 |    1.4 |              0.9 |                   1.0 |               — | Главная мысль верна, но пропущены `$1T crypto drawdown` и две из четырёх рыночных метрик.                 |

## Flagged For Manual Review

- `golden_v2_q33` — content в целом полезный, но в финальный ответ утек `</think>`; это заметно бьёт по usability, хотя core facts в порядке.
- `golden_v2_q35` — content в целом полезный, но в финальный ответ утек `</think>` и артефакт оформления `[1]`; это заметно бьёт по usability, хотя core facts в порядке.
- `golden_v3_q041` — частично верно, но пропущены важные `required claims`: падение crypto market на `~$1T` и две биржевые метрики. Это не просто minor omission.

## Summary

Самые сильные ответы здесь: `golden_v3_q039`, `golden_v3_q040`, `golden_v2_q31`, `golden_v2_q34`, `golden_v2_q36`, `golden_v3_q038`.

Самые слабые: `golden_v3_q041`, затем `golden_v2_q33` и `golden_v2_q35`.

Повторяющиеся паттерны ошибок в батче: format leakage через `</think>` и артефакты оформления, а также частично верный, но неполный ответ с пропуском критичных `required claims`.
