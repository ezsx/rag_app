# RUN-009 Judge Verdict — Batch 06

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_06.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id       | factual | useful | evidence_support | retrieval_sufficiency | short_reason                                                                                                         |
| -------------- | ------: | -----: | ---------------: | --------------------: | -------------------------------------------------------------------------------------------------------------------- |
| golden_v3_q053 |     1.0 |    1.9 |              0.9 |                   1.0 | Required claims закрыты; extra capabilities частично опираются на adjacent doc.                                     |
| golden_v3_q054 |     1.0 |    1.9 |              1.0 |                   1.0 | Все три required claim по релизу PyTorch 2.10 покрыты прямо и чисто.                                                |
| golden_v3_q056 |     1.0 |    1.8 |              1.0 |                   1.0 | Llama3.1 8B, WildChat-1M и симуляция эмоций или ошибок пользователя переданы точно.                                 |
| golden_v3_q057 |     1.0 |    1.9 |              1.0 |                   1.0 | ЦА курса, темы и `n8n` названы верно; ответ полезно расширен.                                                        |
| golden_v3_q058 |     0.9 |    1.8 |              1.0 |                   1.0 | Core facts про Wan-Move верны; есть лёгкая путаница Wan-Move vs Wan-I2V-14B.                                        |
| golden_v3_q060 |     1.0 |    1.9 |              0.9 |                   1.0 | `302` вариантов, `16` жизнеспособных и first-of-its-kind claim закрыты.                                             |
| golden_v3_q061 |     0.9 |    1.8 |              1.0 |                   1.0 | База, итеративность и Over-turn masking переданы; есть minor numeric slip про `30` vs `32` шага.                   |
| golden_v3_q062 |     0.6 |    1.2 |              0.6 |                   1.0 | Понята общая идея про 5 роботов и прототипы, но список спорный: часть моделей не выглядит “доступной прямо сейчас”. |
| golden_v3_q063 |     1.0 |    1.9 |              1.0 |                   1.0 | `20` запусков, отсутствие memory access и рекомендация GPT 5 Thinking или Pro покрыты полностью.                   |
| golden_v3_q064 |     1.0 |    1.8 |              0.8 |                   1.0 | Reward-free learning, scaling data или model и риск “сломать симуляцию” переданы верно; citation trail небрежный.  |

## Flagged For Manual Review

- `golden_v3_q061` — в целом сильный ответ, но есть конкретная числовая оговорка: вместо scaling до `32` шагов сказано про `30`. Небольшая ошибка, но она именно factual.
- `golden_v3_q062` — заметный factual drift: ответ правильно понял общий intent про “роботов, которых уже можно купить”, но в список попали как минимум спорные позиции, которые сам же ответ описывает как не вполне доступные прямо сейчас. Это уже не minor nuance.

## Summary

Самые сильные ответы здесь: `golden_v3_q054`, `golden_v3_q057`, `golden_v3_q063`, а также `golden_v3_q053`, `golden_v3_q056`, `golden_v3_q060`.

Самые слабые: `golden_v3_q062`, затем `golden_v3_q061`.

Повторяющиеся паттерны в батче: локальные numeric slips и частично верный, но неточно интерпретированный список или claim при в целом хорошем retrieval.
