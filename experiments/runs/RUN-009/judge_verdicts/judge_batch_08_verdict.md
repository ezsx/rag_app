# RUN-009 Judge Verdict — Batch 08

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_08.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id       | factual | useful | evidence_support | retrieval_sufficiency | short_reason                                                                                            |
| -------------- | ------: | -----: | ---------------: | --------------------: | ------------------------------------------------------------------------------------------------------- |
| golden_v3_q076 |     1.0 |    1.9 |              1.0 |                   1.0 | Evolution AI Factory, SLA or 24x7 support и тарифы `35` / `70 ₽` покрыты полностью.                    |
| golden_v3_q077 |     0.2 |    0.6 |              0.8 |                   0.5 | Ответ ушёл в другой Karpathy-тезис и не покрывает i18n или fluent required claim.                      |
| golden_v3_q078 |     1.0 |    1.9 |              1.0 |                   1.0 | CSI `+50%`, bias `x10` вниз и паритет или превосходство над WeatherNext2 названы точно.                |
| golden_v3_q079 |     1.0 |    1.8 |              0.9 |                   1.0 | Три core risk claims про autonomy, resource misuse и limits of post-training alignment покрыты хорошо. |
| golden_v3_q080 |     1.0 |    1.8 |                — |                     — | Использован `hot_topics`; все три required темы и `465` постов названы корректно.                      |
| golden_v3_q081 |     1.0 |    1.8 |                — |                     — | Использован `hot_topics`; все три required темы недели и `447` постов переданы верно.                  |
| golden_v3_q082 |     0.8 |    1.6 |                — |                     — | `440` постов и 2 из 3 required topic buckets есть, но `that/on/is` как отдельный hot topic потерян.   |
| golden_v3_q083 |     1.0 |    1.8 |                — |                     — | Все три required topic buckets и `424` поста совпадают с `hot_topics` output.                          |
| golden_v3_q084 |     1.0 |    1.8 |                — |                     — | `416` постов и все три required bucket’а отражены корректно.                                            |
| golden_v3_q085 |     1.0 |    1.8 |                — |                     — | `409` постов и все три required hot topic bucket’а покрыты точно.                                       |

## Flagged For Manual Review

- `golden_v3_q077` — самый явный fail в батче: required claims про “исполнительные, но прямолинейные агенты” и пример с `i18n` или `fluent` не покрыты, вместо этого дан другой набор тезисов и другой code example. Похоже на retrieval или interpretation miss.
- `golden_v3_q082` — неплохой analytics-answer, но один из трёх required hot-topic buckets потерян и заменён более свободным пересказом недели.

## Summary

Самые сильные ответы здесь: `golden_v3_q076`, `golden_v3_q078`, `golden_v3_q080`, `golden_v3_q081`, `golden_v3_q083`, `golden_v3_q084`, `golden_v3_q085`.

Самые слабые: `golden_v3_q077`, затем `golden_v3_q082`.

Повторяющиеся паттерны в батче: ошибочный подбор примера при тематически близком ответе и у analytics-кейсов свободный пересказ weekly digest вместо точного воспроизведения hot-topic buckets.
