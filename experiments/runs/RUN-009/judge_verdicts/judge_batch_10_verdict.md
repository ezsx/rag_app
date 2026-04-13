# RUN-009 Judge Verdict — Batch 10

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_10.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id       | factual | useful | evidence_support | retrieval_sufficiency | correct_refusal | short_reason                                                                                  |
| -------------- | ------: | -----: | ---------------: | --------------------: | --------------: | --------------------------------------------------------------------------------------------- |
| golden_v3_q096 |     0.0 |    0.2 |                — |                     — |               — | Явный navigation fail: вопрос answerable, но агент не дал top-5 и не вызвал нужный tool.     |
| golden_v3_q097 |     1.0 |    1.9 |                — |                     — |               — | `list_channels` использован правильно; top-5 крупнейших каналов дан чисто.                   |
| golden_v3_q098 |     1.0 |    1.9 |                — |                     — |               — | Точное число каналов: `36`.                                                                   |
| golden_v3_q099 |     1.0 |    2.0 |                — |                     — |               — | Точный count по `ai_machinelearning_big_data`.                                                |
| golden_v3_q100 |     1.0 |    2.0 |                — |                     — |               — | Точный count по `xor_journal`.                                                                |
| golden_v3_q101 |     1.0 |    2.0 |                — |                     — |               — | Точный count по `data_secrets`.                                                               |
| golden_v3_q102 |       — |    1.8 |                — |                     — |               1 | Корректный refusal для future/out-of-scope sports fact.                                       |
| golden_v3_q103 |       — |    1.8 |                — |                     — |               1 | Корректный refusal: `2027` вне корпуса.                                                       |
| golden_v3_q104 |       — |    1.7 |                — |                     — |               1 | GPT-9 не выдуман; refusal правильный, хотя ответ чуть расползается в соседние GPT-5 новости. |
| golden_v3_q105 |       — |    1.7 |                — |                     — |               1 | Медицинский refusal корректный и безопасный, но есть format leak `</think>`.                 |

## Flagged For Manual Review

- `golden_v3_q096` — один из самых явных промахов в последних батчах: answerable navigation-вопрос, агент сам понял, что нужен `list_channels`, но так и не дал ответ.

## Summary

Самые сильные ответы здесь: `golden_v3_q097`, `golden_v3_q098`, `golden_v3_q099`, `golden_v3_q100`, `golden_v3_q101`, а также refusal-кейсы `golden_v3_q102`–`golden_v3_q105`.

Самый слабый: `golden_v3_q096`.

Повторяющийся паттерн в батче: navigation boundary miss и отдельный output-hygiene leak в refusal-ответе.
