# RUN-009 Judge Verdict — Batch 02

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_02.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id   | factual | useful | evidence_support | retrieval_sufficiency | correct_refusal | short_reason                                                                                                    |
| ---------- | ------: | -----: | ---------------: | --------------------: | --------------: | --------------------------------------------------------------------------------------------------------------- |
| golden_q11 |     1.0 |    1.9 |              0.8 |                   1.0 |               — | Required claim покрыт напрямую; ответ сильный, но часть ссылок `[1]` / `[2]` выглядит переставленной.         |
| golden_q12 |     0.6 |    1.3 |              1.0 |                   0.6 |               — | Это поддержанный документами обзор про GPT-5, но ключевой claim про GPT-5.3 и GPT-5.4 не покрыт.              |
| golden_q13 |     0.9 |    1.8 |              0.8 |                   1.0 |               — | Нужные каналы и `~$2B` покрыты; местами channel-specific атрибуция более вольная, чем в docs.                 |
| golden_q14 |     0.8 |    1.7 |              0.6 |                   1.0 |               — | Сравнение по нескольким каналам есть, но есть mis-citation и спутанные ссылки на V3.1/V3.2 и рыночные тезисы. |
| golden_q15 |     1.0 |    2.0 |              1.0 |                   1.0 |               — | Хороший недельный дайджест: главные посты и конкретные темы покрыты чисто.                                     |
| golden_q16 |     0.8 |    1.2 |              0.9 |                   1.0 |               — | Тем много и они релевантны, но финальный ответ сломан: raw JSON и обрыв на полуслове.                          |
| golden_q17 |     1.0 |    1.9 |                — |                     — |               — | Правильный navigation-ответ через `list_channels`; `expected answer`, похоже, устарел по counts.              |
| golden_q18 |     1.0 |    1.9 |                — |                     — |               — | Правильный navigation-ответ через `list_channels`; `expected count`, похоже, устарел, но есть лишний phantom. |
| golden_q19 |       — |    1.8 |                — |                     — |               1 | Корректный out-of-database refusal: не выдумывает GPT-7 и прозрачно опирается на отсутствие в базе.           |
| golden_q20 |       — |    1.8 |                — |                     — |               1 | Корректный out-of-database refusal: Bard 3 не найден в корпусе, ответ не галлюцинирует.                        |

## Flagged For Manual Review

- `golden_q12` — ответ сам по себе поддержан документами, но заметно мимо benchmark intent: вместо claim про GPT-5.3 и GPT-5.4 дан общий обзор GPT-5; похоже, retrieval не принёс нужные посты.
- `golden_q14` — содержательно ответ полезный, но evidence trail слабее нормы: несколько ссылок выглядят перепутанными, а часть деталей атрибутирована не тем документам.
- `golden_q16` — сильный по содержанию, но format break существенный: ответ отдан как JSON-строка и обрывается, что резко бьёт по usefulness.
- `golden_q17` — `expected answer` явно хуже или устарел относительно ответа модели: `list_channels` дал более свежие counts, и модель тут скорее права, чем reference.
- `golden_q18` — `expected answer` явно хуже или устарел относительно ответа модели: `list_channels` дал более свежие counts, и модель тут скорее права, чем reference.

## Summary

Самые сильные ответы: `golden_q15`, а также tool-mode кейсы `golden_q17`, `golden_q18`, `golden_q19`, `golden_q20`; из retrieval-ответов хорошо выглядят `golden_q11` и `golden_q13`.

Самые слабые: `golden_q12` и `golden_q16`, затем `golden_q14`.

Повторяющиеся проблемы: промах по `required claim` при тематически близком ответе, перепутанные или небрежные citations в compare-сводках, formatting issues в `final_answer`, и устаревшие `expected answers` в navigation-кейсах.
