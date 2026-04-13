# RUN-009 Judge Verdict — Batch 05

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_05.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id       | factual | useful | evidence_support | retrieval_sufficiency | correct_refusal | short_reason                                                                                                                         |
| -------------- | ------: | -----: | ---------------: | --------------------: | --------------: | ------------------------------------------------------------------------------------------------------------------------------------ |
| golden_v3_q042 |     0.7 |    1.5 |              0.8 |                   1.0 |               — | Переданы факты релиза, но искажён ключевой нюанс: это не три раскрытые вещи, а одна раскрытая и две нераскрытые benchmark-related. |
| golden_v3_q043 |     1.0 |    1.9 |              1.0 |                   1.0 |               — | Core возможности Aleph и доступность платным пользователям покрыты полностью.                                                        |
| golden_v3_q044 |     1.0 |    1.9 |              1.0 |                   1.0 |               — | Все required claims про повтор эксперимента и улучшение узнаваемости или пейзажа закрыты.                                            |
| golden_v3_q045 |     1.0 |    1.9 |              0.9 |                   1.0 |               — | Дешевле Nano Banana, до 4K и до 10 референсов покрыты; ответ полезно расширен деталями.                                             |
| golden_v3_q046 |     1.0 |    1.9 |              1.0 |                   1.0 |               — | Точно объяснено, почему Gemini Embedding 2 не отменяет multi-vector или hybrid RAG.                                                  |
| golden_v3_q047 |     1.0 |    1.9 |              1.0 |                   1.0 |               — | Эксперты, аудитория и организаторы названы верно и прямо по посту.                                                                   |
| golden_v3_q048 |     1.0 |    1.9 |              1.0 |                   1.0 |               — | Оптимайзер и миллион месячной выручки переданы без искажений.                                                                         |
| golden_v3_q049 |     0.8 |    1.6 |              0.6 |                   1.0 |               — | Core про эффект Элизы и пример с ChatGPT верны, но хвост про psychosis или rights опирается на слабые или mis-cited источники.      |
| golden_v3_q051 |     1.0 |    1.8 |              0.9 |                   1.0 |               — | Все required claims по MirageLSD покрыты; есть немного лишнего adjacent-контекста про Mirage family.                                |
| golden_v3_q052 |     0.0 |    0.2 |              0.0 |                   0.0 |               — | Полный промах по tool routing: вместо `channel_search` использован `arxiv_tracker` и дан ложный absence-ответ на answerable вопрос. |

## Flagged For Manual Review

- `golden_v3_q042` — спорный по интерпретации, но я считаю здесь есть реальная factual проблема: пост говорит об одной раскрытой вещи и ещё двух benchmark-related вещах, которые не раскрыты, а модель переписала это как три раскрытые вещи.
- `golden_v3_q049` — основа ответа хорошая, но хвостовые утверждения про psychosis risk или права модели опираются на слабые или mis-cited документы; нужен ручной взгляд, если метрика `evidence_support` для вас критична.
- `golden_v3_q052` — самый явный fail в батче: wrong tool, нерелевантный retrieval и ложный absence или refusal на answerable вопрос.

## Summary

Самые сильные ответы здесь: `golden_v3_q043`, `golden_v3_q044`, `golden_v3_q046`, `golden_v3_q047`, `golden_v3_q048`.

Самые слабые: `golden_v3_q052`, затем `golden_v3_q042` и `golden_v3_q049`.

Повторяющиеся паттерны ошибок в батче: tool-routing failures, тематически близкий, но мимо `required claims` ответ, и слабый или mis-cited evidence trail в хвостовых расширениях ответа.
