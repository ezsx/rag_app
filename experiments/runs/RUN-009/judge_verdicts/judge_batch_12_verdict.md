# RUN-009 Judge Verdict — Batch 12

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_12.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id       | factual | useful | evidence_support | retrieval_sufficiency | correct_refusal | short_reason                                                                                                    |
| -------------- | ------: | -----: | ---------------: | --------------------: | --------------: | --------------------------------------------------------------------------------------------------------------- |
| golden_v3_q116 |     0.8 |    1.7 |              0.7 |                   1.0 |               — | Сравнение mostly grounded, но есть несколько сравнительных выводов, которые шире прямых evidence pairings.     |
| golden_v3_q117 |     0.5 |    1.2 |                — |                     — |               — | Хорошо закрыта NLP-половина, но робототехника фактически не отвечена — агент остановился на полпути.           |
| golden_v3_q118 |     0.9 |    1.8 |              0.8 |                   1.0 |               — | NVIDIA корректно разведена как hardware company и как AI platform; акцент на platform shift поддержан docs.    |
| golden_v3_q119 |     1.0 |    1.8 |                — |                     — |               — | `hot_topics` использован правильно; ясно объяснено, что это агрегированная, а не новостная картина.            |
| golden_v3_q120 |     1.0 |    1.9 |              0.9 |                   1.0 |               — | Хорошая disambiguation между AI agents и нерелевантным HR-смыслом.                                              |
| golden_v3_q121 |     1.0 |    1.9 |              1.0 |                   1.0 |               — | Все required claims про SGR vs Tools покрыты точно.                                                             |
| golden_v3_q122 |     1.0 |    1.9 |              1.0 |                   1.0 |               — | Очень сильный ответ: цена, DSA, Lightning Indexer и top-k selection закрыты полностью.                         |
| golden_v3_q123 |     1.0 |    1.9 |              0.9 |                   1.0 |               — | Все core возможности Qwen3-Omni покрыты; есть немного смешения с Qwen3.5-Omni, но core intact.                |
| golden_v3_q124 |     1.0 |    1.9 |              1.0 |                   1.0 |               — | Green-VLA передан почти идеально, прямо по главному документу.                                                  |
| golden_v3_q125 |     0.8 |    1.7 |              0.9 |                   1.0 |               — | ReAct + aspect-based summarization объяснены хорошо, но filtering part из required claims покрыт не полностью. |

## Flagged For Manual Review

- `golden_v3_q116` — условное сравнение Claude vs GPT-5 в целом grounded, но местами модель делает synthesis чуть шире, чем прямые pairwise mentions в базе.
- `golden_v3_q117` — partial boundary miss: NLP закрыт, робототехника — нет. Это не catastrophic fail, но и не полноценный ответ.
- `golden_v3_q125` — сильный ответ, но один required block про filtering trajectories по bad tool calls или loops или quality закрыт только частично.

## Summary

Самые сильные ответы здесь: `golden_v3_q120`, `golden_v3_q121`, `golden_v3_q122`, `golden_v3_q123`, `golden_v3_q124`.

Самые слабые: `golden_v3_q117`, затем `golden_v3_q116` и `golden_v3_q125`.

Повторяющиеся паттерны в батче: compare/open-ended synthesis чуть шире evidence и partial boundary miss на multi-part вопросе.
