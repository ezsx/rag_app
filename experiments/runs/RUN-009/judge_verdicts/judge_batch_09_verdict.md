# RUN-009 Judge Verdict — Batch 09

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_09.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id       | factual | useful | evidence_support | retrieval_sufficiency | correct_refusal | short_reason                                                                                           |
| -------------- | ------: | -----: | ---------------: | --------------------: | --------------: | ------------------------------------------------------------------------------------------------------ |
| golden_v3_q086 |     1.0 |    1.9 |                — |                     — |               — | `hot_topics` и число постов `409` переданы точно.                                                       |
| golden_v3_q087 |     1.0 |    1.9 |                — |                     — |               — | `396` постов и ключевые weekly topics покрыты; ответ чуть шире, но по делу.                            |
| golden_v3_q088 |     0.7 |    1.5 |                — |                     — |               — | Профиль канала в целом верный, но часть required top topics потеряна.                                  |
| golden_v3_q089 |     0.6 |    1.4 |                — |                     — |               — | Общий профиль `data_secrets` верный, но missing `total_posts` и часть требуемых topic buckets.         |
| golden_v3_q090 |     0.9 |    1.8 |                — |                     — |               — | `xor_journal` описан хорошо; сущности совпадают, есть только мелкая неполнота по exact topic list.     |
| golden_v3_q091 |     0.9 |    1.8 |                — |                     — |               — | `boris_again` описан по `channel_expertise` почти полностью; minor omissions по exact structured fields. |
| golden_v3_q092 |     0.9 |    1.8 |                — |                     — |               — | `ai_newz` покрыт хорошо, с нужными entity/topic направлениями; нет лишь части формальных метаданных.   |
| golden_v3_q093 |     0.9 |    1.8 |                — |                     — |               — | `aioftheday` описан точно и полезно, почти все required claims закрыты.                                 |
| golden_v3_q094 |     0.9 |    1.8 |                — |                     — |               — | `gonzo_ml` передан корректно; хороший аналитический профиль, лишь не идеально по exact topic artifacts. |
| golden_v3_q095 |     0.9 |    1.8 |                — |                     — |               — | `seeallochnaya` покрыт хорошо, включая сущности и основные topic направления.                           |

## Flagged For Manual Review

- `golden_v3_q088` — ответ в целом верный по `channel_expertise`, но заметно недодаёт exact structured topics из required claims. Это скорее good summary, weak exact extraction, чем outright factual fail.
- `golden_v3_q089` — ответ в целом верный по `channel_expertise`, но заметно недодаёт `total_posts` и часть exact structured topics из required claims. Это скорее good summary, weak exact extraction, чем outright factual fail.

## Summary

Самые сильные ответы здесь: `golden_v3_q086`, `golden_v3_q087`, `golden_v3_q090`, `golden_v3_q091`, `golden_v3_q092`, `golden_v3_q093`, `golden_v3_q094`, `golden_v3_q095`.

Самые слабые: `golden_v3_q089`, затем `golden_v3_q088`.

Повторяющийся паттерн в батче: хороший human-readable analytics summary вместо точного воспроизведения всех structured fields из `channel_expertise`.
