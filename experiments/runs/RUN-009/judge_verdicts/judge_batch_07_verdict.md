# RUN-009 Judge Verdict — Batch 07

- Source packet: `experiments/runs/RUN-009/judge_packets_10q/judge_batches/eval_20260410-220837_10q/judge_batch_07.md`
- Judge model: `GPT-5.4 Pro (web)`
- Notes: `expected_answer` treated as weak reference, not strict answer key

| query_id       | factual | useful | evidence_support | retrieval_sufficiency | short_reason                                                                                       |
| -------------- | ------: | -----: | ---------------: | --------------------: | -------------------------------------------------------------------------------------------------- |
| golden_v3_q065 |     1.0 |    1.9 |              1.0 |                   1.0 | Scaling law, interaction metrics и критика “больше агентов = лучше” покрыты полностью.            |
| golden_v3_q067 |     0.9 |    1.8 |              0.9 |                   1.0 | Три required метода названы верно; ответ слегка расширен соседними `techno_yandex` постами.       |
| golden_v3_q068 |     1.0 |    1.9 |              1.0 |                   1.0 | Все требуемые цифры по HSE, Shanghai, Maryland и top-15 за 2024–2025 даны точно.                  |
| golden_v3_q069 |     1.0 |    1.8 |              0.9 |                   1.0 | Крупные компании шумят больше, но внедряют медленнее; малые кейсы и примеры описаны верно.        |
| golden_v3_q070 |     0.9 |    1.8 |              0.8 |                   1.0 | Core contrast Alice vs ChatGPT закрыт, но хвост про экосистемный контекст уже менее load-bearing. |
| golden_v3_q071 |     1.0 |    1.9 |              1.0 |                   1.0 | Dueling DQN, 6 признаков и fallback на LRU после 500 мкс переданы точно.                          |
| golden_v3_q072 |     1.0 |    1.9 |              0.9 |                   1.0 | 50k статей или 30 лет, Karlgren 1990 и CACM 1997 закрыты; дальше идёт уместное расширение.        |
| golden_v3_q073 |     0.9 |    1.7 |              1.0 |                   1.0 | MAI-DxO и Chai Discovery покрыты; лишний ERNIE 4.5 делает ответ чуть менее сфокусированным.       |
| golden_v3_q074 |     1.0 |    1.9 |              1.0 |                   1.0 | Fireworks, Cerebras, Groq и позиции Opus 4.1 или Mistral Medium 3.1 переданы без искажений.       |
| golden_v3_q075 |     0.9 |    1.7 |              0.8 |                   1.0 | Главный вывод про small parameter changes верен, но ответ расползается в широкий synthesis.       |

## Flagged For Manual Review

- `golden_v3_q067` — в целом сильный ответ, но есть adjacent expansion в соседние `techno_yandex` посты; если нужна строгая привязка к одному посту, кейс слегка спорный.
- `golden_v3_q070` — core contrast закрыт, но хвост про экосистемный контекст уже менее load-bearing; можно пересмотреть, если нужна жёсткая evidence discipline.
- `golden_v3_q075` — core claim закрыт, но ответ превращается в слишком широкий synthesis из нескольких adjacent экспериментов; если для вас критична точная привязка к одному авторскому выводу, кейс спорный.

## Summary

Самые сильные ответы здесь: `golden_v3_q065`, `golden_v3_q068`, `golden_v3_q071`, `golden_v3_q074`, а также `golden_v3_q069` и `golden_v3_q072`.

Самые слабые: `golden_v3_q075`, затем `golden_v3_q070` и `golden_v3_q073`.

Повторяющиеся паттерны в батче: сильный core answer + шумный хвост из adjacent facts и слишком широкий synthesis поверх корректного основного ответа.
