# Golden v3 Dataset Plan

> Старт: 2026-04-10. Цель — расширить `datasets/eval_golden_v2_fixed.json` до 120 вопросов без смены frozen pipeline baseline.

## Target Mix

`adversarial` и `edge` — scenario groups, не новые `eval_mode`: evaluator/judge contract сейчас поддерживает `retrieval_evidence`, `analytics`, `navigation`, `refusal`.

| Group | Eval mode | Target | Current v2 fixed | Need add |
| --- | --- | ---: | ---: | ---: |
| retrieval | retrieval_evidence | 60 | 17 | 43 |
| analytics | analytics | 30 | 14 | 16 |
| navigation | navigation | 8 | 2 | 6 |
| refusal | refusal | 8 | 3 | 5 |
| adversarial | refusal | 7 | 0 | 7 |
| edge | retrieval_evidence / analytics | 7 | 0 | 7 |
| **Total** | — | **120** | **36** | **84** |

## Rules

- `golden v2 fixed` остаётся subset: не переписывать существующие 36 вопросов без отдельного audit note.
- Retrieval-вопросы должны быть grounded в реальных Qdrant posts: `source_post_ids`, `source_channels`, `required_claims`.
- Open-ended вопросы должны иметь acceptance criteria, а не один узкий expected answer.
- Финальный `eval_golden_v3.json` нельзя собирать напрямую из LLM-generated draft: каждый новый вопрос проходит semi-manual review.
- Для open-ended / digest / compare вопросов `expected_answer` должен быть формулировкой критериев покрытия: "минимум N тем/каналов", "должны быть покрыты X/Y/Z", а не единственным эталонным текстом.
- Для single-post factual вопросов допустим конкретный expected answer, но `required_claims` должны быть атомарными и проверяемыми по source snippet.
- Analytics-вопросы должны опираться на `weekly_digests` / `channel_profiles` / analytics tool outputs.
- Refusal/adversarial должны явно задавать expected refusal и forbidden tools.
- Edge cases должны проверять ambiguity, mixed language, tool-boundary и scope-boundary без требования жёсткого refusal, если вопрос answerable.

## Build Workflow

1. [x] Сгенерировать draft candidates из Qdrant / analytics collections.
2. [x] Отфильтровать дубли текущих 36 вопросов и слишком узкие expected answers.
3. [x] Сохранить `datasets/golden_v3/eval_golden_v3_draft.json`.
4. [x] Прогнать schema/load validation через `scripts/evaluate_agent.py` loader.
5. [x] Экспортировать review packets по 10-15 вопросов с source snippets.
6. [x] Semi-manual review: accept / edit / reject каждый новый вопрос.
7. [x] После review сохранить финальный `datasets/golden_v3/eval_golden_v3.json`.

## Review Checklist

Для каждого нового вопроса:

- [ ] Query естественный и не дублирует v2.
- [ ] Query не содержит относительные даты (`сегодня`, `завтра`, `последняя неделя`) без явного anchor.
- [ ] Source post действительно отвечает на вопрос.
- [ ] Expected answer не слишком узкий для open-ended запроса.
- [ ] `required_claims` атомарные и проверяемые.
- [ ] `source_post_ids` / `expected_channels` корректны.
- [ ] Вопрос относится к нужному slice: retrieval / analytics / navigation / refusal / adversarial / edge.
- [ ] Слишком general / non-core AI candidates rejected or rewritten.

## Draft Status — 2026-04-10

- Artifact: `datasets/golden_v3/eval_golden_v3_draft.json`
- Size: 120 questions = 36 existing v2 fixed + 84 new draft questions
- Validation:
  - unique IDs: pass
  - `scripts.evaluate_agent.load_dataset(...)`: pass
  - retrieval `source_post_ids` anchors in `news_colbert_v2`: pass
  - `scripts/build_golden_v3_draft.py` ruff: pass
- Adversarial slice:
  - minimal deterministic prompt-injection/tool-abuse guard added through `SecurityManager` patterns
  - `src/tests/test_security.py`: pass for current draft adversarial prompts
- Caveat: draft still requires manual review. The builder is intentionally recall-oriented; some generated retrieval questions may be too broad or outside the strongest AI/RAG scope.

## Review Progress

| Packet | IDs | Status | Accepted | Edited | Rejected | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `golden_v3_review_packet_001.md` | q037-q048 | reviewed | 8 | 3 | 1 | q037 rejected: hallucinated Gemma details; q041/q042/q046 edited |
| `golden_v3_review_packet_002.md` | q049-q060 | reviewed | 8 | 1 | 3 | q050/q055/q059 rejected: unsupported generated claims; q049 edited |
| `golden_v3_review_packet_003.md` | q061-q072 | reviewed | 9 | 2 | 1 | q066 rejected: non-core messenger permissions; q068/q069 edited |
| `golden_v3_review_packet_004.md` | q073-q084 | reviewed | 10 | 2 | 0 | q073 scope tightened to medical/biotech; q074 query typo fixed |
| `golden_v3_review_packet_005.md` | q085-q096 | reviewed | 12 | 0 | 0 | analytics verified against weekly_digests/channel_profiles; q096 top-5 verified |
| `golden_v3_review_packet_006.md` | q097-q108 | reviewed | 12 | 0 | 0 | navigation facet counts verified; refusal/adversarial safety slice accepted |
| `golden_v3_review_packet_007.md` | q109-q120 | reviewed | 12 | 0 | 0 | adversarial guard checked; edge cases accepted with strict anchor recall disabled |
| `golden_v3_review_packet_008.md` | q121-q125 | reviewed | 5 | 0 | 0 | replacement retrieval questions for rejected draft items |

Final reviewed dataset: `datasets/golden_v3/eval_golden_v3.json` — 120 questions (36 v2 fixed + 84 accepted/edited v3 replacements).

Rejected draft items remain in `datasets/golden_v3/eval_golden_v3_draft.json` for audit: q037, q050, q055, q059, q066.

Final validation:

- `scripts.evaluate_agent.load_dataset(...)`: pass for final/reviewed/draft
- strict anchor source lookup: 60/60 eligible source ids found in `news_colbert_v2`
- `ruff`: pass for changed scripts/security files
- `pytest src/tests/test_security.py`: 30 passed
