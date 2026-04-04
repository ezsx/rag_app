# Claude Review: SPEC-RAG-18 (Golden v2 + Offline Judge Artifacts)

> Ты — независимый reviewer спецификации перед реализацией.
> Нужно проверить не код, а **корректность и полноту spec**, который задаёт новый eval pipeline:
>
> - [docs/specifications/active/SPEC-RAG-18-golden-v2-offline-judge.md](../specifications/active/SPEC-RAG-18-golden-v2-offline-judge.md)
>
> Цель: убедиться, что spec:
> 1. не противоречит уже существующему eval/runtime
> 2. действительно решает проблему сломанного strict recall
> 3. корректно использует наши research findings по evaluation / metrics / grounding
> 4. не вводит новый неявный scope creep

---

## Что нужно сделать

### 1. Проверь spec на internal consistency

Сверь `SPEC-RAG-18` с репозиторием:

- `datasets/eval_golden_v1.json`
- `scripts/evaluate_agent.py`
- `docs/specifications/completed/SPEC-RAG-14-evaluation-pipeline.md`
- `docs/specifications/active/SPEC-RAG-16-hot-topics-channel-expertise.md`
- `docs/specifications/active/SPEC-RAG-17-production-hardening.md`
- `docs/progress/experiment_log.md`
- `docs/progress/project_scope.md`

Проверь:

- не ломает ли `golden_v2` backward compatibility с `golden_v1`
- не конфликтуют ли новые поля (`eval_mode`, `required_claims`, `acceptable_evidence_sets`) с текущим loader/reporting
- реалистично ли то, что runner будет сохранять `retrieved_docs`, `tool_observations`, `citations with excerpts`
- не смешаны ли в spec разные сущности: recall, grounding, factual correctness, tool routing

### 2. Сверь spec с research base

Обязательно сравни с research reports:

- `docs/research/reports/R18-deep-evaluation-methodology-dataset.md`
- `docs/research/reports/R19-deep-nli-citation-faithfulness.md`
- `docs/research/reports/R20-deep-retrieval-robustness-ndr-rsr-ror.md`
- `docs/research/reports/R22-deep-production-gap-analysis.md`

Проверь:

- корректно ли spec интерпретирует R18: где strict retrieval metric уместен, а где нет
- не путает ли spec `faithfulness`, `retrieval_sufficiency`, `evidence_support`, `factual correctness`
- не залезает ли spec в scope NLI / robustness, которые должны остаться отдельными tracks
- насколько разумно оставлять `strict_anchor_recall` как diagnostic-only metric

### 3. Проверь offline judge workflow

Это критичный блок spec.

Нужно сказать:

- достаточно ли информации в offline judging packet, чтобы Codex / Claude могли адекватно судить batch'ами по 30 вопросов
- не слишком ли бедный artifact для оценки retrieval sufficiency
- не появляется ли скрытая зависимость от новых DB lookups, хотя spec их запрещает
- правильно ли ограничен judge:
  - можно смотреть retrieved/cited docs
  - нельзя делать новый свободный semantic search по всей БД

Если чего-то не хватает в packet:
- перечисли конкретно какие поля нужно добавить

### 4. Проверь реалистичность реализации

Мне нужен не abstract feedback, а инженерный.

Скажи:

- реалистично ли внедрить `SPEC-RAG-18` инкрементально
- какой порядок реализации логичнее:
  1. schema / loader
  2. dataset
  3. report artifacts
  4. offline judge export
  5. aggregate redesign
- нет ли скрытых сложных мест, которые spec сейчас недооценивает
- что стоит явно вынести в "не входит в scope", чтобы implementation не расползлась

### 5. Проверь dataset design

Оцени саму идею `golden_v2`:

- достаточно ли разделения на `retrieval_evidence / analytics / navigation / refusal`
- правильно ли включать `hot_topics` и `channel_expertise` в `analytics`
- нужно ли ещё одно поле вроде `judge_mode` или `grounding_mode`
- достаточно ли `acceptable_evidence_sets`, или стоит сразу предусмотреть claim-level evidence structure

### 6. Дай patch-level suggestions

Не переписывай spec полностью.

Дай:

- конкретные findings
- severity
- file refs
- минимальный набор правок, после которого spec можно пускать в реализацию

---

## Формат ответа

```markdown
## Findings
- severity + file refs

## Research Alignment
- где spec хорошо согласована с R18/R19/R20/R25
- где interpretation сомнительна

## Offline Judge Review
- хватает ли artifact-а
- чего не хватает

## Implementation Risks
- скрытые сложности
- порядок внедрения

## Patch Suggestions
- что именно поправить в SPEC-RAG-18

## Verdict
- ready / needs one more edit pass
```

Если spec уже достаточно сильная, скажи это прямо. Если нет — укажи минимальный обязательный набор правок перед implementation.

