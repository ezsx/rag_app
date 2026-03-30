# Prompt: Codex Judge — Qwen3.5 + Langfuse eval run (2026-03-30)

## Задача

Провести independent judge evaluation для eval прогона RAG-агента. Claude Opus уже провёл свой judge — нужен consensus.

## Контекст

- **Модель**: Qwen3.5-35B-A3B Q4_K_M (свап с Qwen3-30B-A3B)
- **Dataset**: golden_v2 (36 Qs, 4 eval_modes)
- **Ключевые изменения**: planner переведён на chat_completion (39s → 4s), JSON raw_decode для subqueries, Langfuse observability, prompt fix для post-search pipeline
- **Claude judge**: factual 0.889, useful 1.722/2, KTA 0.970

## Что нужно сделать

1. Прочитать `results/eval_qwen35_langfuse/raw/eval_results_20260330-120258.json` — поле `per_question[].offline_judge_packet`
2. Для каждого из 36 вопросов оценить:
   - **Factual correctness** (0-1): 1.0 = все claims корректны и grounded, 0.5 = частично, 0 = hallucination/wrong
   - **Usefulness** (0-2): 2 = полный ответ с citations, 1 = partial/missing key info, 0 = бесполезен
3. Сохранить результаты в `results/eval_qwen35_langfuse/codex_judge_verdicts.md`

## Conventions (из SPEC-RAG-18)

- **Ungrounded answer** (ответ без compose_context, coverage=0): max factual 0.5
- **Expected claims** (из `dataset_contract.required_claims`): hard check, если claim не покрыт — factual не может быть 1.0
- **eval_mode=refusal** + correct refusal: factual 1.0, useful 2.0
- **eval_mode=analytics**: ответ должен опираться на analytics tool output, не на search
- **eval_mode=navigation**: ответ должен использовать list_channels/meta capability

## Формат verdict таблицы

```markdown
| Q | Mode | Factual | Useful | Notes |
|---|------|---------|--------|-------|
| q01 | retrieval | 1.0 | 2.0 | ... |
```

## Файлы

- Judge packets: `results/eval_qwen35_langfuse/raw/eval_results_20260330-120258.json`
- Claude judge (для сравнения после): `results/eval_qwen35_langfuse/claude_judge_verdicts.md`
- Dataset: `datasets/eval_golden_v2.json`

## Важно

- **НЕ** смотри Claude judge verdicts до завершения своего judge — это нужно для independent consensus
- Будь строже чем Claude если видишь основания — consensus строится на различиях
- Обращай внимание на:
  - Покрыты ли required_claims из dataset_contract
  - Есть ли hallucinations (claims без source)
  - Правильный ли tool path (eval_mode=analytics → analytics tool, не search)
  - Citations [N] ведут ли к реальным источникам
