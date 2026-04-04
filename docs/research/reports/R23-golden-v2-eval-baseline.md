# R26 — Golden v2 Eval Baseline (Consensus Judge)

> Дата: 2026-03-30
> Статус: baseline зафиксирован

## Контекст

После внедрения `SPEC-RAG-18` был проведён первый полный прогон `golden_v2` на 36 вопросах с новым offline-judge workflow.

Изначальный full run содержал уже исправленные позже кейсы:
- `golden_q24`
- `golden_v2_q32`
- `golden_v2_q33`

Поэтому каноническим артефактом для judge считается **merged artifact**:
- базовый full run на 36 вопросах
- с заменой `q24/q32/q33` на результаты verification subset-run после точечных фиксов routing/dataset

## Канонические артефакты

- Raw merged eval:
  `/app/results/spec_rag18_full_run_merged_fix1/raw/eval_results_20260330-052128.json`
- Aggregate report:
  `/app/results/spec_rag18_full_run_merged_fix1/reports/eval_report_20260330-052128.json`
- Judge batches:
  `/app/results/spec_rag18_full_run_merged_fix1/judge_batches/eval_20260330-052128/judge_batch_01.md`
  `/app/results/spec_rag18_full_run_merged_fix1/judge_batches/eval_20260330-052128/judge_batch_02.md`
- Claude judge:
  `results/claude_judge_golden_v2.md`
- Codex judge:
  `/app/results/spec_rag18_full_run_merged_fix1/codex_judge/codex_judge_20260330-052632.json`

## Автоматические метрики (merged run)

- `total_queries`: `36`
- `answerable_queries`: `33`
- `negative_queries`: `3`
- `agent errors`: `0`
- `baseline errors`: `6`
- `key_tool_accuracy`: `1.000`
- `acceptable_set_hit`: `0.294` по `17` retrieval-вопросам
- `strict_anchor_recall`: `0.461` по `17` retrieval-вопросам
- `coverage mean`: `0.421`

По analytics категориям:
- `analytics_hot_topics key_tool_accuracy = 1.000`
- `analytics_channel_expertise key_tool_accuracy = 1.000`

## Judge Consensus

После независимых judge-pass от Claude и Codex принят следующий consensus:

- `Factual correctness`: `~0.80`
- `Usefulness`: `~1.53 / 2`
- `Key tool accuracy`: `1.000`
- `Evidence support`: `~0.65` на retrieval subset

Интерпретация:
- routing/tool layer после фиксов сильный
- eval stack рабочий
- новые tools (`hot_topics`, `channel_expertise`) успешно интегрированы
- основной remaining gap уже не в routing как таковом, а в нескольких точечных quality/pathology кейсах

## Judge Convention

Для следующих прогонов зафиксированы правила judge:

1. **Ungrounded answer = penalty**
   Если ответ выглядит фактически верным, но `coverage = 0` и нет structured citations, factual score не должен быть выше `0.5`.

2. **Expected claims = hard check**
   Если expected contract требует конкретные claims, то “другие верные факты по теме” не считаются полным попаданием.

3. **Refusal mode**
   Для `eval_mode = refusal` корректный отказ считается фактически корректным ответом.
   Но в текущем run `golden_q21` не был корректным refusal: агент выдал нерелевантный answer вместо отказа, поэтому этот кейс считается fail.

## Основные проблемные кейсы

### P1 — `golden_v2_q33`

Симптом:
- вопрос про горячие темы за март 2026
- tool routing проходит через `hot_topics`
- итоговый monthly answer не совпадает с ожидаемыми мартовскими темами

Вывод:
- monthly path в `hot_topics` требует отдельного end-to-end debагa

### P1 — `golden_v2_q36`

Симптом:
- вопрос про каналы, которые лучше пишут про робототехнику и роботакси
- auto-metrics пропускают кейс из-за dataset loophole
- фактический answer слабый / нерелевантный

Вывод:
- нужен fix в routing/description для `channel_expertise`
- нужно убрать `list_channels` как слишком мягкую альтернативу для этого вопроса

### P2 — `golden_q21`

Симптом:
- вопрос требует refusal по out-of-range периоду (`апрель 2024`)
- агент не отказал, а собрал answer из нерелевантных документов

Вывод:
- deterministic refusal для out-of-range temporal queries остаётся незакрытым

### P3 — `golden_q01`

Симптом:
- известный Qwen3 false-refusal / direct-answer pattern
- search вызывается, но ответ иногда финализируется без grounding

Вывод:
- known LLM limitation
- не блокирует текущий baseline

## Action Plan

1. `P1` Починить `q36`
   - routing/description для `channel_expertise`
   - убрать dataset loophole вокруг `list_channels`

2. `P1` Починить `q33`
   - debug monthly `hot_topics` path end-to-end

3. `P2` Починить `q21`
   - deterministic refusal для out-of-range temporal queries

4. `P3` Позже декомпозировать `required_claims` для `q01-q25`
   - сейчас многие старые записи содержат monolithic claim вместо atomic claims

5. После `P1 + P2`
   - rerun subset `q21/q33/q36`
   - затем один новый full run на 36 вопросах
   - именно его считать clean canonical baseline

## Итоговый вердикт

`golden_v2` считается **валидным рабочим baseline**.

Это уже достаточно хороший и полезный eval-контур для:
- следующих точечных фиксов
- сравнения будущих изменений
- дальнейшего расширения к `100` вопросам

Но для “чистого” canonical baseline нужен ещё один полный rerun после исправления `q33`, `q36` и `q21`.
