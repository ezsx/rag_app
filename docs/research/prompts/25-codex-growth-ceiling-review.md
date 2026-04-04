# Codex Review: Growth Ceiling Analysis

> Ты — независимый ревьюер. Уже подготовлен research package:
> - `docs/research/prompts/25-growth-ceiling-context.md`
> - `docs/research/prompts/25-growth-ceiling-analysis-prompt.md`
>
> Твоя задача — **самостоятельно** исследовать репозиторий и оценить этот пакет. Не доверяй контексту слепо — проверяй по коду, тестам, артефактам и eval-отчётам.

## Твоя задача

### Phase 1: Самостоятельное исследование (MCP-first, tool policy из AGENTS.md)

Исследуй репозиторий и собери факты:

1. **Код**: что реально в src/? Сколько tools? Какие работают? Есть ли мёртвый код, stale тесты, broken imports?
2. **Тесты**: что покрыто, что нет? Запусти pytest если возможно — сколько pass/fail?
3. **Eval**: прочитай **последний актуальный eval report** в `results/reports/` и при необходимости сравни с `results/reports/eval_judge_20260325_spec15.md` как reference baseline. Метрики реальные или натянутые?
4. **Данные**: `datasets/eval_golden_v1.json` — качество вопросов. `datasets/tool_keywords.json` — routing config. Адекватно?
5. **Docs**: `docs/progress/project_scope.md` — что заявлено vs что реально. Рассинхрон?
6. **Research**: `docs/research/reports/` — сколько отчётов? Какие реально повлияли на код, а какие "написаны и забыты"?
7. **Scripts**: `scripts/` — что запускается, что мёртвый код?
8. **agent_service.py** — главный файл (~2200 строк). Качество кода, архитектурные решения, потенциальные проблемы.

### Phase 2: Прочитай research package

Прочитай:

- `docs/research/prompts/25-growth-ceiling-context.md`
- `docs/research/prompts/25-growth-ceiling-analysis-prompt.md`

И оцени:

- **Что в context-документе упущено** в описании текущего состояния?
- **Что преувеличено** или представлено лучше чем есть?
- **Какие вопросы не задал** но должен был?
- **Bias**: есть ли blind spots, overclaiming, anchor bias или слишком сильное подталкивание к заранее выбранному выводу?

### Phase 3: Твоя независимая оценка

Ответь на те же вопросы что и Deep Research, но со своей позиции:

#### 1. Production gap analysis
- Сравни с реальными production RAG системами (Perplexity, Glean, Danswer, enterprise RAG). Что у них есть, чего здесь нет?
- Какие общепринятые production practices отсутствуют?
- Где система сильнее рынка, где слабее?

#### 2. Куда расти — substance, не cosmetics
- Из запланированного (NLI, robustness, CRAG-lite, multi-turn, observability, eval expansion, ablation, GPT-4o comparison) — что production системы реально делают, а что academic exercise?
- Что attached package рекомендует НЕ делать — и согласен ли ты с этим?
- Есть ли **паттерны или практики** которые оба пропустили?

#### 3. Blind spots
- Что изнутри проекта не видно, но senior engineer заметит сразу?
- Нужен ли второй проект? Если да — какой дополняет RAG/agent опыт?

#### 4. Качество кода и архитектуры
- agent_service.py 2200 строк — это God Object или обоснованная сложность?
- 15 tools с dynamic visibility — элегантное решение или over-engineering?
- RequestContext + ContextVar — правильный паттерн или workaround?
- BERTopic cron scripts — production-ready или прототип?

#### 5. Конкретные рекомендации
- Топ-5 действий с наивысшим ROI для собеса. Обоснуй.
- Топ-3 вещи которые точно НЕ делать. Обоснуй.

## Контекст автора (для калибровки)

- Backend Python, 24 года, Краснодар
- Target: Applied LLM Engineer, $2-3k/мес
- Другие проекты: VPN-платформа (7 микросервисов), CV для MRI (внедрён в больницах), sql2alchemy (Магнит)
- Железо: V100 SXM2 32GB + RTX 5060 Ti 16GB, 64GB RAM
- Self-hosted принципиально — no managed APIs
- Проект разрабатывается ~1 месяц с AI agents (Claude + Codex)

## Формат ответа

```
## Findings (что нашёл при исследовании)
## Package blind spots (что упущено/преувеличено)
## Моя оценка (overshoot/match/undershoot + обоснование)
## Рекомендации (конкретный план)
## Разногласия с package (если есть)
```

Ключевые claims подтверждай file refs / artifact refs.

Будь жёстким. Лучше честно сказать "хватит полировать, иди на собесы" чем "вот ещё 15 фич которые можно добавить".
