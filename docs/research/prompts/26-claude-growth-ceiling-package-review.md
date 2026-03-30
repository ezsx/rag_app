# Claude Review: Growth Ceiling Research Package

> Ты — независимый reviewer research package перед запуском Deep Research.
> Нужно проверить **не сам проект по существу с нуля**, а качество пакета, который пойдёт в Deep Research:
>
> - `docs/research/prompts/25-growth-ceiling-context.md`
> - `docs/research/prompts/25-growth-ceiling-analysis-prompt.md`
>
> Цель: убедиться, что context-doc и prompt не врут, не устарели, не слишком bias-ят вывод и не пропускают важные вопросы.

## Что нужно сделать

### 1. Проверь factual consistency

Сверь package с репозиторием:

- `README.md`
- `docs/planning/project_scope.md`
- `docs/specifications/active/SPEC-RAG-16-hot-topics-channel-expertise.md`
- `docs/specifications/active/SPEC-RAG-17-production-hardening.md`
- последний актуальный eval report в `results/reports/`
- `src/services/agent_service.py`
- `src/tests/`

Проверь:

- не завышены ли claims
- не устарели ли counts / metrics / statuses
- нет ли внутренних противоречий между context-doc и prompt

### 2. Проверь bias / anchoring

Оцени:

- не подталкивает ли context-doc к заранее выбранному выводу
- не слишком ли он продаёт проект вместо честной калибровки
- не пропущены ли неприятные, но важные факты
- достаточно ли prompt провоцирует **independent judgment**, а не подыгрывание автору

### 3. Проверь качество исследовательского вопроса

Скажи:

- хорошо ли сформулирован главный вопрос про growth ceiling / diminishing returns
- хватает ли вопросов про:
  - hiring signal
  - proof layer
  - portfolio packaging
  - second project decision
- нет ли важных blind spots, которые стоит добавить до запуска research

### 4. Дай patch-level suggestions

Не переписывай всё с нуля. Дай:

- конкретные findings
- что править в `25-growth-ceiling-context.md`
- что править в `25-growth-ceiling-analysis-prompt.md`
- какие вопросы добавить / убрать / смягчить

## Формат ответа

```markdown
## Findings
- severity + file refs

## Bias Check
- что anchor'ит
- что звучит честно

## Missing Questions
- чего не хватает для Deep Research

## Patch Suggestions
- context doc
- analysis prompt

## Verdict
- ready / needs one more edit pass
```

Если package уже достаточно хорош, скажи это прямо. Если нет — укажи минимальный набор правок перед запуском.
