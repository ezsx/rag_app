# Documentation Governance

> Свод правил ведения документации проекта. ОБЯЗАТЕЛЕН для всех агентов (Claude, Codex).
> Нарушение этих правил приводит к "свалке" — главной проблеме которую мы решаем.

---

## Принципы

1. **Каждый документ имеет ровно одно место** — если не знаешь куда положить, прочитай этот файл.
2. **architecture/ = зеркало кода** — всегда отражает текущее состояние. Устаревший docs/architecture хуже чем его отсутствие.
3. **research/ = неизменяемый архив** — отчёты НЕ редактируются post-factum. Новые findings = новый отчёт.
4. **specifications/ = мост между research и code** — spec описывает ЧТО делать, КАК, и ЗАЧЕМ.
5. **planning/ = операционные живые документы** — scope, playbook, планы внедрения. Обновляются при каждом значимом изменении.

---

## Структура docs/

```
docs/
├── architecture/           ← ИСТОЧНИК ПРАВДЫ: "что сделано и зачем"
│   ├── 00-meta/           (как пользоваться документацией, governance)
│   ├── 01-scope/          (scope проекта)
│   ├── 02-glossary/       (термины)
│   ├── 03-invariants/     (правила которые нельзя нарушать)
│   ├── 04-system/         (системная архитектура, диаграммы)
│   ├── 05-flows/          (потоки: ingest, agent, eval)
│   ├── 06-api/            (API контракты)
│   ├── 07-data-model/     (Qdrant schema, payloads, vectors)
│   ├── 08-security/       (auth, sanitization)
│   ├── 09-observability/  (logging, metrics)
│   ├── 10-open-questions/ (нерешённые вопросы)
│   └── 11-decisions/      (decision log — ADR)
│
├── research/               ← ИССЛЕДОВАНИЯ: "что изучили"
│   ├── prompts/            (промпты для Deep Research, пронумерованы)
│   └── reports/            (отчёты R01-R99, пронумерованы, неизменяемы)
│
├── specifications/         ← СПЕЦИФИКАЦИИ: "что планируем сделать"
│   ├── active/             (текущие и следующие specs)
│   └── completed/          (имплементированные — архив для reference)
│
└── planning/               ← ОПЕРАЦИОННЫЕ ДОКИ: "как идём к цели"
    ├── project_scope.md    (roadmap, фазы, метрики)
    ├── retrieval_improvement_playbook.md  (история экспериментов)
    └── [plan].md           (планы внедрения конкретных фич)
```

---

## Workflow: когда что создавать и обновлять

### 1. Research — новое исследование

**Когда**: появился вопрос требующий deep dive, сравнения подходов, выбора технологии.

**Действия**:
- Создать промпт: `docs/research/prompts/NN-topic-name.md` (следующий номер)
- Получить отчёт: `docs/research/reports/RNN-topic-name.md`
- Quick и Deep варианты: `RNN-quick-topic.md`, `RNN-deep-topic.md`
- Зафиксировать findings в `docs/planning/` (playbook или scope)

**Правила**:
- Номера промптов и отчётов последовательные, не пропускать
- Отчёты **неизменяемы** после создания (append clarification ОК, rewrite НЕТ)
- Промпты содержат полный контекст проекта (hardware, текущий pipeline, метрики)

### 2. Specification — решили внедрять

**Когда**: research завершён, решение принято, нужно описать ЧТО конкретно делать.

**Действия**:
- Создать spec: `docs/specifications/active/SPEC-RAG-NN-name.md`
- Spec содержит: цель, контекст (ссылки на reports), что менять, acceptance criteria
- После реализации: переместить в `docs/specifications/completed/`

**Правила**:
- Spec пишется ДО кода, не после
- Spec ссылается на конкретные research reports
- Acceptance criteria — конкретные (recall > X, latency < Y, тест Z проходит)

### 3. Implementation — пишем код

**Когда**: spec утверждён, начинается разработка.

**Действия**:
- Коммиты со ссылками на spec (в commit message или PR)
- Обновить `docs/planning/` с результатами (playbook, scope)

### 4. Documentation — обновляем architecture/

**Когда**: после каждого значимого изменения в коде (новая фича, смена архитектуры, новый компонент).

**Что обновлять** (чеклист):
- [ ] `04-system/overview.md` — если изменилась архитектура, компоненты, их связи
- [ ] `05-flows/` — если изменился flow (ingest, agent, eval)
- [ ] `07-data-model/` — если изменилась schema Qdrant, payloads, vectors
- [ ] `11-decisions/decision-log.md` — если принято архитектурное решение
- [ ] `06-api/` — если изменились API endpoints
- [ ] `03-invariants/` — если появились новые правила "нельзя нарушать"

**Правила**:
- architecture/ описывает ТЕКУЩЕЕ состояние, не историю
- Если что-то удалено из кода — удалить из docs (никаких "removed in v2" комментариев)
- Decision log — единственное место где фиксируется ПОЧЕМУ (исторический контекст)

---

## Правила именования

| Тип | Паттерн | Пример |
|-----|---------|--------|
| Research prompt | `NN-kebab-case.md` | `07-tool-router-architecture-prompt.md` |
| Research report | `RNN-kebab-case.md` | `R13-deep-tool-router-architecture.md` |
| Specification | `SPEC-RAG-NN-kebab-case.md` | `SPEC-RAG-11-adaptive-retrieval.md` |
| Architecture doc | `kebab-case.md` в соотв. папке | `05-flows/FLOW-02-agent.md` |
| Planning doc | `kebab-case.md` | `adaptive_retrieval_plan.md` |

---

## Что НЕ создавать

1. **Автогенерированные описания файлов** — код читается через MCP, не через markdown зеркало
2. **Дубликаты** — одна тема = один документ. Если есть в architecture, не повторять в planning
3. **Временные файлы** (plan.md, temp.md, notes.md) — либо в planning/ с нормальным именем, либо не создавать
4. **Pricing/cost analysis** — не часть технической документации
5. **Module-per-file docs** — docs/ai/modules/ был ошибкой, не повторять

---

## Ответственность агентов

### При КАЖДОМ коммите — проверить:
- Не создаю ли файл в неправильном месте?
- Нужно ли обновить architecture/?
- Нужно ли обновить planning/ (scope, playbook)?

### При создании нового файла — спросить себя:
- Это research? → `docs/research/`
- Это план что делать? → `docs/specifications/active/`
- Это описание текущего состояния? → `docs/architecture/`
- Это живой операционный документ? → `docs/planning/`
- Не подходит ни к чему? → **Не создавать. Обсудить с пользователем.**

### При рефакторинге/крупном изменении:
- Обновить `docs/architecture/04-system/overview.md`
- Добавить запись в `docs/architecture/11-decisions/decision-log.md`
- Обновить `docs/planning/project_scope.md` если затронуты фазы/метрики
