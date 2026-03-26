# Parallel Agent Dispatch Protocol

> Адаптация [superpowers/dispatching-parallel-agents](https://github.com/obra/superpowers) под Claude + Codex workflow.
> Подчиняется AGENTS.md как верхнеуровневому контракту. MCP/tool policy остаётся обязательным.
> Встроенные sub-agents использовать только когда это разрешено текущим runtime/policy.

## Core Principle

Одна задача = один агент с focused scope. По умолчанию isolated context. Наследовать контекст только если задача напрямую зависит от уже собранного узкого контекста.

## Паттерны

### Паттерн 1: Owner + Sidecar (основной)

Наш самый частый workflow:
- **Owner** (Claude/один агент) — владеет implementation/spec, пишет код
- **Sidecar** (Codex/второй агент) — независимый review, eval analysis, targeted verification

Owner делает работу, Sidecar проверяет. Sidecar НЕ наследует session history Owner'а — получает crafted prompt с файлами для review.

### Паттерн 2: Sequential Handoff

- Agent A делает research/spec
- Agent B реализует по spec
- Agent A или B делает post-implementation review

Контекст передаётся через артефакты (spec файл, prompt файл), не через shared session.

### Паттерн 3: Parallel Implementation

Когда 2+ независимых компонентов:
- Agent 1 → файл/модуль A
- Agent 2 → файл/модуль B
- Merge + integration test

**Только когда:** агенты НЕ редактируют одни файлы, fix одного НЕ влияет на другой.

### Паттерн 4: Parallel Audit

- Agent A: judge ответы / review код (часть 1)
- Agent B: judge ответы / review код (часть 2, independent)
- Merge: сравнить scores, resolve disagreements

## Когда НЕ параллелить

- Failures связаны (fix одного может починить другой)
- Агенты будут редактировать одни файлы
- Exploratory debugging (непонятно что сломано)
- Нужно полное системное понимание
- Integration-файл (один агент на всю интеграцию)

## Craft Prompt

### Полный шаблон (для implementation/analysis)

```markdown
## Задача: [название]

### Scope
- Файлы: [точные пути]
- НЕ трогать: [что read-only]

### Context
- [минимальный необходимый контекст]
- Reference: [working example]

### Goal
- [measurable результат]

### Constraints
- [ограничения]

### Success criteria
- [как понять что задача выполнена]

### Known non-goals
- [что явно НЕ нужно делать]

### Output format
- [diff / summary / scores / etc.]
```

### Mini-шаблон (для review / one-file fix / targeted debug)

```markdown
## Задача: [название]
Файлы: [пути]
Goal: [что сделать]
Output: [формат ответа]
```

Для маленьких задач полный шаблон избыточен — достаточно mini.

## Merge Protocol

1. **Read summaries** — что каждый агент сделал/нашёл
2. **Check conflicts** — трогали ли одни файлы?
3. **Check assumptions vs spec** — результат соответствует требованиям?
4. **Validate** — syntax, JSON, integration (для runtime-кода: rebuild + targeted retest; для docs/spec: review only)
5. **Decision**:
   - `merge now` — всё ок
   - `merge with known debt` — есть minor issues, зафиксировать и двигаться
   - `needs one more fix cycle` — critical issue, ещё один раунд

Full eval — только если затронут critical path или changed assumptions. Не для каждого merge.

## Типичные ошибки

- **Слишком широкий scope**: "почини все тесты" → агент теряется. Лучше: "почини тесты в файле X"
- **Нет context**: "почини race condition" → агент не знает где. Лучше: paste error messages
- **Нет constraints**: агент может зарефакторить всё. Лучше: "НЕ трогай production код"
- **Нет формата output**: "почини" → непонятно что изменилось. Лучше: "верни summary + diff"
- **Shared files**: два агента редактируют один файл → конфликты
