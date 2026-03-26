# Задача: review новых workflow протоколов

## Контекст

Добавлены два новых модуля в `agent_context/modules/`:
- `debugging_protocol.md` — systematic debugging (адаптация superpowers framework)
- `parallel_agents.md` — parallel agent dispatch для Claude + Codex workflow

Эти протоколы определяют как мы (Claude + Codex) работаем над проектом. Не runtime код — а правила совместной разработки.

## Что нужно

### 1. Прочитай оба файла
- `agent_context/modules/debugging_protocol.md`
- `agent_context/modules/parallel_agents.md`

### 2. Оцени применимость к нашему workflow

Вспомни как мы работали над SPEC-RAG-15:
- Claude писал code, ты делал review (3 раунда)
- Ты нашёл CRITICAL issues (path to dictionary, verify bypass, year_week Range)
- Claude фиксил, ты re-review
- Ты расширял keywords и policies
- Ты делал independent eval judge (28/30 agreement)
- Ты делал failure analysis (q01/q03/q19/q22/q25)

Вопросы:
1. Debugging protocol — согласен с 4 фазами? Что бы добавил/убрал? Есть ли случаи когда протокол мешает?
2. Parallel agents — 4 паттерна описаны. Какой мы чаще всего используем? Какие missing?
3. Merge protocol — 5 шагов. Реалистично? Мы так делаем?
4. Prompt template — полезен или избыточен?
5. Есть ли конфликты с существующими правилами в AGENTS.md или CLAUDE.md?

### 3. Предложи улучшения

Если видишь gaps или проблемы — конкретные предложения что изменить.

### 4. Подтверди или возрази

Финальный вердикт: принимаешь ли ты эти протоколы как рабочие правила? Если нет — что нужно изменить чтобы принял?

## Формат ответа

Для каждого протокола:
```
### [Protocol name]
**Verdict**: accept / accept with changes / reject
**Что хорошо**: ...
**Что изменить**: ...
**Что добавить**: ...
```
