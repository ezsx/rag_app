# Systematic Debugging Protocol

> Адаптация [superpowers/systematic-debugging](https://github.com/obra/superpowers) под rag_app.
> Подчиняется AGENTS.md как верхнеуровневому контракту. MCP/tool policy остаётся обязательным.

## Iron Law

```
НЕ ФИКСИТЬ БЕЗ ROOT CAUSE.
```

- Для **runtime/behavior bugs**: Phase 1 обязательна до любого фикса.
- Для **static code/spec bugs** (видно по коду без воспроизведения): допускается прямой fix с явным объяснением причины.

## Классификация проблемы

Перед началом — определи тип:

| Тип | Пример | Подход |
|-----|--------|--------|
| **static bug** | Неверный path к файлу, Range по keyword полю | Прямой fix + объяснение |
| **runtime bug** | LLM выбирает не тот tool, forced search не срабатывает | Полный 4-phase протокол |
| **eval mismatch** | Strict recall низкий, но ответ правильный | Проблема в dataset/metric, не в agent |

Также определи **failure domain**: orchestration / retrieval / tool / LLM generation / infra / eval.

## Когда использовать полный протокол

- Runtime bugs, behavior issues
- Eval failure где непонятен root cause
- Проблема не воспроизводится стабильно (LLM stochastic — прогони 2-3 раза)
- Уже попробовал 1 fix и не помогло
- "Быстро пофиксим и поедем дальше" — сигнал что нужен протокол

## 4 фазы

### Phase 1: Root Cause Investigation

**До любого фикса:**

1. **Прочитай ошибку полностью**
   - SSE events, логи сервисов, eval results (failure_type, metrics)

2. **Воспроизведи**
   - Для LLM-stochastic bugs: targeted rerun 2-3 раза, не только один раз

3. **Проверь что изменилось** (optional для isolated failures)
   - git diff, config changes, Docker rebuild

4. **Trace по компонентам** (для multi-component issues)
   - Определи границы компонентов в цепочке обработки
   - На каждой границе: что вошло, что вышло
   - Прогони один раз с диагностикой — найди ГДЕ ломается

5. **Trace data flow назад**
   - Начни от плохого результата
   - Иди назад по цепочке: что вернул последний шаг? что получил на вход? откуда пришло?
   - Чини в источнике, не в месте проявления симптома

### Phase 2: Pattern Analysis

1. **Найди РАБОТАЮЩИЙ аналог** — похожий запрос/tool/flow который работает. В чём разница?
2. **Прочитай reference ПОЛНОСТЬЮ** — не skim, не "примерно помню"
3. **Список КАЖДОГО отличия** — query, visible tools, LLM content, tool arguments

### Phase 3: Hypothesis

1. **Одна конкретная гипотеза** — "X — root cause потому что Y". НЕ "что-то с refusal logic"
2. **Минимальный тест** — одно изменение, один запрос, проверить
3. **Не сработало → новая гипотеза** — НЕ добавлять второй fix поверх первого

### Phase 4: Fix

1. **Воспроизведи баг** — убедись что failing case стабилен
2. **Один fix** — не "заодно поправлю ещё это"
3. **Verify** — targeted retest на конкретном вопросе/case
4. **3+ неудачных fix → СТОП**
   - Пересобери evidence и определи: code bug, eval issue, infra issue или architecture problem
   - Обсудить с пользователем прежде чем пробовать fix #4

## Выходной формат

После debugging — зафиксировать:
- **Symptom**: что наблюдали
- **Root cause**: почему происходит
- **Fix**: что изменили
- **Verification**: как проверили

## Типичные failure domains

- **Выбран не тот tool** → какие tools видны, какие keywords матчились, eviction?
- **Tool упал** → output tool'а, логи сервисов, размер payload
- **Ложный отказ** → что LLM сгенерировал, refusal markers, forced search
- **Фактически неверный ответ** → answer vs source documents, coverage, hallucination
- **Ничего не найдено** → query expansion, BM25 keywords, фильтры

## Red flags — СТОП

Если ловишь себя на:
- "Быстро поправлю и потом разберусь"
- "Попробую поменять X и посмотрю"
- "Наверное проблема в Y" (без trace)
- Предлагаю fix до того как собрал evidence

→ СТОП. Вернуться к Phase 1.
