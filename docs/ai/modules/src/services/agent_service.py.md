# AgentService - ReAct агент с composite coverage guardrails

## Обзор

`AgentService` управляет ReAct циклом, SSE-событиями и детерминированной логикой после `compose_context`.
В Phase 1 он использует composite coverage metric, abort guard по `dense_score` и low-coverage disclaimer.

## AgentState

```python
class AgentState:
    coverage: float
    refinement_count: int
    max_refinements: int
    low_coverage_disclaimer: bool
```

`low_coverage_disclaimer` выставляется, если после исчерпания refinement-раундов покрытие остаётся низким.

## Что делает `_normalize_tool_params()` для `compose_context`

- игнорирует `hits`, пришедшие от LLM
- инжектирует `query` из `self._current_query`
- собирает `docs` из `self._last_search_hits`
- прокидывает `dense_score` в каждый документ, чтобы `compose_context` мог считать composite coverage

## Детеминированная логика после `compose_context`

После каждого успешного `compose_context` агент:

1. сохраняет `citation_coverage` в `agent_state.coverage`
2. эмитит SSE-событие `citations`
3. считает `max_sim` из `self._last_search_hits[*].dense_score`

### Abort guard

Если:
- `max_sim < 0.30`
- и это первый проход без refinement

то агент:
- эмитит системную мысль о недостатке релевантных документов
- не запускает refinement
- выставляет `agent_state.coverage = 0.0`
- помечает `low_coverage_disclaimer = True`

### Refinement

Если abort guard не сработал и `coverage < settings.coverage_threshold`, агент может сделать refinement search.

### Hedged disclaimer

Если:
- `coverage < 0.50`
- и `refinement_count >= settings.max_refinements`

то `agent_state.low_coverage_disclaimer = True`.

При формировании финального ответа к тексту добавляется примечание о том, что найдено ограниченное количество релевантной информации.
