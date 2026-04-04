# SPEC-RAG-11: Adaptive Retrieval — Strategy Routing

> **Статус**: Active
> **Создан**: 2026-03-21
> **Research basis**: R13-quick, R13-deep, R14-quick, R14-deep
> **Plan**: реализовано в SPEC-RAG-11/13/15/16/17

---

## Цель

Pipeline сейчас линейный — каждый запрос идёт через один и тот же путь (broad hybrid search). Temporal, channel, entity запросы требуют разных Qdrant фильтров и стратегий.

Добавить **strategy routing** в существующий pipeline с минимальными изменениями:
- Routing встроен в query_plan (zero extra LLM calls)
- Rule-based pre-validator как deterministic safety net
- MetadataFilters уже поддерживаются — нужно лишь заполнять их осмысленно

## Контекст

**Что уже есть** (не нужно создавать):
- `MetadataFilters` в `schemas/search.py` — date_from, date_to, channel_usernames
- `_build_filter()` в `HybridRetriever` — строит Qdrant `models.Filter`
- `QueryPlannerService` с JSON schema enforcement
- Multi-query search + round-robin merge в `search.py`
- Dynamic tool visibility в `_get_step_tools()`

**Что отсутствует**:
- `strategy` field в SearchPlan — нет понятия "тип поиска"
- Rule-based extraction сигналов из query — LLM ненадёжно заполняет фильтры
- Entity name injection в queries для BM25 boost
- Tracking routing decision в AgentState

---

## Что менять

### 1. Новый файл: `src/services/query_signals.py`

Rule-based pre-validator. Deterministic extraction из query text.

```python
@dataclass
class QuerySignals:
    strategy_hint: str | None    # "temporal" | "channel" | "entity" | None
    confidence: float            # 0.0–1.0
    date_from: str | None        # ISO YYYY-MM-DD
    date_to: str | None          # ISO YYYY-MM-DD
    channels: list[str]          # ["gonzo_ml", ...]
    entities: list[str]          # ["NVIDIA", "GPT-5", ...]

def extract_query_signals(query: str) -> QuerySignals:
    """Regex-based extraction. <1ms. Вызывается ДО LLM."""
```

**Паттерны**:
- Temporal: русские/английские месяцы + год, "последняя неделя", "недавно", ISO даты
- Channel: exact match против known_channels list (36 каналов), @mentions
- Entity: крупные AI/ML brands и products (NVIDIA, GPT-5, Claude, Gemini, etc.)
- Comparison: "vs", "сравни", "отличия"

**Known channels**: список загружается из settings или хардкод 36 каналов.

### 2. Расширить `schemas/search.py` — SearchPlan

Добавить поле `strategy`:

```python
class SearchPlan(BaseModel):
    # ... existing fields ...
    strategy: Literal["broad", "temporal", "channel", "entity"] = Field(
        "broad", description="Стратегия поиска, определяемая query_plan или rule-based hints"
    )
```

### 3. Расширить `QueryPlannerService`

**JSON schema** — добавить `strategy` field:
```json
"strategy": {
    "type": "string",
    "enum": ["broad", "temporal", "channel", "entity"]
}
```

**Промпт** — добавить strategy selection rules:
```
Strategy selection rules:
- "temporal": запрос упоминает даты, месяцы, периоды, "недавно", "последний"
- "channel": запрос упоминает конкретный канал или автора
- "entity": запрос о конкретном продукте, компании, человеке, технологии
- "broad": default для общих/сравнительных/мульти-тематических запросов
```

**Rule override**: если `QuerySignals.confidence > 0.8`, strategy из rule-based validator overrides LLM strategy. Логика в `post_validate()`.

### 4. Модифицировать `search.py` — strategy dispatch

**Перед формированием SearchPlan**:
```python
# 1. Extract signals
signals = extract_query_signals(original_query)

# 2. If strategy requires entity boost — inject entity into queries
if strategy == "entity" and signals.entities:
    for entity in signals.entities:
        if entity not in deduped_queries:
            deduped_queries.insert(0, entity)  # BM25 keyword match

# 3. Build metadata_filters from strategy + signals
if strategy == "temporal" and (signals.date_from or signals.date_to):
    metadata_filters = MetadataFilters(
        date_from=signals.date_from,
        date_to=signals.date_to,
    )
elif strategy == "channel" and signals.channels:
    metadata_filters = MetadataFilters(
        channel_usernames=signals.channels,
    )
```

**Fallback**: если strategy != "broad" и результатов < 3 → retry с "broad" strategy.

### 5. Расширить `AgentState`

```python
class AgentState:
    # ... existing fields ...
    strategy: str = "broad"              # broad|temporal|channel|entity
    applied_filters: dict = {}           # что реально применили к Qdrant
    routing_source: str = "llm"          # llm|rules|fallback
```

### 6. Logging + SSE observability

В `tool_invoked` SSE event добавить:
```json
{
    "tool": "search",
    "input": {...},
    "strategy": "temporal",
    "routing_source": "rules",
    "applied_filters": {"date_from": "2026-01-01", "date_to": "2026-01-31"}
}
```

---

## Что НЕ менять

- **AGENT_TOOLS schemas** — search tool schema не меняется для LLM. Strategy routing прозрачен для модели.
- **HybridRetriever** — `_build_filter()` и `search_with_plan()` уже поддерживают MetadataFilters. Не трогаем.
- **ColBERT / RRF / channel_dedup** — ничего не меняется в retrieval pipeline.
- **SSE контракт** — новые поля в data dict (strategy, routing_source) не ломают клиентов.
- **Количество LLM tools** — остаётся 5 (query_plan, search, rerank, compose_context, final_answer). Strategy — внутренний routing, не новый tool.

---

## Файлы, затрагиваемые изменениями

| Файл | Тип изменения | Строк |
|------|---------------|-------|
| `src/services/query_signals.py` | **Новый** | ~80 |
| `src/schemas/search.py` | Добавить `strategy` в SearchPlan | ~5 |
| `src/services/query_planner_service.py` | JSON schema + prompt + rule override | ~30 |
| `src/services/tools/search.py` | Strategy dispatch + entity boost + fallback | ~40 |
| `src/services/agent_service.py` | AgentState fields + logging | ~10 |

**Итого**: ~165 строк нового/изменённого кода. 1 новый файл.

---

## Acceptance Criteria

### Функциональные

1. **Temporal routing**: запрос "Что нового в январе 2026" → strategy=temporal, Qdrant date_from=2026-01-01, date_to=2026-01-31
2. **Channel routing**: запрос "Что писал gonzo_ml про трансформеры" → strategy=channel, filter channel=gonzo_ml
3. **Entity routing**: запрос "NVIDIA Vera Rubin анонс" → strategy=entity, "NVIDIA" и "Vera Rubin" injected в queries
4. **Broad fallback**: запрос "Сравни GPT-5 и Claude" → strategy=broad, без фильтров
5. **Fallback chain**: если temporal search возвращает <3 результатов → automatic retry с broad

### Regression

6. **v1 recall ≥ 0.76**: broad queries не деградируют
7. **v2 recall ≥ 0.61**: существующие запросы не ломаются
8. **SSE контракт**: Web UI продолжает работать

### Метрики (target)

9. **v2 Q6** (NVIDIA 2026): recall > 0 (сейчас 0.0 — LLM не знает "Vera Rubin")
10. **Routing accuracy**: ≥80% correct strategy на 20 test queries

---

## Test Plan

1. **Unit tests** для `extract_query_signals()`:
   - "Что было в январе 2026" → temporal, date_from=2026-01-01
   - "Что писал gonzo_ml" → channel, channels=["gonzo_ml"]
   - "NVIDIA Vera Rubin" → entity, entities=["NVIDIA", "Vera Rubin"]
   - "Сравни GPT-5 и Claude" → broad (или entity с low confidence)

2. **Single query verification** (перед full eval):
   - Temporal query через API → проверить что Qdrant получил date filter
   - Channel query → проверить channel filter
   - Broad query → проверить что фильтры не применяются

3. **Full eval**: v1 (10 Qs) + v2 (10 Qs) → recall не ниже baseline

4. **Routing eval**: 20 queries с expected_strategy → accuracy ≥80%
