# SPEC-RAG-15: Entity Analytics Tools (entity_tracker + arxiv_tracker)

> **Статус**: Active
> **Создан**: 2026-03-25
> **Research basis**: R17-deep §1 (entity_tracker, arxiv_tracker), R16-deep §2 (visibility, "Less is More")
> **Зависимости**: SPEC-RAG-12 (payload indexes: entities, entity_orgs, entity_models, arxiv_ids, year_week), SPEC-RAG-13 (11 tools, dynamic visibility, sync bridge pattern)
> **Review**: GPT-5.4 review 2026-03-25 — 10 issues found, fixes applied

---

## Цель

Добавить 2 аналитических LLM-visible tools в агент: **entity_tracker** (P0) и **arxiv_tracker** (P1). Оба используют Qdrant Facet API для агрегаций по проиндексированным payload-полям.

Итого после этой spec: 11 существующих + 2 новых = **13 tools**, max 5 видимых.

**Ключевое отличие от search-tools**: аналитические tools возвращают **counts/aggregations**, не документы. Они не проходят через `rerank → compose_context`, а формируют текстовый summary для прямого использования LLM в `final_answer`. Ответы на analytics-запросы не требуют citation `[1] [2]` — данные получены агрегацией, а не из конкретных постов.

**Важно: point-level counts**. Коллекция chunked (один пост может дать несколько points). Facet counts = point-level, не post-level. Все summaries и expected answers используют формулировки "упоминаний" (не "постов"), числа приблизительны.

## Контекст

### Что уже есть

- **11 LLM-visible tools**: query_plan, search, temporal_search, channel_search, cross_channel_compare, summarize_channel, list_channels, rerank, related_posts, compose_context, final_answer
- **2 системных**: verify, fetch_docs
- **Dynamic visibility** в `_get_step_tools()`: phase-based (PRE-SEARCH / POST-SEARCH / NAV-COMPLETE), keyword routing, max 5 видимых
- **Sync bridge**: `hybrid_retriever._run_sync(async_coro)` — все tools sync
- **Payload indexes** (SPEC-RAG-12): `entities` (keyword), `entity_orgs` (keyword), `entity_models` (keyword), `arxiv_ids` (keyword), `year_week` (keyword), `year_month` (keyword) — все готовы для Facet API
- **Entity dictionary**: 91 entity в 6 категориях (org/model/framework/technique/conference/tool), `datasets/entity_dictionary.json`

### Реальные данные (verified)

| Metric | Value |
|--------|-------|
| Total points | 13,088 |
| Channels | 36 |
| Weeks | 38 (2025-W27 → 2026-W12) |
| Top entities | OpenAI=1,597, Google=1,127, Claude=794, Gemini=713, Anthropic=701, GPT-5=680, NVIDIA=525 |
| Arxiv papers | ~50-70 unique IDs, max 4 mentions per paper |
| Posts/week | ~300-450 |

### Facet API (verified, <10ms)

```python
# Top entities
await client.facet("news_colbert_v2", key="entities", limit=20)
# → [FacetHit(value="OpenAI", count=1597), ...]

# Entity timeline
await client.facet("news_colbert_v2", key="year_week",
    facet_filter=models.Filter(must=[
        models.FieldCondition(key="entities", match=models.MatchValue(value="DeepSeek"))
    ]))
# → [FacetHit(value="2025-W48", count=42), FacetHit(value="2025-W49", count=38), ...]

# Co-occurrence
await client.facet("news_colbert_v2", key="entities",
    facet_filter=models.Filter(must=[
        models.FieldCondition(key="entities", match=models.MatchValue(value="NVIDIA"))
    ]))
# → [FacetHit(value="NVIDIA", count=525), FacetHit(value="OpenAI", count=89), ...]

# Top arxiv
await client.facet("news_colbert_v2", key="arxiv_ids", limit=15)
# → [FacetHit(value="2502.13266", count=4), ...]
```

### Референсная реализация

`list_channels.py` — паттерн Facet API tool:
- Sync wrapper → `hybrid_retriever._run_sync(async_coro)`
- Возвращает dict (не hits)
- Error handling: `{"error": str}`
- Логирование: `time.perf_counter()`

---

## Что добавляем

| Tool | Qdrant API | Modes | State machine | Output |
|------|-----------|-------|---------------|--------|
| `entity_tracker` | `facet()` | top, timeline, compare, co_occurrence | `analytics_done = True` | summary + data |
| `arxiv_tracker` | `facet()` + `scroll()` | top, lookup | `analytics_done = True` | summary + data/hits |

---

## Что менять

### 1. Новые файлы в `src/services/tools/`

#### `entity_tracker.py`

```python
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import models

logger = logging.getLogger(__name__)

# --- Entity normalization: alias → canonical ---
_ENTITY_DICT_PATH = Path(__file__).resolve().parent.parent.parent / "datasets" / "entity_dictionary.json"
_alias_map: Optional[Dict[str, str]] = None


def _load_alias_map() -> Dict[str, str]:
    """Загрузить маппинг alias→canonical из entity_dictionary.json."""
    global _alias_map
    if _alias_map is not None:
        return _alias_map
    _alias_map = {}
    try:
        with open(_ENTITY_DICT_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        for canonical, info in raw.items():
            _alias_map[canonical.lower()] = canonical
            for alias in info.get("aliases", []):
                _alias_map[alias.lower()] = canonical
    except Exception:
        logger.warning("entity_dictionary.json not found, normalization disabled")
        _alias_map = {}
    return _alias_map


def _normalize_entity(name: str) -> str:
    """Нормализация имени entity: 'openai' → 'OpenAI', 'deepseek v3' → 'DeepSeek-V3'."""
    alias_map = _load_alias_map()
    return alias_map.get(name.lower(), name)  # fallback: вернуть как есть


def entity_tracker(
    mode: str = "top",
    entity: Optional[str] = None,
    entities: Optional[List[str]] = None,
    period_from: Optional[str] = None,
    period_to: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
    hybrid_retriever=None,
) -> Dict[str, Any]:
    """Аналитический tool: агрегации по AI/ML сущностям через Qdrant Facet API.

    Modes:
    - top: топ-N сущностей (overall или за period)
    - timeline: динамика упоминаний entity по неделям
    - compare: сравнение двух+ entities по timeline
    - co_occurrence: что упоминается вместе с entity
    """
    t0 = time.perf_counter()
    store = hybrid_retriever._store

    # Нормализация entity names через dictionary
    if entity:
        entity = _normalize_entity(entity)
    if entities:
        entities = [_normalize_entity(e) for e in entities]

    try:
        if mode == "top":
            return _mode_top(store, hybrid_retriever, period_from, period_to,
                             category, limit)
        elif mode == "timeline":
            if not entity:
                return {"error": "timeline mode requires 'entity' parameter"}
            return _mode_timeline(store, hybrid_retriever, entity,
                                  period_from, period_to)
        elif mode == "compare":
            compare_list = entities or ([entity] if entity else [])
            if len(compare_list) < 2:
                return {"error": "compare mode requires 'entities' list with ≥2 items"}
            return _mode_compare(store, hybrid_retriever, compare_list,
                                 period_from, period_to)
        elif mode == "co_occurrence":
            if not entity:
                return {"error": "co_occurrence mode requires 'entity' parameter"}
            return _mode_co_occurrence(store, hybrid_retriever, entity, limit)
        else:
            return {"error": f"Unknown mode: {mode}. Use: top, timeline, compare, co_occurrence"}
    except Exception as exc:
        logger.exception("entity_tracker error: mode=%s", mode)
        return {"error": str(exc), "mode": mode}
    finally:
        logger.debug("entity_tracker mode=%s took %.1fms", mode,
                      (time.perf_counter() - t0) * 1000)


def _build_period_filter(
    period_from: Optional[str] = None,
    period_to: Optional[str] = None,
) -> Optional[models.Filter]:
    """Фильтр по дате через DatetimeRange на поле 'date'.

    Принимает ISO dates (YYYY-MM-DD). Поле 'date' имеет DatetimeIndex.
    НЕ используем Range на keyword поле year_week — это невалидно.
    """
    conditions = []
    if period_from:
        conditions.append(
            models.FieldCondition(
                key="date",
                range=models.DatetimeRange(gte=period_from),
            )
        )
    if period_to:
        conditions.append(
            models.FieldCondition(
                key="date",
                range=models.DatetimeRange(lte=period_to),
            )
        )
    return models.Filter(must=conditions) if conditions else None


def _mode_top(store, retriever, period_from, period_to, category, limit):
    """Топ-N сущностей (overall или за period/category)."""
    # Выбираем поле: category-specific или все entities
    facet_key = {
        "org": "entity_orgs",
        "model": "entity_models",
    }.get(category, "entities")

    facet_filter = _build_period_filter(period_from, period_to)

    async def _facet():
        return await store._client.facet(
            collection_name=store.collection,
            key=facet_key,
            limit=limit,
            exact=True,
            facet_filter=facet_filter,
        )

    result = retriever._run_sync(_facet())
    data = [{"entity": h.value, "count": h.count} for h in result.hits]

    # Текстовый summary для LLM
    period_str = ""
    if period_from or period_to:
        period_str = f" за {period_from or '...'} — {period_to or '...'}"
    cat_str = f" (категория: {category})" if category else ""

    top_str = ", ".join(f"{d['entity']} ({d['count']})" for d in data[:5])
    summary = f"Top-{len(data)} entities{period_str}{cat_str}: {top_str}"
    if len(data) > 5:
        summary += f" и ещё {len(data) - 5}"

    return {"summary": summary, "data": data, "mode": "top"}


def _mode_timeline(store, retriever, entity, period_from, period_to):
    """Динамика упоминаний entity по неделям."""
    conditions = [
        models.FieldCondition(
            key="entities",
            match=models.MatchValue(value=entity),
        )
    ]
    # Фильтр по дате через DatetimeRange (не Range на keyword year_week)
    if period_from:
        conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(gte=period_from))
        )
    if period_to:
        conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(lte=period_to))
        )

    async def _facet():
        return await store._client.facet(
            collection_name=store.collection,
            key="year_week",
            limit=100,  # все недели
            exact=True,
            facet_filter=models.Filter(must=conditions),
        )

    result = retriever._run_sync(_facet())
    data = sorted(
        [{"week": h.value, "count": h.count} for h in result.hits],
        key=lambda x: x["week"],
    )

    total = sum(d["count"] for d in data)
    if data:
        peak = max(data, key=lambda x: x["count"])
        summary = (
            f"{entity}: {total} упоминаний за {len(data)} недель. "
            f"Пик: {peak['week']} ({peak['count']}). "
            f"Диапазон: {data[0]['week']} — {data[-1]['week']}"
        )
    else:
        summary = f"{entity}: не найдено упоминаний"

    return {"summary": summary, "data": data, "entity": entity, "mode": "timeline",
            "total": total}


def _mode_compare(store, retriever, entities_list, period_from, period_to):
    """Сравнение нескольких entities по timeline."""
    all_timelines = {}
    for ent in entities_list:
        result = _mode_timeline(store, retriever, ent, period_from, period_to)
        all_timelines[ent] = result

    # Summary: сравнение total и peak
    parts = []
    for ent, tl in all_timelines.items():
        total = tl.get("total", 0)
        peak_data = tl.get("data", [])
        peak_info = ""
        if peak_data:
            peak = max(peak_data, key=lambda x: x["count"])
            peak_info = f", пик {peak['week']}"
        parts.append(f"{ent}: {total} упоминаний{peak_info}")

    summary = "Сравнение: " + "; ".join(parts)

    return {"summary": summary, "timelines": all_timelines, "mode": "compare"}


def _mode_co_occurrence(store, retriever, entity, limit):
    """Сущности, упоминаемые вместе с entity."""
    async def _facet():
        return await store._client.facet(
            collection_name=store.collection,
            key="entities",
            limit=limit + 1,  # +1 потому что entity сам будет в списке
            exact=True,
            facet_filter=models.Filter(must=[
                models.FieldCondition(
                    key="entities",
                    match=models.MatchValue(value=entity),
                )
            ]),
        )

    result = retriever._run_sync(_facet())
    # Исключаем саму entity из результатов
    data = [
        {"entity": h.value, "count": h.count}
        for h in result.hits
        if h.value != entity
    ][:limit]

    top_str = ", ".join(f"{d['entity']} ({d['count']})" for d in data[:5])
    summary = f"Чаще всего с {entity} упоминаются: {top_str}"

    return {"summary": summary, "data": data, "entity": entity, "mode": "co_occurrence"}
```

#### `arxiv_tracker.py`

```python
import logging
import time
from typing import Any, Dict, Optional

from qdrant_client import models

logger = logging.getLogger(__name__)


def arxiv_tracker(
    mode: str = "top",
    arxiv_id: Optional[str] = None,
    period_from: Optional[str] = None,
    period_to: Optional[str] = None,
    limit: int = 10,
    hybrid_retriever=None,
) -> Dict[str, Any]:
    """Аналитический tool: arxiv-статьи в корпусе через Qdrant Facet/Scroll API.

    Modes:
    - top: самые обсуждаемые papers
    - lookup: посты обсуждающие конкретную paper (возвращает hits)
    """
    t0 = time.perf_counter()
    store = hybrid_retriever._store

    try:
        if mode == "top":
            return _mode_top(store, hybrid_retriever, period_from, period_to, limit)
        elif mode == "lookup":
            if not arxiv_id:
                return {"error": "lookup mode requires 'arxiv_id' parameter"}
            return _mode_lookup(store, hybrid_retriever, arxiv_id, limit)
        else:
            return {"error": f"Unknown mode: {mode}. Use: top, lookup"}
    except Exception as exc:
        logger.exception("arxiv_tracker error: mode=%s", mode)
        return {"error": str(exc), "mode": mode}
    finally:
        logger.debug("arxiv_tracker mode=%s took %.1fms", mode,
                      (time.perf_counter() - t0) * 1000)


def _build_date_filter(period_from, period_to):
    """Фильтр по дате через DatetimeRange."""
    conditions = []
    if period_from:
        conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(gte=period_from))
        )
    if period_to:
        conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(lte=period_to))
        )
    return models.Filter(must=conditions) if conditions else None


def _mode_top(store, retriever, period_from, period_to, limit):
    """Самые обсуждаемые arxiv papers."""
    facet_filter = _build_date_filter(period_from, period_to)

    async def _facet():
        return await store._client.facet(
            collection_name=store.collection,
            key="arxiv_ids",
            limit=limit,
            exact=True,
            facet_filter=facet_filter,
        )

    result = retriever._run_sync(_facet())
    data = [{"arxiv_id": h.value, "mentions": h.count} for h in result.hits]

    top_str = ", ".join(
        f"arxiv:{d['arxiv_id']} ({d['mentions']} упом.)" for d in data[:5]
    )
    summary = f"Top-{len(data)} обсуждаемых papers: {top_str}"

    return {"summary": summary, "data": data, "mode": "top"}


def _mode_lookup(store, retriever, arxiv_id, limit):
    """Посты обсуждающие конкретную arxiv paper.

    Без period_from/period_to — lookup ищет все упоминания.
    order_by date для хронологического порядка.
    Dedup по root_message_id (один пост может иметь несколько chunks).
    """
    async def _scroll():
        results, _ = await store._client.scroll(
            collection_name=store.collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(
                    key="arxiv_ids",
                    match=models.MatchAny(any=[arxiv_id]),
                )
            ]),
            order_by=models.OrderBy(key="date", direction="asc"),
            limit=limit * 3,  # fetch extra для dedup
            with_payload=True,
            with_vectors=False,
        )
        return results

    results = retriever._run_sync(_scroll())

    # Dedup по root_message_id (chunk-level → post-level)
    seen_roots = set()
    hits = []
    for p in results:
        root_id = p.payload.get("root_message_id", str(p.id))
        if root_id in seen_roots:
            continue
        seen_roots.add(root_id)
        hits.append({
            "id": str(p.id),
            "score": 1.0,
            "dense_score": 1.0,
            "text": p.payload.get("text", ""),
            "snippet": (p.payload.get("text", ""))[:200],
            "meta": {
                "channel": p.payload.get("channel"),
                "date": p.payload.get("date"),
                "url": p.payload.get("url"),
            },
        })
        if len(hits) >= limit:
            break

    channels = list({h["meta"]["channel"] for h in hits if h["meta"].get("channel")})
    summary = (
        f"Paper arxiv:{arxiv_id} — {len(hits)} постов в {len(channels)} каналах: "
        + ", ".join(channels)
    )

    return {"summary": summary, "hits": hits, "arxiv_id": arxiv_id, "mode": "lookup"}
```

### 2. Обновить `src/services/tools/__init__.py`

Добавить импорты:
```python
from services.tools.entity_tracker import entity_tracker
from services.tools.arxiv_tracker import arxiv_tracker
```

### 3. Обновить AGENT_TOOLS в `agent_service.py`

Добавить 2 новых tool schemas после существующих 11:

```python
{
    "type": "function",
    "function": {
        "name": "entity_tracker",
        "description": (
            "Аналитика AI/ML сущностей: популярность, динамика, сравнение, связи. "
            "Используй когда спрашивают 'какие компании/модели популярны', "
            "'как менялось обсуждение X', 'что связано с X', 'сравни популярность X и Y'. "
            "Возвращает агрегации (counts), не документы."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["top", "timeline", "compare", "co_occurrence"],
                    "description": (
                        "top — топ сущностей по упоминаниям; "
                        "timeline — динамика одной сущности по неделям; "
                        "compare — сравнение 2+ сущностей; "
                        "co_occurrence — что упоминается вместе с сущностью"
                    ),
                },
                "entity": {
                    "type": "string",
                    "description": "Имя сущности (для timeline, compare, co_occurrence). Примеры: OpenAI, DeepSeek, GPT-5, NVIDIA",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Список сущностей для compare mode (≥2)",
                },
                "period_from": {
                    "type": "string",
                    "description": "Начало периода ISO date: 2025-11-01",
                },
                "period_to": {
                    "type": "string",
                    "description": "Конец периода ISO date: 2026-03-25",
                },
                "category": {
                    "type": "string",
                    "enum": ["org", "model"],
                    "description": "Фильтр по категории (только для mode=top). v1: org и model имеют отдельные payload indexes; остальные категории (framework, technique, conference, tool) доступны через общий entities без фильтра",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Количество результатов",
                },
            },
            "required": ["mode"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "arxiv_tracker",
        "description": (
            "Аналитика arxiv-статей: популярные papers, поиск обсуждений конкретной статьи. "
            "Используй когда спрашивают 'какие статьи обсуждались', "
            "'кто обсуждал paper X', 'самые популярные arxiv papers'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["top", "lookup"],
                    "description": (
                        "top — самые обсуждаемые papers; "
                        "lookup — посты обсуждающие конкретную paper"
                    ),
                },
                "arxiv_id": {
                    "type": "string",
                    "description": "ID arxiv статьи для lookup (например: 2502.13266, 1706.03762)",
                },
                "period_from": {
                    "type": "string",
                    "description": "Начало периода ISO date (только для mode=top): 2025-11-01",
                },
                "period_to": {
                    "type": "string",
                    "description": "Конец периода ISO date (только для mode=top): 2026-03-25",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                },
            },
            "required": ["mode"],
        },
    },
},
```

### 4. Обновить `_get_step_tools()` — visibility

Аналитические tools добавляются в PRE-SEARCH по keyword match:

```python
# В _get_step_tools(), секция PRE-SEARCH keyword-based:

# Entity analytics keywords
if any(kw in query_lower for kw in
       ["entity", "entities", "сущност", "trending", "тренд", "популярн",
        "сколько раз", "как часто", "динамика", "timeline", "упоминан",
        "co-occurrence", "вместе с", "связан", "топ компани", "топ модел"]):
    visible_names.add("entity_tracker")

# Arxiv keywords
if any(kw in query_lower for kw in
       ["arxiv", "paper", "статья", "статей", "исследован", "публикац"]):
    visible_names.add("arxiv_tracker")
```

**Hard cap eviction order** (deterministic, при >5 visible):
```python
# Eviction priority (убираем первыми):
_EVICTION_ORDER = [
    "arxiv_tracker",        # P1, sparse data
    "entity_tracker",       # P0 но analytics, не search
    "list_channels",        # navigation
    "summarize_channel",    # specialized search
    "search",               # generic fallback (если есть specialized)
]
# Убираем по порядку пока len(visible) <= 5
for tool_name in _EVICTION_ORDER:
    if len(visible_names) <= 5:
        break
    if tool_name in visible_names:
        visible_names.discard(tool_name)
```

Также добавить в **ANALYTICS-COMPLETE** фазу (новая, по аналогии с NAV-COMPLETE):

```python
# После NAV-COMPLETE check:
if agent_state.analytics_done and agent_state.search_count == 0:
    # ANALYTICS-COMPLETE: можно final_answer или продолжить с search
    visible_names = {"final_answer", "search", "entity_tracker", "arxiv_tracker"}
```

### 5. Обновить AgentState

Добавить поле `analytics_done`:

```python
class AgentState:
    def __init__(self) -> None:
        # ... existing fields ...
        self.analytics_done: bool = False  # entity_tracker/arxiv_tracker ответили
```

### 6. Обновить `_apply_action_state()`

Добавить обработку аналитических tools:

```python
# В _apply_action_state():
if action.tool in ("entity_tracker", "arxiv_tracker"):
    self._agent_state.analytics_done = True
    # arxiv_tracker(mode="lookup") возвращает hits — сохранить для rerank/compose
    if action.tool == "arxiv_tracker" and action.output.data.get("hits"):
        self._last_search_hits = list(action.output.data.get("hits", []))
        self._agent_state.search_count += 1  # lookup = search-like
    return
```

**Важно**: `arxiv_tracker(mode="lookup")` возвращает посты (hits) — это search-like операция. Поэтому для lookup инкрементим `search_count`, чтобы разблокировать `rerank → compose_context → final_answer`. Для всех остальных modes — только `analytics_done`.

### 7. Обновить SYSTEM_PROMPT

Добавить в секцию выбора инструментов:

```python
# В SYSTEM_PROMPT, после списка search-tools:
"""
АНАЛИТИКА (агрегации без поиска документов):
   - entity_tracker — популярность, динамика, сравнение AI/ML сущностей
     • mode=top: "какие компании/модели популярны"
     • mode=timeline: "как менялось обсуждение DeepSeek"
     • mode=compare: "сравни популярность OpenAI и Anthropic"
     • mode=co_occurrence: "что обычно упоминается с NVIDIA"
   - arxiv_tracker — arxiv-статьи в каналах
     • mode=top: "какие papers обсуждались"
     • mode=lookup: "кто обсуждал paper 2502.13266"

ПРАВИЛА ДЛЯ АНАЛИТИКИ:
- Аналитические ответы НЕ требуют цитат [1] [2] — данные получены агрегацией
- Числа приблизительны (point-level, не post-level)
- Для entity_tracker используй canonical имена: OpenAI (не openai), DeepSeek (не deepseek)
- После analytics можно сразу final_answer или продолжить search для деталей
"""
```

### 8. Регистрация в ToolRunner (`deps.py`)

```python
# После existing tool registrations:
def entity_tracker_wrapper(**kwargs):
    return entity_tracker_fn(hybrid_retriever=hybrid_retriever, **kwargs)

def arxiv_tracker_wrapper(**kwargs):
    return arxiv_tracker_fn(hybrid_retriever=hybrid_retriever, **kwargs)

tool_runner.register("entity_tracker", entity_tracker_wrapper, timeout_sec=10)
tool_runner.register("arxiv_tracker", arxiv_tracker_wrapper, timeout_sec=10)
```

### 9. Обновить forced search bypass

В `stream_agent_response()` — analytics-complete должен bypass forced search (как navigation):

```python
# В секции forced search check:
if (self._agent_state.search_count == 0
    and not self._agent_state.navigation_answered
    and not self._agent_state.analytics_done  # NEW
    and not _has_refusal_markers(content)):
    # forced search...
```

---

## Output format — детали

**Все counts = point-level** (chunked collection). Формулировки используют "упоминаний", не "постов".

### entity_tracker output

Все modes возвращают `summary` (текстовый) + `data` (structured):

```python
# mode=top
{"summary": "Top-10 entities: OpenAI (1597), Google (1127), ...", "data": [...], "mode": "top"}

# mode=timeline
{"summary": "DeepSeek: 344 упоминаний за 28 недель. Пик: 2025-W49 (42).",
 "data": [{"week": "2025-W48", "count": 38}, ...], "entity": "DeepSeek", "mode": "timeline", "total": 344}

# mode=compare
{"summary": "Сравнение: OpenAI: 1597 упоминаний, пик 2026-W10; DeepSeek: 344 упоминаний, пик 2025-W49",
 "timelines": {"OpenAI": {...}, "DeepSeek": {...}}, "mode": "compare"}

# mode=co_occurrence
{"summary": "Чаще всего с NVIDIA упоминаются: OpenAI (89), Google (67), ...",
 "data": [...], "entity": "NVIDIA", "mode": "co_occurrence"}
```

### arxiv_tracker output

```python
# mode=top
{"summary": "Top-10 обсуждаемых papers: arxiv:2502.13266 (4 упом.), ...",
 "data": [{"arxiv_id": "2502.13266", "mentions": 4}, ...], "mode": "top"}

# mode=lookup (deduped по root_message_id, ordered by date)
{"summary": "Paper arxiv:2502.13266 — 3 постов в 2 каналах: gonzo_ml, seeallochnaya",
 "hits": [...], "arxiv_id": "2502.13266", "mode": "lookup"}
```

LLM использует `summary` напрямую в `final_answer` **без citations [1][2]** — данные агрегированные. Для `arxiv_tracker(mode="lookup")` — hits проходят через стандартный citation pipeline (с citations).

---

## State machine — полная схема

```
                    ┌─────────────────────────────┐
                    │         PRE-SEARCH           │
                    │  query_plan, search,         │
                    │  temporal/channel/compare...  │
                    │  + entity_tracker (keyword)   │
                    │  + arxiv_tracker (keyword)    │
                    └──────┬───────────┬───────────┘
                           │           │
              search_count++    analytics_done=True
                           │           │
                    ┌──────▼──┐  ┌─────▼──────────────┐
                    │POST-    │  │ANALYTICS-COMPLETE   │
                    │SEARCH   │  │final_answer, search,│
                    │rerank,  │  │entity/arxiv_tracker │
                    │compose, │  └─────────────────────┘
                    │final,   │
                    │related  │         navigation_answered=True
                    └─────────┘               │
                                       ┌──────▼──────┐
                                       │NAV-COMPLETE  │
                                       │final_answer  │
                                       └──────────────┘
```

**arxiv_tracker(mode="lookup")** — особый случай: устанавливает `analytics_done=True` И `search_count++`, поэтому переходит в POST-SEARCH (hits доступны для rerank/compose).

---

## Golden dataset — обновления

### Обновить existing future_baseline вопросы

**golden_q22** (entity timeline): обновить `key_tools`, `future_tool_flag`, и `expected_answer` (упростить до того что даёт tool):
```json
{
    "id": "golden_q22",
    "key_tools": ["entity_tracker"],
    "future_tool_flag": false,
    "future_key_tools": null,
    "expected_answer": "OpenAI — одна из самых упоминаемых сущностей (~1600 упоминаний). Динамика по неделям показывает стабильно высокий интерес с пиками на фоне крупных релизов."
}
```

**golden_q23** (arxiv papers): обновить, упростить expected (tool даёт IDs + counts, не titles):
```json
{
    "id": "golden_q23",
    "key_tools": ["arxiv_tracker"],
    "future_tool_flag": false,
    "future_key_tools": null,
    "expected_answer": "В каналах упоминаются ~50-70 уникальных arxiv papers. Наиболее обсуждаемые: arxiv:2502.13266 (4 упоминания) и другие. Максимум 4 упоминания на статью."
}
```

### Новые вопросы (добавить)

```json
[
  {
    "id": "golden_q26",
    "version": "1.0",
    "query": "Какие AI-компании чаще всего упоминаются в каналах?",
    "expected_answer": "Топ AI-компаний по упоминаниям (point-level): OpenAI, Google, Anthropic, NVIDIA, DeepSeek, Meta AI, Mistral AI и другие.",
    "category": "future_baseline",
    "difficulty": "easy",
    "answerable": true,
    "expected_refusal": false,
    "refusal_reason": null,
    "key_tools": ["entity_tracker"],
    "forbidden_tools": ["search", "temporal_search", "channel_search"],
    "acceptable_alternatives": [],
    "source_post_ids": [],
    "source_channels": [],
    "future_tool_flag": false,
    "future_key_tools": null,
    "calibration": true,
    "metadata": {
      "created_at": "2026-03-25",
      "created_by": "human+claude",
      "tags": ["analytics", "entity_tracker", "top"]
    }
  },
  {
    "id": "golden_q27",
    "version": "1.0",
    "query": "Что обычно упоминается вместе с NVIDIA в каналах?",
    "expected_answer": "Вместе с NVIDIA чаще всего упоминаются другие сущности из dictionary: OpenAI, Google, Anthropic, Gemini и другие AI-компании и модели (co-occurrence из entity payload).",
    "category": "future_baseline",
    "difficulty": "medium",
    "answerable": true,
    "expected_refusal": false,
    "refusal_reason": null,
    "key_tools": ["entity_tracker"],
    "forbidden_tools": ["search", "temporal_search"],
    "acceptable_alternatives": [],
    "source_post_ids": [],
    "source_channels": [],
    "future_tool_flag": false,
    "future_key_tools": null,
    "calibration": false,
    "metadata": {
      "created_at": "2026-03-25",
      "created_by": "human+claude",
      "tags": ["analytics", "entity_tracker", "co_occurrence", "nvidia"]
    }
  },
  {
    "id": "golden_q28",
    "version": "1.0",
    "query": "Сравни популярность OpenAI и DeepSeek — кого обсуждают больше?",
    "expected_answer": "OpenAI упоминается значительно чаще DeepSeek. DeepSeek имел выраженный пик в конце ноября 2025 на фоне выхода V3. Числа — point-level counts.",
    "category": "future_baseline",
    "difficulty": "medium",
    "answerable": true,
    "expected_refusal": false,
    "refusal_reason": null,
    "key_tools": ["entity_tracker"],
    "forbidden_tools": ["search", "list_channels"],
    "acceptable_alternatives": [],
    "source_post_ids": [],
    "source_channels": [],
    "future_tool_flag": false,
    "future_key_tools": null,
    "calibration": false,
    "metadata": {
      "created_at": "2026-03-25",
      "created_by": "human+claude",
      "tags": ["analytics", "entity_tracker", "compare", "openai", "deepseek"]
    }
  },
  {
    "id": "golden_q29",
    "version": "1.0",
    "query": "Какие arxiv-статьи чаще всего цитировались в каналах?",
    "expected_answer": "Наиболее обсуждаемые arxiv papers по point-level counts: arxiv:2502.13266 и другие. Корпус содержит десятки уникальных arxiv IDs, данные sparse (max ~4 упоминания на paper).",
    "category": "future_baseline",
    "difficulty": "medium",
    "answerable": true,
    "expected_refusal": false,
    "refusal_reason": null,
    "key_tools": ["arxiv_tracker"],
    "forbidden_tools": ["search", "temporal_search"],
    "acceptable_alternatives": [],
    "source_post_ids": [],
    "source_channels": [],
    "future_tool_flag": false,
    "future_key_tools": null,
    "calibration": false,
    "metadata": {
      "created_at": "2026-03-25",
      "created_by": "human+claude",
      "tags": ["analytics", "arxiv_tracker", "top"]
    }
  },
  {
    "id": "golden_q30",
    "version": "1.0",
    "query": "Кто обсуждал arxiv paper 1706.03762?",
    "expected_answer": "Paper arxiv:1706.03762 (Attention Is All You Need) упоминалась в нескольких постах. Конкретные каналы и контекст обсуждений доступны через lookup.",
    "category": "future_baseline",
    "difficulty": "hard",
    "answerable": true,
    "expected_refusal": false,
    "refusal_reason": null,
    "key_tools": ["arxiv_tracker"],
    "forbidden_tools": ["list_channels"],
    "acceptable_alternatives": ["search"],
    "source_post_ids": [],
    "source_channels": [],
    "future_tool_flag": false,
    "future_key_tools": null,
    "calibration": false,
    "metadata": {
      "created_at": "2026-03-25",
      "created_by": "human+claude",
      "tags": ["analytics", "arxiv_tracker", "lookup", "attention"]
    }
  }
]
```

---

## Future ideas (out of scope)

- **Arxiv full-text ingest**: fetch arxiv papers, extract abstract + metadata, ingest в отдельную коллекцию → full-text semantic search по самим статьям. Потенциально мощная фича для исследовательского use-case.
- **Entity timeline visualization**: SSE-event с timeline data для фронтенда (chart.js / recharts).
- **Hot topics / topic clustering**: BERTopic-based weekly digests (R17 T2).

---

## Acceptance Criteria

1. **13 tools зарегистрированы** в AGENT_TOOLS (11 старых + 2 новых)
2. **entity_tracker(mode="top")** возвращает top-N entities с counts через Facet API
3. **entity_tracker(mode="timeline")** возвращает динамику по неделям для заданной entity
4. **entity_tracker(mode="compare")** возвращает сравнение timelines для 2+ entities
5. **entity_tracker(mode="co_occurrence")** возвращает co-occurring entities (исключая саму entity)
6. **arxiv_tracker(mode="top")** возвращает top papers по mentions
7. **arxiv_tracker(mode="lookup")** возвращает посты обсуждающие paper в формате `{"hits": [...]}`
8. **State machine**: `analytics_done` флаг → final_answer доступен без search; `arxiv_tracker(mode="lookup")` → `search_count++`
9. **Dynamic visibility**: entity_tracker/arxiv_tracker появляются в PRE-SEARCH по keyword match
10. **Forced search bypass**: analytics_done=True → не форсить search
11. **System prompt** обновлён с описанием analytics tools
12. **Все tools sync**: Qdrant вызовы через `_run_sync()` bridge
12a. **Entity normalization**: `_normalize_entity()` через dictionary (alias→canonical), "openai"→"OpenAI"
12b. **Period filter**: `DatetimeRange` на поле `date`, не `Range` на keyword `year_week`
12c. **arxiv lookup dedup**: по `root_message_id`, `order_by=date`
13. **Smoke test**: агент корректно выбирает entity_tracker для "какие компании популярны" и arxiv_tracker для "какие papers обсуждались"
14. **Golden dataset**: q22, q23 обновлены; 5 новых вопросов добавлены (q26-q30)
15. **Regression**: 10 старых eval вопросов — recall не упал более чем на 0.03 (mean of 2 runs)

---

## Чеклист реализации

### Код — tools (sync, через _run_sync bridge)
- [ ] `src/services/tools/entity_tracker.py` — 4 modes, Facet API, entity normalization
- [ ] `src/services/tools/arxiv_tracker.py` — 2 modes, Facet + Scroll, dedup по root_message_id
- [ ] Обновить `src/services/tools/__init__.py` — импорты

### Код — интеграция
- [ ] Обновить AGENT_TOOLS в `agent_service.py` (2 новых schemas)
- [ ] Добавить `analytics_done: bool` в `AgentState`
- [ ] Обновить `_get_step_tools()` — keyword routing для analytics + hard cap eviction order
- [ ] Добавить ANALYTICS-COMPLETE фазу в visibility logic
- [ ] Обновить `_apply_action_state()` — entity_tracker/arxiv_tracker handlers
- [ ] Обновить forced search bypass — check `analytics_done`
- [ ] Обновить SYSTEM_PROMPT — analytics tools + citation exception + canonical names
- [ ] Регистрация в ToolRunner (`deps.py`) — wrappers + register

### Тестирование
- [ ] Unit test: entity_tracker 4 modes — sync, валидный результат
- [ ] Unit test: entity normalization — "openai"→"OpenAI", "deepseek v3"→"DeepSeek-V3"
- [ ] Unit test: arxiv_tracker 2 modes — sync, валидный результат
- [ ] Unit test: arxiv lookup dedup — chunk-level→post-level via root_message_id
- [ ] Visibility test: analytics keywords → entity_tracker/arxiv_tracker видны
- [ ] State machine: entity_tracker → analytics_done → final_answer доступен
- [ ] State machine: arxiv_tracker(lookup) → search_count++ → rerank/compose доступны
- [ ] Forced search: analytics_done → NO forced search
- [ ] Agent e2e: 3 analytics запроса через SSE endpoint
- [ ] Regression: 10 старых eval вопросов — recall не упал более чем на 0.03 (mean of 2 runs)

### Golden dataset
- [ ] Обновить golden_q22, golden_q23 — key_tools, future_tool_flag
- [ ] Добавить golden_q26-q30 (5 новых analytics вопросов)
- [ ] Обновить `datasets/eval_golden_v1.json`

### Документация
- [ ] Обновить `always_on.md` — 13 tools, entity_tracker/arxiv_tracker
- [ ] Обновить `agent_context/modules/agent.md` — analytics phase, new tools
- [ ] Обновить `docs/architecture/05-flows/` — agent flow с analytics
- [ ] Decision log — DEC-XXXX (analytics_done pattern, arxiv lookup=search)
