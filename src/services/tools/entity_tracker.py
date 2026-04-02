"""
Tool: entity_tracker — аналитика AI/ML сущностей.
Qdrant Facet API. Point-level counts.
SPEC-RAG-15.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from qdrant_client import models

logger = logging.getLogger(__name__)

# --- Entity normalization: alias → canonical ---
def _find_datasets_file(filename: str) -> Path:
    """Найти файл в datasets/ поднимаясь от __file__ (работает и в контейнере и локально)."""
    base = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = base / "datasets" / filename
        if candidate.exists():
            return candidate
        base = base.parent
    return Path("datasets") / filename  # fallback: cwd


_ENTITY_DICT_PATH = _find_datasets_file("entity_dictionary.json")
_alias_map: dict[str, str] | None = None


def _load_alias_map() -> dict[str, str]:
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
    """Нормализация: 'openai' → 'OpenAI', 'deepseek v3' → 'DeepSeek-V3'."""
    alias_map = _load_alias_map()
    return alias_map.get(name.lower().strip(), name)


def entity_tracker(
    mode: str = "top",
    entity: str | None = None,
    entities: list[str] | None = None,
    period_from: str | None = None,
    period_to: str | None = None,
    category: str | None = None,
    limit: int = 10,
    hybrid_retriever: Any = None,
) -> dict[str, Any]:
    """Аналитический tool: агрегации по AI/ML сущностям через Qdrant Facet API.

    Modes:
    - top: топ-N сущностей (overall или за period)
    - timeline: динамика упоминаний entity по неделям
    - compare: сравнение двух+ entities по timeline
    - co_occurrence: что упоминается вместе с entity

    Все counts — point-level (коллекция chunked).
    """
    if not hybrid_retriever:
        return {"error": "HybridRetriever not provided"}

    t0 = time.perf_counter()
    store = hybrid_retriever.store

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
        took = (time.perf_counter() - t0) * 1000
        logger.debug("entity_tracker mode=%s took=%.1fms", mode, took)


# --- Helpers ---

def _build_period_filter(
    period_from: str | None = None,
    period_to: str | None = None,
) -> models.Filter | None:
    """Фильтр по дате через DatetimeRange на поле 'date'.

    Принимает ISO dates (YYYY-MM-DD). НЕ Range на keyword year_week.
    """
    conditions: list = []
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


# --- Modes ---

def _mode_top(store, retriever, period_from, period_to, category, limit):
    """Топ-N сущностей (overall или за period/category)."""
    facet_key = {
        "org": "entity_orgs",
        "model": "entity_models",
    }.get(category, "entities")

    facet_filter = _build_period_filter(period_from, period_to)

    async def _facet():
        return await store.client.facet(
            collection_name=store.collection,
            key=facet_key,
            limit=limit,
            exact=True,
            facet_filter=facet_filter,
        )

    result = retriever.run_sync(_facet())
    data = [{"entity": h.value, "count": h.count} for h in result.hits]

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
    if period_from:
        conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(gte=period_from))
        )
    if period_to:
        conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(lte=period_to))
        )

    async def _facet():
        return await store.client.facet(
            collection_name=store.collection,
            key="year_week",
            limit=100,
            exact=True,
            facet_filter=models.Filter(must=conditions),
        )

    result = retriever.run_sync(_facet())
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

    # Truncate data для LLM history — полный ряд раздувает контекст
    # и может вызвать llama-server bad request (q22).
    # Возвращаем top-10 недель по count + первую/последнюю для range.
    data_truncated = sorted(data, key=lambda x: -x["count"])[:10]
    data_truncated.sort(key=lambda x: x["week"])  # хронологический порядок

    return {
        "summary": summary, "data": data_truncated, "entity": entity,
        "mode": "timeline", "total": total, "weeks_total": len(data),
    }


def _mode_compare(store, retriever, entities_list, period_from, period_to):
    """Сравнение нескольких entities по timeline."""
    all_timelines = {}
    for ent in entities_list:
        result = _mode_timeline(store, retriever, ent, period_from, period_to)
        all_timelines[ent] = result

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
        return await store.client.facet(
            collection_name=store.collection,
            key="entities",
            limit=limit + 1,  # +1 т.к. entity сам будет в списке
            exact=True,
            facet_filter=models.Filter(must=[
                models.FieldCondition(
                    key="entities",
                    match=models.MatchValue(value=entity),
                )
            ]),
        )

    result = retriever.run_sync(_facet())
    data = [
        {"entity": h.value, "count": h.count}
        for h in result.hits
        if h.value != entity
    ][:limit]

    top_str = ", ".join(f"{d['entity']} ({d['count']})" for d in data[:5])
    summary = f"Чаще всего с {entity} упоминаются: {top_str}"

    return {"summary": summary, "data": data, "entity": entity, "mode": "co_occurrence"}
