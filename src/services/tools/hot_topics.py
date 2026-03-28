"""
hot_topics — LLM tool для получения pre-computed weekly digests.

SPEC-RAG-16. Читает из auxiliary коллекции `weekly_digests`,
заполняемой weekly cron (`scripts/compute_weekly_digest.py`).
Свой QdrantClient (не через DI singleton QdrantStore).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Коллекция для weekly digests (отдельная от основной news_colbert_v2)
_DIGEST_COLLECTION = "weekly_digests"


def _get_qdrant_url() -> str:
    """URL Qdrant из settings/env."""
    return os.getenv("QDRANT_URL", "http://localhost:16333")


def _get_client():
    """Sync QdrantClient для auxiliary коллекции."""
    from qdrant_client import QdrantClient
    return QdrantClient(url=_get_qdrant_url())


def _resolve_period(period: str) -> str:
    """Resolve human-readable period в ISO week label."""
    now = datetime.utcnow()
    if period == "this_week":
        iso = now.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    elif period == "last_week":
        last = now - __import__("datetime").timedelta(days=7)
        iso = last.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    elif period == "this_month":
        return f"month:{now.strftime('%Y-%m')}"
    else:
        # Assume ISO week format: 2026-W12
        return period


def hot_topics(
    *,
    hybrid_retriever: Any = None,  # не используется, для совместимости с tool_runner
    period: str = "this_week",
    top_n: int = 5,
) -> Dict[str, Any]:
    """Возвращает pre-computed hot topics за период.

    Читает из `weekly_digests` коллекции. Для "this_month" — агрегация
    последних 4 weekly digests.
    """
    resolved = _resolve_period(period)

    try:
        client = _get_client()

        # Проверить что коллекция существует
        collections = [c.name for c in client.get_collections().collections]
        if _DIGEST_COLLECTION not in collections:
            return {"error": f"Коллекция {_DIGEST_COLLECTION} не найдена. Запустите compute_weekly_digest.py"}

        if resolved.startswith("month:"):
            # Агрегация: scroll последние 4 недели
            return _get_month_digest(client, resolved.split(":")[1], top_n)
        else:
            return _get_week_digest(client, resolved, top_n)

    except Exception as e:
        logger.error("hot_topics error: %s", e, exc_info=True)
        return {"error": f"Ошибка при получении hot topics: {str(e)}"}


def _get_week_digest(client, week_label: str, top_n: int) -> Dict[str, Any]:
    """Получить digest конкретной недели."""
    from qdrant_client import models

    results = client.scroll(
        collection_name=_DIGEST_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="period",
                match=models.MatchValue(value=week_label),
            )]
        ),
        limit=1,
        with_payload=True,
    )
    points, _ = results

    if not points:
        return {
            "period": week_label,
            "error": f"Дайджест за {week_label} не найден. Запустите compute_weekly_digest.py --week {week_label}",
        }

    payload = points[0].payload or {}
    topics = payload.get("topics", [])[:top_n]

    return {
        "period": payload.get("period", week_label),
        "date_from": payload.get("date_from"),
        "date_to": payload.get("date_to"),
        "post_count": payload.get("post_count", 0),
        "summary": payload.get("summary", ""),
        "topics": topics,
        "top_entities": payload.get("top_entities", []),
        "burst_events": payload.get("burst_events", []),
    }


def _get_month_digest(client, year_month: str, top_n: int) -> Dict[str, Any]:
    """Агрегация последних 4 weekly digests для месячного обзора."""
    from qdrant_client import models

    # Scroll all digests, filter by prefix
    results = client.scroll(
        collection_name=_DIGEST_COLLECTION,
        limit=100,
        with_payload=True,
    )
    points, _ = results

    # Filter by year-month
    year, month = year_month.split("-")
    relevant = []
    for p in points:
        period = (p.payload or {}).get("period", "")
        date_from = (p.payload or {}).get("date_from", "")
        if date_from and date_from.startswith(f"{year}-{month}"):
            relevant.append(p.payload)

    if not relevant:
        return {"period": f"month:{year_month}", "error": f"Дайджесты за {year_month} не найдены"}

    # Агрегация: объединяем topics, entities, summaries
    all_topics: Dict[str, Dict] = {}
    all_entities: Dict[str, int] = {}
    total_posts = 0
    summaries = []

    for digest in relevant:
        total_posts += digest.get("post_count", 0)
        summaries.append(digest.get("summary", ""))
        for t in digest.get("topics", []):
            key = t.get("label", "")
            if key in all_topics:
                all_topics[key]["post_count"] += t.get("post_count", 0)
                all_topics[key]["hot_score"] = max(all_topics[key]["hot_score"], t.get("hot_score", 0))
                all_topics[key]["channels"] = list(
                    set(all_topics[key]["channels"]) | set(t.get("channels", []))
                )
            else:
                all_topics[key] = dict(t)
        for e in digest.get("top_entities", []):
            all_entities[e["entity"]] = all_entities.get(e["entity"], 0) + e.get("count", 0)

    sorted_topics = sorted(all_topics.values(), key=lambda x: x.get("hot_score", 0), reverse=True)
    sorted_entities = sorted(all_entities.items(), key=lambda x: x[1], reverse=True)

    return {
        "period": f"month:{year_month}",
        "weeks": len(relevant),
        "post_count": total_posts,
        "summary": " | ".join(summaries[:4]),
        "topics": sorted_topics[:top_n],
        "top_entities": [{"entity": e, "count": c} for e, c in sorted_entities[:10]],
    }
