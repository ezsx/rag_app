"""
channel_expertise — LLM tool для pre-computed channel profiles.

SPEC-RAG-16. Читает из auxiliary коллекции `channel_profiles`,
заполняемой monthly cron (`scripts/compute_channel_profiles.py`).
Свой QdrantClient (не через DI singleton QdrantStore).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_PROFILES_COLLECTION = "channel_profiles"


def _get_qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:16333")


def _get_embedding_url() -> str:
    return os.getenv("EMBEDDING_TEI_URL", "http://localhost:8082")


def _get_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=_get_qdrant_url())


def _embed_query(text: str) -> list[float]:
    """Embed query через gpu_server для vector search."""
    payload = json.dumps({"inputs": [text]}).encode()
    req = urllib.request.Request(
        f"{_get_embedding_url()}/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data[0]


def channel_expertise(
    *,
    hybrid_retriever: Any = None,  # не используется
    channel: str | None = None,
    topic: str | None = None,
    metric: str = "authority",
) -> dict[str, Any]:
    """Возвращает channel profiles / ranking / topic-based search.

    Modes:
    - channel=X: профиль конкретного канала
    - topic=Y: каналы-эксперты по теме (vector search)
    - ни то ни другое: ranking по metric
    """
    try:
        client = _get_client()

        collections = [c.name for c in client.get_collections().collections]
        if _PROFILES_COLLECTION not in collections:
            return {"error": f"Коллекция {_PROFILES_COLLECTION} не найдена. Запустите compute_channel_profiles.py"}

        if channel:
            return _get_channel_profile(client, channel)
        elif topic:
            return _search_by_topic(client, topic, metric)
        else:
            return _get_ranking(client, metric)

    except Exception as e:
        logger.error("channel_expertise error: %s", e, exc_info=True)
        return {"error": f"Ошибка: {e!s}"}


def _get_channel_profile(client, channel: str) -> dict[str, Any]:
    """Получить профиль конкретного канала."""
    from qdrant_client import models

    # Поиск по payload field channel
    results = client.scroll(
        collection_name=_PROFILES_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="channel",
                match=models.MatchValue(value=channel.lstrip("@")),
            )]
        ),
        limit=1,
        with_payload=True,
    )
    points, _ = results

    if not points:
        return {"channel": channel, "error": f"Профиль канала {channel} не найден"}

    payload = points[0].payload or {}
    return {
        "channel": payload.get("channel", channel),
        "display_name": payload.get("display_name", ""),
        "total_posts": payload.get("total_posts", 0),
        "post_frequency": payload.get("post_frequency", {}),
        "top_entities": payload.get("top_entities", [])[:10],
        "top_topics": payload.get("top_topics", [])[:5],
        "authority_score": payload.get("authority_score", 0),
        "speed_score": payload.get("speed_score", 0),
        "breadth_score": payload.get("breadth_score", 0),
        "profile_summary": payload.get("profile_summary", ""),
    }


def _search_by_topic(client, topic: str, metric: str, top_n: int = 10) -> dict[str, Any]:
    """Vector search по profile summaries → каналы-эксперты по теме."""
    query_vector = _embed_query(topic)

    results = client.query_points(
        collection_name=_PROFILES_COLLECTION,
        query=query_vector,
        limit=top_n,
        with_payload=True,
    )

    channels = []
    for point in results.points:
        p = point.payload or {}
        channels.append({
            "channel": p.get("channel", ""),
            "display_name": p.get("display_name", ""),
            metric + "_score": p.get(f"{metric}_score", 0),
            "relevance": round(point.score, 3) if hasattr(point, "score") else 0,
            "top_entities": p.get("top_entities", [])[:5],
            "profile_summary": (p.get("profile_summary", "") or "")[:200],
        })

    # Sort by requested metric
    metric_key = f"{metric}_score"
    channels.sort(key=lambda x: x.get(metric_key, 0), reverse=True)

    return {
        "topic": topic,
        "metric": metric,
        "channels": channels,
    }


def _get_ranking(client, metric: str, top_n: int = 15) -> dict[str, Any]:
    """Ranking всех каналов по metric."""
    results = client.scroll(
        collection_name=_PROFILES_COLLECTION,
        limit=100,
        with_payload=True,
    )
    points, _ = results

    channels = []
    metric_key = f"{metric}_score"
    for p in points:
        payload = p.payload or {}
        channels.append({
            "channel": payload.get("channel", ""),
            "display_name": payload.get("display_name", ""),
            metric_key: payload.get(metric_key, 0),
            "total_posts": payload.get("total_posts", 0),
        })

    channels.sort(key=lambda x: x.get(metric_key, 0), reverse=True)

    return {
        "metric": metric,
        "channels": channels[:top_n],
    }
