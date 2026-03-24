"""
Tool: summarize_channel — посты канала за период в хронологическом порядке.
Qdrant scroll с filter + order_by.
СЧИТАЕТСЯ SEARCH → инкрементит search_count.
SPEC-RAG-13.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TIME_DELTAS = {"day": 1, "week": 7, "month": 30}


def summarize_channel(
    channel: str = "",
    time_range: str = "week",
    limit: int = 20,
    hybrid_retriever=None,
) -> Dict[str, Any]:
    """Получает посты канала за период в хронологическом порядке.

    LLM суммаризует на этапе final_answer.

    Args:
        channel: имя канала (как в payload.channel)
        time_range: "day" | "week" | "month"
        limit: максимум постов
        hybrid_retriever: HybridRetriever для sync bridge
    """
    if not hybrid_retriever:
        return {"hits": [], "error": "HybridRetriever not provided"}
    if not channel:
        return {"hits": [], "error": "channel is required"}

    store = hybrid_retriever._store
    delta = _TIME_DELTAS.get(time_range, 7)
    date_from = (datetime.utcnow() - timedelta(days=delta)).isoformat()
    start = time.perf_counter()

    from qdrant_client import models

    async def _scroll():
        results, _ = await store._client.scroll(
            collection_name=store.collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(
                    key="channel",
                    match=models.MatchValue(value=channel),
                ),
                models.FieldCondition(
                    key="date",
                    range=models.DatetimeRange(gte=date_from),
                ),
            ]),
            order_by=models.OrderBy(key="date", direction="asc"),
            limit=limit * 3,  # overfetch: dedup по root_message_id уменьшит count
            with_payload=True,
            with_vectors=False,
        )
        return results

    try:
        results = hybrid_retriever._run_sync(_scroll())
    except Exception as exc:
        logger.error("summarize_channel failed for %s: %s", channel, exc)
        return {"hits": [], "error": str(exc), "channel": channel}

    # Dedup по root_message_id: один пост может быть разбит на несколько chunks.
    # Берём первый chunk каждого поста (он содержит начало текста).
    seen_roots = set()
    deduped = []
    for p in results:
        root_id = p.payload.get("root_message_id", str(p.id))
        if root_id in seen_roots:
            continue
        seen_roots.add(root_id)
        deduped.append(p)

    # Формат совместимый с citation pipeline.
    # dense_score=1.0 фиксированный — digest не scoring-based,
    # но compose_context использует dense_score для coverage.
    hits = []
    for p in deduped[:limit]:  # cap после dedup
        text_val = p.payload.get("text", "")
        hits.append({
            "id": str(p.id),
            "score": 1.0,
            "dense_score": 1.0,
            "text": text_val,
            "snippet": text_val[:200] + "..." if len(text_val) > 200 else text_val,
            "meta": {
                "channel": p.payload.get("channel"),
                "channel_id": p.payload.get("channel_id"),
                "message_id": p.payload.get("message_id"),
                "date": p.payload.get("date"),
                "url": p.payload.get("url"),
            },
        })

    took_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "summarize_channel | channel=%s | period=%s | posts=%d | took_ms=%d",
        channel, time_range, len(hits), took_ms,
    )
    return {
        "hits": hits,
        "channel": channel,
        "period": time_range,
        "post_count": len(hits),
    }
