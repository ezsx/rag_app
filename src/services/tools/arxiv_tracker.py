"""
Tool: arxiv_tracker — аналитика arxiv-статей в корпусе.
Qdrant Facet + Scroll API. Point-level counts.
SPEC-RAG-15.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from qdrant_client import models

logger = logging.getLogger(__name__)


def arxiv_tracker(
    mode: str = "top",
    arxiv_id: str | None = None,
    period_from: str | None = None,
    period_to: str | None = None,
    limit: int = 10,
    hybrid_retriever=None,
) -> dict[str, Any]:
    """Аналитический tool: arxiv-статьи в корпусе через Qdrant Facet/Scroll API.

    Modes:
    - top: самые обсуждаемые papers (point-level counts)
    - lookup: посты обсуждающие конкретную paper (deduped по root_message_id)

    period_from/period_to — только для mode=top (ISO date YYYY-MM-DD).
    """
    if not hybrid_retriever:
        return {"error": "HybridRetriever not provided"}

    t0 = time.perf_counter()
    store = hybrid_retriever.store

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
        took = (time.perf_counter() - t0) * 1000
        logger.debug("arxiv_tracker mode=%s took=%.1fms", mode, took)


def _mode_top(store, retriever, period_from, period_to, limit):
    """Самые обсуждаемые arxiv papers."""
    conditions: list = []
    if period_from:
        conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(gte=period_from))
        )
    if period_to:
        conditions.append(
            models.FieldCondition(key="date", range=models.DatetimeRange(lte=period_to))
        )
    facet_filter = models.Filter(must=conditions) if conditions else None

    async def _facet():
        return await store.client.facet(
            collection_name=store.collection,
            key="arxiv_ids",
            limit=limit,
            exact=True,
            facet_filter=facet_filter,
        )

    result = retriever.run_sync(_facet())
    data = [{"arxiv_id": h.value, "mentions": h.count} for h in result.hits]

    top_str = ", ".join(
        f"arxiv:{d['arxiv_id']} ({d['mentions']} упом.)" for d in data[:5]
    )
    summary = f"Top-{len(data)} обсуждаемых papers: {top_str}"

    return {"summary": summary, "data": data, "mode": "top"}


def _mode_lookup(store, retriever, arxiv_id, limit):
    """Посты обсуждающие конкретную arxiv paper.

    order_by date, dedup по root_message_id (chunk→post).
    Возвращает hits для citation pipeline.
    """
    async def _scroll():
        results, _ = await store.client.scroll(
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

    results = retriever.run_sync(_scroll())

    # Dedup по root_message_id (chunk-level → post-level)
    seen_roots: set = set()
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
