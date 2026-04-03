"""
Tool: list_channels — список каналов с количеством постов.
Qdrant Facet API. Кэшируемый.
SPEC-RAG-13.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def list_channels(
    channel: str | None = None,
    sort_by: str = "count",
    hybrid_retriever: Any = None,
) -> dict[str, Any]:
    """Возвращает список каналов с количеством постов (point-level counts).

    Args:
        channel: если указан — вернуть только этот канал с count
        sort_by: "count" (по убыванию) или "name" (алфавит)
        hybrid_retriever: HybridRetriever для sync bridge
    """
    if not hybrid_retriever:
        return {"channels": [], "error": "HybridRetriever not provided"}

    store = hybrid_retriever.store
    start = time.perf_counter()

    async def _facet():
        return await store.client.facet(
            collection_name=store.collection,
            key="channel",
            limit=50,
            exact=True,
        )

    try:
        result = hybrid_retriever.run_sync(_facet())
    except Exception as exc:  # broad: tool execution safety
        logger.error("list_channels facet failed: %s", exc)
        return {"channels": [], "error": str(exc)}

    channels = [{"channel": h.value, "count": h.count} for h in result.hits]

    # Single-channel mode
    if channel:
        match = [c for c in channels if c["channel"] == channel]
        took_ms = int((time.perf_counter() - start) * 1000)
        logger.debug("list_channels | channel=%s | found=%s | took_ms=%d",
                      channel, bool(match), took_ms)
        return {"channels": match, "total": 1 if match else 0}

    if sort_by == "name":
        channels.sort(key=lambda x: x["channel"])
    else:
        channels.sort(key=lambda x: -x["count"])

    took_ms = int((time.perf_counter() - start) * 1000)
    logger.debug("list_channels | total=%d | took_ms=%d", len(channels), took_ms)
    return {"channels": channels, "total": len(channels)}
