"""
Tool: related_posts — похожие посты через Qdrant Recommend API.
SPEC-RAG-13.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def related_posts(
    post_id: str = "",
    limit: int = 5,
    hybrid_retriever: Any = None,
) -> dict[str, Any]:
    """Находит посты семантически похожие на указанный.

    Args:
        post_id: UUID или named ID поста из результатов поиска
        limit: количество результатов
        hybrid_retriever: HybridRetriever для sync bridge
    """
    if not hybrid_retriever:
        return {"hits": [], "error": "HybridRetriever not provided"}
    if not post_id:
        return {"hits": [], "error": "post_id is required"}

    store = hybrid_retriever.store
    start = time.perf_counter()

    async def _recommend():
        from qdrant_client.models import RecommendQuery
        return await store.client.query_points(
            collection_name=store.collection,
            query=RecommendQuery(positive=[post_id]),  # type: ignore[call-arg]
            using="dense_vector",
            limit=limit,
            with_payload=True,
        )

    try:
        results = hybrid_retriever.run_sync(_recommend())
    except Exception as exc:  # broad: tool execution safety
        logger.error("related_posts failed for %s: %s", post_id, exc)
        return {"hits": [], "error": str(exc), "source_id": post_id}

    # Формат совместимый с rerank/compose_context citation pipeline
    hits = []
    for p in results.points:
        text_val = p.payload.get("text", "")
        hits.append({
            "id": str(p.id),
            "score": float(p.score) if p.score else 0.0,
            "dense_score": float(p.score) if p.score else 0.0,
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
    logger.debug("related_posts | source=%s | hits=%d | took_ms=%d",
                  post_id, len(hits), took_ms)
    return {"hits": hits, "source_id": post_id}
