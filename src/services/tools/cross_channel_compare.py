"""
Tool: cross_channel_compare — как разные каналы обсуждают одну тему.
Qdrant query_points_groups с prefetch + RRF fusion.
СЧИТАЕТСЯ SEARCH → инкрементит search_count.
SPEC-RAG-13.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def cross_channel_compare(
    topic: str = "",
    date_from: str | None = None,
    date_to: str | None = None,
    max_channels: int = 10,
    posts_per_channel: int = 2,
    hybrid_retriever: Any = None,
) -> dict[str, Any]:
    """Ищет как разные каналы обсуждают одну тему.

    Использует Qdrant query_points_groups с prefetch + RRF fusion.
    Возвращает grouped results + flat hits для citation pipeline.

    Args:
        topic: тема для сравнения
        date_from: начало периода ISO YYYY-MM-DD
        date_to: конец периода ISO YYYY-MM-DD
        hybrid_retriever: HybridRetriever для sync bridge + embedding
    """
    if not hybrid_retriever:
        return {"hits": [], "error": "HybridRetriever not provided"}
    if not topic:
        return {"hits": [], "error": "topic is required"}

    store = hybrid_retriever.store
    start = time.perf_counter()

    # Embed topic: async через sync bridge.
    # Важно: coroutine создаётся внутри async context, не снаружи.
    async def _embed():
        return await hybrid_retriever.embedding_client.embed_query(topic)
    dense_vector = hybrid_retriever.run_sync(_embed())
    # Sparse: fastembed query_embed (не embed — для query-side encoding)
    sparse = next(iter(hybrid_retriever.sparse_encoder.query_embed(topic)))

    # Фильтр по дате
    from qdrant_client import models

    filter_conditions: list[Any] = []
    if date_from:
        filter_conditions.append(
            models.FieldCondition(
                key="date", range=models.DatetimeRange(gte=date_from)
            )
        )
    if date_to:
        filter_conditions.append(
            models.FieldCondition(
                key="date", range=models.DatetimeRange(lte=date_to)
            )
        )
    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    async def _grouped_search():
        return await store.client.query_points_groups(
            collection_name=store.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=store.DENSE_VECTOR,
                    limit=100,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                    using=store.SPARSE_VECTOR,
                    limit=100,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            group_by="channel",
            limit=max_channels,
            group_size=posts_per_channel * 3,  # overfetch: dedup уменьшит
            query_filter=query_filter,
            with_payload=True,
        )

    try:
        results = hybrid_retriever.run_sync(_grouped_search())
    except Exception as exc:  # broad: tool execution safety
        logger.error("cross_channel_compare failed: %s", exc)
        return {"hits": [], "error": str(exc), "topic": topic}

    # Формат совместимый с citation pipeline.
    # Dedup по root_message_id внутри каждой группы.
    # dense_score=1.0 фиксированный — RRF/group score не калиброван
    # для coverage metric, поэтому не используем как dense_score.
    all_hits: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for group in results.groups:
        posts = []
        seen_roots: set = set()
        for p in group.hits:
            root_id = p.payload.get("root_message_id", str(p.id))
            if root_id in seen_roots:
                continue
            seen_roots.add(root_id)
            text_val = p.payload.get("text", "")
            hit = {
                "id": str(p.id),
                "score": float(p.score) if p.score else 0.0,
                "dense_score": 1.0,  # фиксированный — RRF score не калиброван
                "text": text_val,
                "snippet": text_val[:200] + "..." if len(text_val) > 200 else text_val,
                "meta": {
                    "channel": p.payload.get("channel"),
                    "channel_id": p.payload.get("channel_id"),
                    "message_id": p.payload.get("message_id"),
                    "date": p.payload.get("date"),
                    "url": p.payload.get("url"),
                },
            }
            posts.append(hit)
            all_hits.append(hit)
            if len(posts) >= posts_per_channel:
                break  # cap после dedup
        groups.append({"channel": group.id, "posts": posts})

    took_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "cross_channel_compare | topic=%s | channels=%d | hits=%d | took_ms=%d",
        topic[:60], len(groups), len(all_hits), took_ms,
    )
    return {
        "hits": all_hits,
        "groups": groups,
        "topic": topic,
        "channels_found": len(groups),
        "total_found": len(all_hits),
    }
