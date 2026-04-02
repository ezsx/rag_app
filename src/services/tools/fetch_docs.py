"""Инструмент fetch_docs — батч-выгрузка документов по ID из Qdrant."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from adapters.qdrant.store import QdrantStore

logger = logging.getLogger(__name__)


def fetch_docs(
    qdrant_store: QdrantStore,
    ids: list[str] | None = None,
    window: list[int] | None = None,
    doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Батч-выгрузка документов по IDs из Qdrant."""
    final_ids: list[str] = ids or doc_ids or []
    if not final_ids:
        logger.debug("fetch_docs вызван без ids")
        return {"docs": []}

    try:
        records = asyncio.run(qdrant_store.get_by_ids(final_ids))
        docs = []
        for record in records:
            payload = record.payload or {}
            docs.append(
                {
                    "id": str(record.id),
                    "text": payload.get("text", ""),
                    "metadata": {
                        "channel": payload.get("channel"),
                        "message_id": payload.get("message_id"),
                        "date": payload.get("date"),
                        "author": payload.get("author"),
                        "url": payload.get("url"),
                    },
                }
            )
        logger.debug("fetch_docs: %d документов для %d ids", len(docs), len(final_ids))
        return {"docs": docs}

    except Exception as exc:
        logger.error("fetch_docs ошибка для ids=%s: %s", final_ids, exc)
        return {
            "docs": [{"id": _id, "text": "", "metadata": {}} for _id in final_ids]
        }
