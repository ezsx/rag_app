"""Инструмент rerank для переранжирования результатов поиска."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from services.reranker_service import RerankerService

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    docs: List[str],
    top_n: Optional[int] = None,
    reranker: Optional[RerankerService] = None,
    hits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Переранжирует документы и возвращает реальные sigmoid-нормализованные scores."""
    if not reranker:
        return {"indices": [], "scores": [], "error": "RerankerService not provided"}

    if not query or not query.strip():
        return {"indices": [], "scores": [], "error": "Empty query"}

    if hits and not docs:
        docs = [item.get("text", "") for item in hits if item]

    if not docs:
        return {"indices": [], "scores": [], "error": "No documents provided"}

    try:
        if top_n is None:
            top_n = len(docs)

        indices, scores = reranker.rerank_with_scores(
            query=query,
            docs=docs,
            top_n=top_n,
            batch_size=16,
        )

        return {"indices": indices, "scores": scores, "top_n": len(indices)}

    except Exception as e:
        logger.error(f"Error in rerank tool: {e}")
        return {"indices": [], "scores": [], "error": str(e)}
