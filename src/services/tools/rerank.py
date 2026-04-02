"""Инструмент rerank для переранжирования результатов поиска."""

from __future__ import annotations

import logging
from typing import Any

from services.reranker_service import RerankerService

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    docs: list[str],
    top_n: int | None = None,
    reranker: RerankerService | None = None,
    hits: list[dict[str, Any]] | None = None,
    filter_threshold: float = 0.0,
) -> dict[str, Any]:
    """CRAG-style: ранжирует + фильтрует документы по relevance score.

    Документы с score < filter_threshold помечаются как нерелевантные
    и не попадают в compose_context. ColBERT порядок сохраняется,
    cross-encoder только отсекает мусор.
    """
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

        # CRAG-style filtering: отсекаем документы с низким CE score
        filtered_indices = []
        filtered_scores = []
        filtered_out = 0
        for idx, score in zip(indices, scores):
            if score >= filter_threshold:
                filtered_indices.append(idx)
                filtered_scores.append(score)
            else:
                filtered_out += 1

        if filtered_out > 0:
            logger.info(
                "Rerank filter: %d/%d docs removed (score < %.2f)",
                filtered_out, len(indices), filter_threshold,
            )

        return {
            "indices": filtered_indices,
            "scores": filtered_scores,
            "top_n": len(filtered_indices),
            "filtered_out": filtered_out,
        }

    except Exception as e:
        logger.error(f"Error in rerank tool: {e}")
        return {"indices": [], "scores": [], "error": str(e)}
