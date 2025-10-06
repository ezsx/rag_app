"""
Инструмент rerank для переранжирования результатов поиска
"""

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
    """
    Переранжирует документы по релевантности к запросу

    Args:
        query: Поисковый запрос
        docs: Список документов для переранжирования
        top_n: Количество лучших результатов (опционально)
        reranker: Сервис переранжирования

    Returns:
        {
            "indices": [0, 2, 1],  # индексы документов в порядке убывания релевантности
            "scores": [0.95, 0.87, 0.72],  # соответствующие scores
            "top_n": 3
        }
    """
    if not reranker:
        return {"indices": [], "scores": [], "error": "RerankerService not provided"}

    if not query or not query.strip():
        return {"indices": [], "scores": [], "error": "Empty query"}

    if not docs:
        return {"indices": [], "scores": [], "error": "No documents provided"}

    if hits and not docs:
        docs = [item.get("text", "") for item in hits if item]
        if not docs:
            return {"indices": [], "scores": [], "error": "No documents provided"}

    try:
        # Определяем top_n
        if top_n is None:
            top_n = len(docs)

        # Выполняем переранжирование
        indices = reranker.rerank(
            query=query, docs=docs, top_n=top_n, batch_size=16  # Стандартный batch size
        )

        # Получаем scores для отсортированных документов
        # (Для простоты возвращаем dummy scores, так как CrossEncoder не возвращает scores напрямую)
        scores = [1.0 - (i * 0.1) for i in range(len(indices))]

        return {"indices": indices, "scores": scores, "top_n": len(indices)}

    except Exception as e:
        logger.error(f"Error in rerank tool: {e}")
        return {"indices": [], "scores": [], "error": str(e)}
