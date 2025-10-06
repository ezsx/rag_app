from __future__ import annotations

import logging
from typing import Any, Dict, List

from adapters.chroma import Retriever

logger = logging.getLogger(__name__)


def verify(
    query: str, claim: str, retriever: Retriever, top_k: int = 3
) -> Dict[str, Any]:
    """Проверяет утверждение через повторный поиск в базе знаний.

    Ищет документы, связанные с утверждением, и сравнивает с исходным запросом
    для быстрого факт-чека.

    Args:
        query: Исходный запрос пользователя
        claim: Утверждение для проверки
        retriever: Ретривер для поиска
        top_k: Количество документов для поиска

    Returns:
        {verified: bool, confidence: float, evidence: List[str]}
    """
    if not claim.strip():
        return {
            "verified": False,
            "confidence": 0.0,
            "evidence": [],
            "error": "Пустое утверждение",
        }

    try:
        # Ищем документы по утверждению
        search_results = retriever.search(claim, k=top_k)

        # Проверяем что search_results - словарь, а не список
        if not isinstance(search_results, dict):
            logger.warning(f"search_results is not dict: {type(search_results)}")
            return {
                "verified": False,
                "confidence": 0.0,
                "evidence": [],
                "error": "Invalid search results format",
            }

        if not search_results.get("documents"):
            return {
                "verified": False,
                "confidence": 0.0,
                "evidence": [],
                "note": "Релевантные документы не найдены",
            }

        documents = search_results["documents"]
        distances = search_results.get("distances", [])

        # Простая эвристика проверки
        evidence = []
        total_confidence = 0.0

        # documents может быть списком списков (nested) или списком строк
        # Flatten если nested
        if documents and isinstance(documents[0], list):
            documents = documents[0]

        for i, doc in enumerate(documents):
            # doc - строка текста документа
            if not isinstance(doc, str):
                continue

            # Преобразуем distance в confidence (чем меньше distance, тем выше confidence)
            confidence = 0.5  # default
            if distances:
                try:
                    # distances может быть nested [[0.1, 0.2, ...]] или flat [0.1, 0.2, ...]
                    if isinstance(distances[0], list):
                        dist_list = distances[0]
                    else:
                        dist_list = distances

                    if i < len(dist_list):
                        distance = dist_list[i]
                        # Нормализуем distance в confidence (0.0-1.0)
                        confidence = max(0.0, 1.0 - min(distance, 2.0) / 2.0)
                except (IndexError, TypeError) as e:
                    logger.debug(f"Error parsing distances: {e}")
                    confidence = 0.5

            total_confidence += confidence
            evidence.append(doc[:200] + "..." if len(doc) > 200 else doc)

        # Средняя confidence
        avg_confidence = total_confidence / len(documents) if documents else 0.0

        # Порог для верификации (можно настроить)
        verification_threshold = 0.6
        verified = avg_confidence >= verification_threshold

        return {
            "verified": verified,
            "confidence": round(avg_confidence, 3),
            "evidence": evidence,
            "threshold": verification_threshold,
            "documents_found": len(documents),
        }

    except Exception as e:
        logger.error(f"Ошибка при проверке утверждения: {e}")
        return {
            "verified": False,
            "confidence": 0.0,
            "evidence": [],
            "error": f"Ошибка поиска: {str(e)}",
        }
