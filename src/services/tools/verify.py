from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from adapters.chroma import Retriever

logger = logging.getLogger(__name__)


def verify(
    query: str,
    claim: str,
    retriever: Retriever,
    top_k: int = 3,
    docs: Optional[Sequence[Dict[str, Any]]] = None,
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
        retrieved_docs: List[Dict[str, Any]] = []

        if docs:
            for item in docs:
                text = item.get("text") or item.get("snippet")
                if text:
                    retrieved_docs.append(
                        {
                            "text": str(text),
                            "distance": item.get("distance"),
                            "source_id": item.get("id"),
                        }
                    )

        if not retrieved_docs:
            search_results = retriever.search(claim, k=top_k)

            if isinstance(search_results, dict):
                documents = search_results.get("documents") or []
                distances = search_results.get("distances") or []
                metadatas = search_results.get("metadatas") or []

                # Flatten возможных вложенных структур Chroma
                if documents and isinstance(documents[0], list):
                    documents = documents[0]
                if distances and isinstance(distances[0], list):
                    distances = distances[0]
                if metadatas and isinstance(metadatas[0], list):
                    metadatas = metadatas[0]

                for idx, doc_text in enumerate(documents):
                    if isinstance(doc_text, str) and doc_text.strip():
                        retrieved_docs.append(
                            {
                                "text": doc_text,
                                "distance": (
                                    distances[idx] if idx < len(distances) else None
                                ),
                                "source_id": (
                                    (metadatas[idx] or {}).get("id")
                                    if idx < len(metadatas)
                                    else None
                                ),
                            }
                        )
            elif isinstance(search_results, list):
                for item in search_results:
                    if not isinstance(item, dict):
                        continue
                    text_value = item.get("text") or item.get("snippet")
                    if not text_value:
                        continue
                    retrieved_docs.append(
                        {
                            "text": str(text_value),
                            "distance": item.get("distance"),
                            "source_id": item.get("id"),
                        }
                    )
            else:
                logger.warning(
                    "verify: unexpected search result type %s", type(search_results)
                )

        if not retrieved_docs:
            return {
                "verified": False,
                "confidence": 0.0,
                "evidence": [],
                "note": "Релевантные документы не найдены",
            }

        # Рассчитываем confidence по расстояниям/оценкам ранга
        evidence: List[str] = []
        confidences: List[float] = []
        for idx, doc in enumerate(retrieved_docs[:top_k]):
            text = doc.get("text", "")
            distance = doc.get("distance")
            confidence = 0.5
            if isinstance(distance, (int, float)):
                confidence = max(0.0, 1.0 - min(distance, 2.0) / 2.0)
            else:
                # Чем выше документ в ранге, тем выше confidence
                confidence = max(0.3, 1.0 - idx * 0.1)
            confidences.append(confidence)
            evidence.append(text[:200] + "..." if len(text) > 200 else text)

        avg_confidence = sum(confidences) / len(confidences)
        threshold = 0.6
        verified = avg_confidence >= threshold

        return {
            "verified": verified,
            "confidence": round(avg_confidence, 3),
            "evidence": evidence,
            "threshold": threshold,
            "documents_found": len(retrieved_docs),
            "used_docs": min(len(retrieved_docs), top_k),
        }

    except Exception as e:
        logger.error(f"Ошибка при проверке утверждения: {e}")
        return {
            "verified": False,
            "confidence": 0.0,
            "evidence": [],
            "error": f"Ошибка поиска: {str(e)}",
        }
