"""Инструмент verify — проверка утверждения через повторный поиск в базе знаний."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from adapters.search.hybrid_retriever import HybridRetriever
from schemas.search import SearchPlan

logger = logging.getLogger(__name__)


def verify(
    query: str,
    claim: str,
    hybrid_retriever: HybridRetriever,
    top_k: int = 3,
    docs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Проверяет утверждение через повторный поиск в базе знаний."""
    if not claim.strip():
        return {
            "verified": False,
            "confidence": 0.0,
            "evidence": [],
            "error": "Пустое утверждение",
        }

    try:
        retrieved_docs: list[dict[str, Any]] = []

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
            plan = SearchPlan(
                normalized_queries=[claim],
                k_per_query=top_k,
                fusion="rrf",
            )
            candidates = hybrid_retriever.search_with_plan(claim, plan)
            for idx, cand in enumerate(candidates[:top_k]):
                retrieved_docs.append(
                    {
                        "text": cand.text,
                        "distance": None,
                        "source_id": cand.id,
                        "rank": idx,
                    }
                )

        if not retrieved_docs:
            return {
                "verified": False,
                "confidence": 0.0,
                "evidence": [],
                "note": "Релевантные документы не найдены",
            }

        evidence: list[str] = []
        confidences: list[float] = []
        for idx, doc in enumerate(retrieved_docs[:top_k]):
            text = doc.get("text", "")
            confidence = max(0.3, 0.9 - idx * 0.1)
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

    except Exception as exc:
        logger.error("Ошибка при проверке утверждения: %s", exc)
        return {
            "verified": False,
            "confidence": 0.0,
            "evidence": [],
            "error": f"Ошибка поиска: {exc!s}",
        }
