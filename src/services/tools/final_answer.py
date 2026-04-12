from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def final_answer(
    *,
    answer: str,
    sources: list[int] | None = None,
    citations: list[dict[str, Any]] | None = None,
    support_check: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
    coverage: float | None = None,
    refinements: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Формирует финальный ответ с валидацией citations и observability."""
    normalized_answer = answer or ""
    cited_in_text = set(
        int(match) for match in re.findall(r"\[(\d+)\]", normalized_answer)
    )
    all_sources = sorted(set(sources or []) | cited_in_text)

    sentences = re.split(r"[.!?]\s+", normalized_answer)
    non_empty_sentences = [sentence for sentence in sentences if sentence.strip()]
    uncited = [
        sentence.strip()
        for sentence in non_empty_sentences
        if not re.search(r"\[\d+\]", sentence)
    ]
    if uncited:
        logger.info(
            "final_answer: %d/%d предложений без цитат",
            len(uncited),
            len(non_empty_sentences),
        )

    if not all_sources and normalized_answer:
        normalized_answer += (
            "\n\n⚠️ Источники не указаны. Информация может быть неточной."
        )

    payload: dict[str, Any] = {
        "answer": normalized_answer,
        "sources": all_sources,
    }

    if citations:
        # Сохраняем только основные поля citation, чтобы не раздувать SSE payload.
        norm_citations: list[dict[str, Any]] = []
        for c in citations[:50]:  # hard cap to avoid oversized frames
            if isinstance(c, dict):
                norm_citations.append(
                    {
                        "id": c.get("id"),
                        "score": c.get("score"),
                        "source": c.get("source")
                        or c.get("metadata", {}).get("source"),
                        "url": c.get("url") or c.get("metadata", {}).get("url"),
                    }
                )
        if norm_citations:
            payload["citations"] = norm_citations

    if support_check is not None:
        payload["support_check"] = support_check

    if verification is not None:
        payload["verification"] = verification

    if coverage is not None:
        payload["coverage"] = float(coverage)

    if refinements is not None:
        payload["refinements"] = int(refinements)

    # Примешиваем дополнительные поля, если они не конфликтуют с основным payload.
    if extra:
        for k, v in list(extra.items())[:20]:  # cap extras to avoid abuse
            if k not in payload:
                payload[k] = v

    return payload
