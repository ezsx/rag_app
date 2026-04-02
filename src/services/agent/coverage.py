"""
LANCER-inspired nugget-based coverage для retrieval sufficiency.

Вместо cosine-based эвристики — проверяем покрытие конкретных аспектов
(nuggets) запроса найденными документами.

Nuggets = subqueries из query_plan. Если query_plan не вызывался,
fallback на term coverage оригинального запроса.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "are", "was", "from",
    "что", "как", "это", "для", "или", "при", "его", "её", "они",
    "по", "на", "в", "к", "у", "о", "из", "за", "до", "со", "не",
    "ли", "бы", "же", "ещё", "уже", "вот", "тоже", "если", "когда",
})


@dataclass
class CoverageResult:
    """Результат проверки покрытия."""
    score: float  # 0.0-1.0
    total_nuggets: int
    covered_nuggets: int
    uncovered: list[str]  # nuggets без покрытия — для targeted refinement
    details: dict[str, bool] = field(default_factory=dict)  # nugget → covered


def _extract_terms(text: str) -> set[str]:
    """Извлекает значимые термины из текста (lowercase, без стоп-слов)."""
    return {
        t.lower()
        for t in re.findall(r"\w+", text, re.UNICODE)
        if len(t) >= 3 and t.lower() not in _STOP_WORDS
    }


def _nugget_covered(nugget: str, docs: list[dict[str, Any]], threshold: float = 0.5) -> bool:
    """Проверяет покрыт ли nugget хотя бы одним документом.

    Покрытие = доля значимых терминов nugget найденных в тексте документа.
    Если >= threshold — nugget покрыт.
    """
    nugget_terms = _extract_terms(nugget)
    if not nugget_terms:
        return True  # пустой nugget считаем покрытым

    for doc in docs:
        doc_text = str(doc.get("text", "")).lower()
        covered_terms = sum(1 for t in nugget_terms if t in doc_text)
        if covered_terms / len(nugget_terms) >= threshold:
            return True
    return False


def compute_nugget_coverage(
    query: str,
    docs: list[dict[str, Any]],
    nuggets: list[str] | None = None,
    nugget_threshold: float = 0.5,
) -> CoverageResult:
    """Вычисляет покрытие запроса документами через nugget decomposition.

    Args:
        query: оригинальный запрос пользователя
        docs: найденные документы [{text, metadata, ...}]
        nuggets: subqueries из query_plan (если есть)
        nugget_threshold: минимальная доля терминов nugget для покрытия

    Returns:
        CoverageResult с score, uncovered nuggets для targeted refinement
    """
    if not docs:
        return CoverageResult(
            score=0.0, total_nuggets=0, covered_nuggets=0,
            uncovered=nuggets or [query],
        )

    # Если nuggets не предоставлены — fallback: оригинальный query как единственный nugget
    effective_nuggets = nuggets if nuggets else [query]

    # Всегда добавляем оригинальный query как nugget если его нет в списке
    if nuggets and query not in nuggets:
        effective_nuggets = [query, *list(nuggets)]

    details: dict[str, bool] = {}
    uncovered: list[str] = []

    for nugget in effective_nuggets:
        covered = _nugget_covered(nugget, docs, nugget_threshold)
        details[nugget] = covered
        if not covered:
            uncovered.append(nugget)

    total = len(effective_nuggets)
    covered_count = total - len(uncovered)
    score = covered_count / total if total > 0 else 0.0

    logger.info(
        "Nugget coverage: %.2f (%d/%d covered, %d uncovered)",
        score, covered_count, total, len(uncovered),
    )

    return CoverageResult(
        score=score,
        total_nuggets=total,
        covered_nuggets=covered_count,
        uncovered=uncovered,
        details=details,
    )
