from __future__ import annotations

import re
from typing import Any, Dict, List


_NUMERIC_RE = re.compile(r"\b\d+[\d\-:\./]*\b")
_DATE_RE = re.compile(
    r"\b(\d{4}[\-/]\d{1,2}[\-/]\d{1,2}|\d{1,2}[\./]\d{1,2}[\./]\d{2,4})\b"
)
_ENTITY_HINT_RE = re.compile(r"[@#][\w_]+|https?://\S+", re.IGNORECASE)


def router_select(query: str) -> Dict[str, Any]:
    """Простая эвристика выбора маршрута {bm25|dense|hybrid}.

    - короткие, терминальные, числовые, даты → bm25
    - разговорные/длинные/семантические → dense
    - явные фильтры, смешанные признаки, сомнение → hybrid
    Возвращает: {route: str, reasons: [str]}
    """
    text = (query or "").strip()
    reasons: List[str] = []
    route = "dense"

    length = len(text)
    has_number = bool(_NUMERIC_RE.search(text))
    has_date = bool(_DATE_RE.search(text))
    has_entity = bool(_ENTITY_HINT_RE.search(text))

    # Простые признаки «терминальности» запроса
    is_short = length <= 40
    is_very_short = length <= 16

    # Грубая эвристика на признаки диалога
    conversational_hints = [
        "почему",
        "как",
        "объясни",
        "explain",
        "what is",
        "how to",
        "покажи",
    ]
    is_conversational = any(h in text.lower() for h in conversational_hints)

    # Грубая эвристика фильтров (канал/дата)
    filter_hints = ["channel:", "date:", "after:", "before:", "канал:"]
    has_filters = any(h in text.lower() for h in filter_hints)

    # Правила
    if is_very_short or (is_short and (has_number or has_date or has_entity)):
        route = "bm25"
        reasons.append("short_or_entity_numeric")
    elif has_filters or (has_number and is_conversational):
        route = "hybrid"
        reasons.append("filters_or_mixed_signals")
    elif is_conversational or length > 120:
        route = "dense"
        reasons.append("conversational_or_long")
    else:
        # Смешанные признаки → hybrid как безопасный дефолт
        if has_number or has_date or has_entity:
            route = "hybrid"
            reasons.append("mixed_signals")
        else:
            route = "dense"
            reasons.append("default_dense")

    return {"route": route, "reasons": reasons or ["rule_based"]}
