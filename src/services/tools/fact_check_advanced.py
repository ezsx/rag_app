"""
Advanced fact-check tool.
Проверяет утверждение по базе знаний через Retriever и даёт вердикт с confidence.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from adapters.chroma.retriever import Retriever


_NEGATION_RE = re.compile(
    r"\b(не|not|no|never|none|никуда|никак|нигде)\b", re.IGNORECASE
)


def _lexical_overlap(a: str, b: str) -> float:
    at = {w for w in re.findall(r"\w+", a.lower()) if len(w) > 2}
    bt = {w for w in re.findall(r"\w+", b.lower()) if len(w) > 2}
    if not at or not bt:
        return 0.0
    inter = len(at & bt)
    union = len(at | bt)
    return inter / max(1, union)


def fact_check_advanced(
    claim: str,
    query: Optional[str] = None,
    k: int = 6,
    retriever: Optional[Retriever] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()

    if not claim:
        return {
            "verdict": "insufficient",
            "confidence": 0.0,
            "evidence": [],
            "took_ms": int((time.perf_counter() - started) * 1000),
        }

    # Получаем кандидатов
    items: List[Dict[str, Any]] = []
    if retriever is not None:
        try:
            q = query or claim
            items = retriever.search(q, k=k, filters=None)
        except Exception:
            items = []

    # Скоры по совпадению
    evidence: List[Dict[str, Any]] = []
    support_score = 0.0
    refute_score = 0.0

    for it in items[:k]:
        text = it.get("text", "")
        ov = _lexical_overlap(claim, text)
        has_neg = bool(_NEGATION_RE.search(text))
        dist = float(it.get("distance", 0.0))
        # Нормализуем «близость»: чем меньше distance, тем лучше
        dense_score = max(0.0, 1.0 - dist)
        score = 0.6 * ov + 0.4 * dense_score

        evidence.append(
            {
                "id": it.get("id"),
                "score": round(score, 4),
                "overlap": round(ov, 4),
                "distance": dist,
                "negation": has_neg,
                "snippet": text[:300],
                "metadata": it.get("metadata", {}),
            }
        )

        if has_neg and ov > 0.2:
            refute_score += score
        else:
            support_score += score

    # Вердикт
    if not evidence:
        verdict = "insufficient"
        confidence = 0.0
    else:
        total = support_score + refute_score + 1e-6
        support_ratio = support_score / total
        if support_ratio >= 0.65 and support_score > 0.4:
            verdict = "supported"
            confidence = min(0.95, round(support_ratio, 3))
        elif refute_score >= support_score and refute_score > 0.35:
            verdict = "refuted"
            confidence = min(0.9, round(refute_score / total, 3))
        else:
            verdict = "insufficient"
            confidence = round(max(support_score, refute_score) / total, 3)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence,
        "took_ms": int((time.perf_counter() - started) * 1000),
    }
