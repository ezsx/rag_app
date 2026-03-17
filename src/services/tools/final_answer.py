from __future__ import annotations

from typing import Any, Dict, List, Optional


def final_answer(
    *,
    answer: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    verification: Optional[Dict[str, Any]] = None,
    coverage: Optional[float] = None,
    refinements: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Finalize agent answer in a uniform structure.

    This tool normalizes the final payload so that upstream layers (SSE, clients)
    receive a consistent schema regardless of where the answer was produced.
    """

    payload: Dict[str, Any] = {
        "answer": answer or "",
    }

    if citations:
        # Keep only essential citation fields if provided
        norm_citations: List[Dict[str, Any]] = []
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

    if verification is not None:
        payload["verification"] = verification

    if coverage is not None:
        payload["coverage"] = float(coverage)

    if refinements is not None:
        payload["refinements"] = int(refinements)

    # Merge any extra fields (e.g., fallback, error codes) if provided
    if extra:
        for k, v in list(extra.items())[:20]:  # cap extras to avoid abuse
            if k not in payload:
                payload[k] = v

    return payload
