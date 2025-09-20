from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _truncate_by_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def compose_context(
    docs: List[Dict[str, Any]],
    max_tokens_ctx: int = 1800,
    citation_format: str = "footnotes",
) -> Dict[str, Any]:
    """Собирает контекст и цитаты. MVP: ограничение по символам.

    Входные docs: [{id,text,metadata,score?}] — предполагается отсортированный список по релевантности.
    Возвращает: {prompt: str, citations: [{id, spans?}]}.
    """
    if not docs:
        return {"prompt": "", "citations": []}

    # Примитивное соответствие символы~токены: 1 токен ≈ 4 символа
    max_chars = int(max_tokens_ctx * 4)

    chunks: List[str] = []
    citations: List[Dict[str, Any]] = []
    contexts: List[str] = []
    used = 0
    for idx, d in enumerate(docs, start=1):
        text = str(d.get("text", ""))
        remaining = max_chars - used
        if remaining <= 0:
            break
        cut = _truncate_by_chars(text, remaining)
        if not cut:
            continue
        chunks.append(f"[{idx}] {cut}")
        contexts.append(cut)
        citations.append({"id": d.get("id"), "index": idx})
        used += len(cut)

    prompt = "\n\n".join(chunks)
    return {"prompt": prompt, "citations": citations, "contexts": contexts}
