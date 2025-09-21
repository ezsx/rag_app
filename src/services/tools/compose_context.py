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
    enable_lost_in_middle_mitigation: bool = True,
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

    # Собираем все документы с индексами
    indexed_docs = []
    for idx, d in enumerate(docs, start=1):
        text = str(d.get("text", ""))
        remaining = max_chars - used
        if remaining <= 0:
            break
        cut = _truncate_by_chars(text, remaining)
        if not cut:
            continue
        indexed_docs.append((idx, d, cut))
        used += len(cut)

    # Lost-in-the-middle mitigation: переупорядочиваем документы
    if enable_lost_in_middle_mitigation and len(indexed_docs) > 2:
        # Стратегия: наиболее релевантные документы в начало и конец
        # Менее релевантные в середину
        reordered = []
        mid = len(indexed_docs) // 2

        # Первая половина наиболее релевантных идет в начало
        for i in range(0, mid, 2):
            if i < len(indexed_docs):
                reordered.append(indexed_docs[i])

        # Менее релевантные в середину
        middle_docs = []
        for i in range(mid, len(indexed_docs)):
            middle_docs.append(indexed_docs[i])

        # Вторая часть наиболее релевантных в конец
        for i in range(1, mid, 2):
            if i < len(indexed_docs):
                middle_docs.append(indexed_docs[i])

        reordered.extend(middle_docs)
        indexed_docs = reordered

    # Формируем финальные chunks
    for idx, d, cut in indexed_docs:
        chunks.append(f"[{idx}] {cut}")
        contexts.append(cut)
        citations.append(
            {"id": d.get("id"), "index": idx, "metadata": d.get("metadata", {})}
        )

    prompt = "\n\n".join(chunks)

    # Вычисляем citation coverage
    citation_coverage = len(citations) / len(docs) if docs else 1.0

    return {
        "prompt": prompt,
        "citations": citations,
        "contexts": contexts,
        "citation_coverage": citation_coverage,
    }
