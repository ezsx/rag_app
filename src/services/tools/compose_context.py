from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "are",
        "was",
        "from",
        "что",
        "как",
        "это",
        "для",
        "или",
        "при",
        "его",
        "её",
        "они",
        "по",
        "на",
        "в",
        "к",
        "у",
        "о",
        "из",
        "за",
        "до",
        "со",
        "не",
    }
)


def _truncate_by_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _query_term_coverage(query: str, docs: List[Dict[str, Any]]) -> float:
    """
    Считает долю значимых терминов запроса, покрытых текстами документов.

    Если значимых терминов нет, возвращается 1.0: покрывать нечего.
    """
    if not query or not docs:
        return 0.0

    tokens = [
        token.lower()
        for token in re.findall(r"\w+", query, flags=re.UNICODE)
        if len(token) >= 3 and token.lower() not in _STOP_WORDS
    ]
    if not tokens:
        return 1.0

    all_text = " ".join(str(doc.get("text", "")).lower() for doc in docs)
    covered = sum(1 for token in tokens if token in all_text)
    return covered / len(tokens)


def _compute_coverage(
    query: str,
    docs: List[Dict[str, Any]],
    relevance_threshold: float = 0.55,
    target_k: int = 5,
) -> float:
    """
    Вычисляет composite coverage metric по пяти сигналам из DEC-0018.
    """
    if not docs:
        return 0.0

    sims: List[float] = sorted(
        [float(doc.get("dense_score") or doc.get("score") or 0.0) for doc in docs],
        reverse=True,
    )
    top_k = sims[:target_k]
    max_sim = sims[0]
    mean_top_k = sum(top_k) / len(top_k)

    relevant_count = sum(1 for score in sims if score >= relevance_threshold)
    doc_count_adequacy = min(1.0, relevant_count / target_k)

    if max_sim > 0.0 and len(top_k) > 1:
        score_gap = 1.0 - (top_k[0] - top_k[-1]) / max_sim
    else:
        score_gap = 0.0

    above_threshold_ratio = relevant_count / len(sims)
    term_cov = _query_term_coverage(query, docs)

    return min(
        1.0,
        0.25 * max_sim
        + 0.20 * mean_top_k
        + 0.20 * term_cov
        + 0.15 * doc_count_adequacy
        + 0.15 * score_gap
        + 0.05 * above_threshold_ratio,
    )


def compose_context(
    docs: List[Dict[str, Any]],
    query: str = "",
    max_tokens_ctx: int = 1800,
    citation_format: str = "footnotes",
    enable_lost_in_middle_mitigation: bool = True,
) -> Dict[str, Any]:
    """Собирает контекст и цитаты. MVP: ограничение по символам.

    Входные docs: [{id,text,metadata,score?}] — предполагается отсортированный список по релевантности.
    Возвращает: {prompt: str, citations: [{id, spans?}]}.
    """
    if not docs:
        return {
            "prompt": "",
            "citations": [],
            "contexts": [],
            "citation_coverage": 0.0,
        }

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

    # Покрытие считаем по исходному набору кандидатов, а не по усечённому prompt.
    citation_coverage = _compute_coverage(query, docs)

    return {
        "prompt": prompt,
        "citations": citations,
        "contexts": contexts,
        "citation_coverage": citation_coverage,
    }
