from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _logical_key(hit: Dict[str, Any]) -> str:
    mid = (hit.get("id") or "").strip()
    if mid:
        return f"id:{mid}"
    meta = hit.get("metadata") or {}
    url = (meta.get("url") or meta.get("link") or "").strip()
    if url:
        return f"url:{url}"
    text = (hit.get("text") or "").strip()
    return f"hash:{hash(text)}"


def dedup_diversify(
    hits: List[Dict[str, Any]], lambda_: float, k: int
) -> Dict[str, Any]:
    """Дедупликация (по id/url/лексемам) + упрощённый MMR отбор top-k.

    Входные hits: {id,text,score?,metadata,embedding?}
    Возвращает: {hits: [...]} — тот же формат, упорядоченные.
    """
    if not hits:
        return {"hits": []}

    # Дедуп по логическому ключу
    unique: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        key = _logical_key(h)
        if key not in unique:
            unique[key] = h

    uniq_hits: List[Dict[str, Any]] = list(unique.values())

    # Если нет эмбеддингов — вернём первые k как есть
    if not any("embedding" in h for h in uniq_hits):
        return {"hits": uniq_hits[:k]}

    # Подготовка для MMR
    docs = uniq_hits
    embs = [
        np.asarray(h.get("embedding")) for h in docs if h.get("embedding") is not None
    ]
    if len(embs) != len(docs):
        # Не все имеют эмбеддинги — безопасный фолбэк
        return {"hits": docs[:k]}

    # Нормализация релевантности:
    # - если есть score в [0..1], используем его;
    # - иначе, если есть distance (>0), трактуем как расстояние и нормируем в [0..1], relevance=1-norm(distance)
    relevance = np.zeros(len(docs), dtype=float)
    has_any_score = any("score" in h for h in docs)
    if has_any_score:
        vals = []
        for h in docs:
            sc = h.get("score")
            if sc is None:
                vals.append(0.0)
            else:
                vals.append(float(sc))
        rel = np.asarray(vals, dtype=float)
        if not (rel.min() >= 0.0 and rel.max() <= 1.0):
            if rel.size > 0:
                rel = (rel - rel.min()) / (rel.ptp() + 1e-9)
        relevance = rel
    else:
        vals = []
        for h in docs:
            d = h.get("distance")
            vals.append(float(d) if d is not None else 0.0)
        rel = np.asarray(vals, dtype=float)
        if rel.size > 0:
            rel = (rel - rel.min()) / (rel.ptp() + 1e-9)
        relevance = 1.0 - rel

    # Простой MMR
    selected: List[int] = []
    candidate_ids = list(range(len(docs)))
    out_k = max(0, min(k, len(docs)))
    if out_k == 0:
        return {"hits": []}

    # Первый — лучший по релевантности
    first = int(np.argmax(relevance))
    selected.append(first)
    candidate_ids.remove(first)

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    while len(selected) < out_k and candidate_ids:
        best_s = -1e9
        best_i = candidate_ids[0]
        for j in candidate_ids:
            max_sim = 0.0
            for s in selected:
                sim = cos(embs[j], embs[s])
                if sim > max_sim:
                    max_sim = sim
            s_val = lambda_ * float(relevance[j]) - (1.0 - lambda_) * max_sim
            if s_val > best_s:
                best_s = s_val
                best_i = j
        selected.append(best_i)
        candidate_ids.remove(best_i)

    ordered = [docs[i] for i in selected]
    # Гарантируем ровно k элементов, если возможно
    if len(ordered) < k:
        remaining = [d for idx, d in enumerate(docs) if idx not in selected]
        ordered.extend(remaining[: max(0, k - len(ordered))])
    return {"hits": ordered[:k]}
