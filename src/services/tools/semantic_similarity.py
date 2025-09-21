"""
Semantic similarity tool.
Считает косинусную близость для пары или множества текстов через эмбеддинги Retriever.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

import numpy as np
from adapters.chroma.retriever import Retriever


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def semantic_similarity(
    texts: List[str],
    retriever: Retriever,
    pairs: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    if not texts or len(texts) < 2:
        return {"pairs": [], "matrix": [], "count": len(texts or [])}

    vecs = retriever.embed_texts(texts)
    arr = np.stack(vecs, axis=0)

    if pairs:
        results = []
        for a, b in pairs:
            if a < 0 or b < 0 or a >= len(texts) or b >= len(texts):
                score = 0.0
            else:
                score = _cosine(arr[a], arr[b])
            results.append({"a": a, "b": b, "score": float(score)})
        return {"pairs": results, "matrix": [], "count": len(texts)}

    # Полная матрица
    n = len(texts)
    mat = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1.0
        for j in range(i + 1, n):
            s = _cosine(arr[i], arr[j])
            mat[i][j] = float(s)
            mat[j][i] = float(s)

    return {"pairs": [], "matrix": mat, "count": n}
