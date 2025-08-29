from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np


def _get_item_id(document: str, metadata: Dict[str, Any]) -> str:
    # Пытаемся использовать стабильный идентификатор из метаданных
    candidate_keys = [
        "id",
        "message_id",
        "doc_id",
        "_id",
        "uuid",
    ]
    for key in candidate_keys:
        if isinstance(metadata, dict) and key in metadata and metadata[key] is not None:
            return f"id:{metadata[key]}"
    # Фоллбек — хэш текста
    return f"hash:{hash(document)}"


def rrf_merge(
    list_of_ranked_results: List[List[Tuple[str, float, Dict[str, Any]]]], k: int = 60
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Reciprocal Rank Fusion.

    list_of_ranked_results: список списков результатов поиска для каждого под-запроса.
      Элемент результата — кортеж (document, distance, metadata) в порядке возрастания distance
      или по убыванию релевантности (важен порядок, а не абсолютные значения).

    Возвращает слитый и отсортированный список тех же кортежей без дубликатов.
    """
    scores: Dict[str, float] = {}
    best_item_by_id: Dict[str, Tuple[str, float, Dict[str, Any]]] = {}

    for results in list_of_ranked_results:
        for rank, (doc, distance, meta) in enumerate(results):
            item_id = _get_item_id(doc, meta)
            # RRF использует только ранг; меньший rank — лучше
            score_inc = 1.0 / (k + rank + 1)
            scores[item_id] = scores.get(item_id, 0.0) + score_inc
            # Сохраняем самый «лучший» вариант представления (по минимальной дистанции)
            if item_id not in best_item_by_id or distance < best_item_by_id[item_id][1]:
                best_item_by_id[item_id] = (doc, distance, meta)

    # Сортируем по суммарному RRF скору по убыванию
    sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return [best_item_by_id[i] for i in sorted_ids]


def mmr_merge(*args, **kwargs):
    raise NotImplementedError("MMR будет реализован позднее")


def _safe_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Косинусная похожесть между двумя векторами (1D). Возвращает 0.0 если хотя бы один нулевой.
    """
    if vec_a is None or vec_b is None:
        return 0.0
    if vec_a.ndim != 1:
        vec_a = vec_a.reshape(-1)
    if vec_b.ndim != 1:
        vec_b = vec_b.reshape(-1)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def mmr_select(
    candidates: List[Dict[str, Any]],
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    lambda_: float,
    out_k: int,
) -> List[Dict[str, Any]]:
    """
    Классический Maximal Marginal Relevance (MMR) отбирает подмножество документов,
    балансируя релевантность и диверсификацию.

    Args:
        candidates: список элементов с ключами как минимум {"id", "text", "score"}.
            Допустимы дополнительные поля: "metadata", "distance", "embedding".
        query_embedding: np.ndarray формы (D,)
        doc_embeddings: np.ndarray формы (N, D) — эмбеддинги для candidates в том же порядке
        lambda_: вес релевантности (0..1)
        out_k: число элементов на выходе

    Returns:
        Список выбранных кандидатов (словарей) длины <= out_k в исходном формате.

    Raises:
        ValueError: если отсутствуют или не согласованы эмбеддинги.
    """
    if candidates is None or len(candidates) == 0:
        return []
    if doc_embeddings is None or query_embedding is None:
        raise ValueError("MMR: отсутствуют эмбеддинги для документов или запроса")

    if not isinstance(doc_embeddings, np.ndarray):
        doc_embeddings = np.asarray(doc_embeddings)
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.asarray(query_embedding)

    num_docs = doc_embeddings.shape[0]
    if num_docs != len(candidates):
        raise ValueError(
            f"MMR: число эмбеддингов ({num_docs}) не совпадает с числом кандидатов ({len(candidates)})"
        )
    if doc_embeddings.ndim != 2:
        raise ValueError("MMR: doc_embeddings должен быть двумерным массивом (N, D)")
    if query_embedding.ndim != 1:
        raise ValueError("MMR: query_embedding должен быть вектором (D,)")

    # Предрасчет релевантности: sim(query, d)
    relevance: List[float] = [
        _safe_cosine_similarity(query_embedding, doc_embeddings[i])
        for i in range(num_docs)
    ]

    selected_indices: List[int] = []
    candidate_indices: List[int] = list(range(num_docs))
    k = max(0, min(out_k, num_docs))

    # Быстрый путь: k == 0
    if k == 0:
        return []

    # Инициализация: берем лучший по релевантности
    first_idx = int(np.argmax(relevance))
    selected_indices.append(first_idx)
    candidate_indices.remove(first_idx)

    # Итеративный отбор
    while len(selected_indices) < k and candidate_indices:
        best_score = -math.inf
        best_idx: Optional[int] = None
        # Для каждого еще не выбранного кандидата считаем MMR-оценку
        for idx in candidate_indices:
            # Максимальное сходство с уже выбранными (diversity penalty)
            max_sim_to_selected = 0.0
            for s_idx in selected_indices:
                sim_ds = _safe_cosine_similarity(
                    doc_embeddings[idx], doc_embeddings[s_idx]
                )
                if sim_ds > max_sim_to_selected:
                    max_sim_to_selected = sim_ds
            score = lambda_ * relevance[idx] - (1.0 - lambda_) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]
