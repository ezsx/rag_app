from typing import Dict, List, Tuple, Any


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
