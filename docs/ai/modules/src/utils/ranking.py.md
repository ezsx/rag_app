### Модуль: `src/utils/ranking.py`

Назначение: алгоритмы слияния и отбора результатов: RRF и MMR.

#### Функции
- `_get_item_id(document, metadata)` — устойчивый идентификатор по метаданным либо hash.
- `rrf_merge(list_of_ranked_results, k=60)` — слияние списков по Reciprocal Rank Fusion.
- `_safe_cosine_similarity(a, b)` — косинусная похожесть с защитой на нулевые векторы.
- `mmr_select(candidates, query_embedding, doc_embeddings, lambda_, out_k)` — классический MMR‑отбор.





