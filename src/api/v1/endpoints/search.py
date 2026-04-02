"""
Search endpoints: добавление маршрутов плана и выполнения поиска с Query Planner
"""

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from core.deps import (
    get_hybrid_retriever,
    get_query_planner,
    get_redis_client,
    get_reranker,
    get_retriever,
)
from core.settings import Settings, get_settings
from schemas.search import (
    Candidate,
    SearchPlan,
    SearchPlanRequest,
    SearchRequest,
    SearchResponse,
)
from utils.ranking import _get_item_id, mmr_select, rrf_merge

logger = logging.getLogger(__name__)
router = APIRouter()


def _normalize_search_items(search_results) -> list[dict]:
    """Нормализует разные форматы search() к списку dict items."""
    if search_results is None:
        return []

    if isinstance(search_results, tuple) and len(search_results) == 3:
        documents, distances, metadatas = search_results
        items: list[dict] = []
        for idx, doc_text in enumerate(documents):
            items.append(
                {
                    "text": doc_text,
                    "distance": distances[idx] if idx < len(distances) else 0.0,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                }
            )
        return items

    if isinstance(search_results, list):
        if not search_results:
            return []
        if isinstance(search_results[0], Candidate):
            return [
                {
                    "id": item.id,
                    "text": item.text,
                    "metadata": item.metadata,
                    "distance": 0.0,
                    "embedding": (item.metadata or {}).get("_dense_vector"),
                }
                for item in search_results
            ]
        if isinstance(search_results[0], dict):
            return search_results

    return []


from core.cache import cache_get, cache_set


@router.post("/search/plan", response_model=SearchPlan, tags=["search"])
async def build_plan(
    request: SearchPlanRequest,

    planner=Depends(get_query_planner),
    settings: Settings = Depends(get_settings),
) -> SearchPlan:
    """Возвращает только план поиска для запроса"""
    plan = planner.make_plan(request.query)
    return plan


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def semantic_search(
    request: SearchRequest,

    retriever=Depends(get_retriever),
    planner=Depends(get_query_planner),
    redis_client=Depends(get_redis_client),
    settings: Settings = Depends(get_settings),
    reranker=Depends(get_reranker),
    hybrid=Depends(get_hybrid_retriever),
) -> SearchResponse:
    """
    Выполняет поиск с Query Planner (если включен) и слиянием результатов через RRF.
    """
    try:
        logger.info(f"Получен поисковый запрос: {request.query[:100]}...")

        # Используем текущую коллекцию из настроек (без переключения)

        # Подготовка ключа кеша
        cache_key = f"search:{hash(request.query + str(request.plan_debug))}"

        # Проверяем кеш
        if settings.redis_enabled:
            cached_result = cache_get(redis_client, cache_key)
            if cached_result:
                return SearchResponse(**cached_result)

        # Планирование и поиск
        if settings.enable_query_planner:
            plan = planner.make_plan(request.query)
            # Если включен гибрид и он доступен — используем единый гибридный список
            if settings.hybrid_enabled and hybrid is not None:
                try:
                    candidates = hybrid.search_with_plan(request.query, plan)
                    merged_items: list[dict] = [
                        {
                            "id": c.id,
                            "text": c.text,
                            "metadata": c.metadata,
                            "distance": 0.0,
                        }
                        for c in candidates
                    ]
                except Exception as e:
                    logger.warning(
                        f"Hybrid retriever failed in /search, fallback to dense-only: {e}"
                    )
                    merged_items = []
            else:
                merged_items = []

            # Сбор результатов поиска для каждого подзапроса (dense‑ветка) если гибрид не сработал
            results_for_fusion: list[list[tuple[str, float, dict]]] = []
            items_by_id: dict[str, dict] = {}
            if not merged_items:
                for q in plan.normalized_queries:
                    raw_items = retriever.search(
                        q,
                        k=plan.k_per_query,
                        filters=(
                            plan.metadata_filters.dict(exclude_none=True)
                            if plan.metadata_filters
                            else None
                        ),
                    )
                    items = _normalize_search_items(raw_items)
                    triples: list[tuple[str, float, dict]] = []
                    for it in items:
                        doc = it.get("text", "")
                        dist = float(it.get("distance", 0.0))
                        meta = it.get("metadata", {})
                        triples.append((doc, dist, meta))
                        item_id = _get_item_id(doc, meta)
                        if item_id not in items_by_id:
                            items_by_id[item_id] = it
                    results_for_fusion.append(triples)

                merged = rrf_merge(results_for_fusion)

                # Преобразуем merged к полным Item через items_by_id
                merged_items = []
                for doc, dist, meta in merged:
                    item_id = _get_item_id(doc, meta)
                    item = items_by_id.get(item_id) or {
                        "id": item_id,
                        "text": doc,
                        "metadata": meta,
                        "distance": float(dist),
                    }
                    merged_items.append(item)

            # MMR (опционально)
            final_items: list[dict] = merged_items
            if settings.enable_mmr and merged_items:
                top_n = min(len(merged_items), settings.mmr_top_n)
                # Убедимся, что есть эмбеддинги для top-N
                need_indices = [
                    i for i in range(top_n) if "embedding" not in merged_items[i]
                ]
                if need_indices:
                    try:
                        embs = retriever.embed_texts(
                            [merged_items[i]["text"] for i in need_indices]
                        )
                        for j, i in enumerate(need_indices):
                            merged_items[i]["embedding"] = np.asarray(
                                embs[j], dtype=float
                            )
                    except Exception:
                        pass
                try:
                    query_emb = retriever.embed_texts([f"query: {request.query}"])[0]
                except Exception:
                    query_emb = None
                if query_emb is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Не удалось получить эмбеддинг запроса для MMR",
                    )
                docs_embs: list[np.ndarray] = []
                for it in merged_items[:top_n]:
                    emb = it.get("embedding")
                    if emb is None:
                        raise HTTPException(
                            status_code=500,
                            detail="Отсутствуют эмбеддинги документов для MMR",
                        )
                    docs_embs.append(np.asarray(emb, dtype=float))
                candidates = [
                    {
                        "id": it.get("id"),
                        "text": it.get("text"),
                        "score": float(it.get("distance", 0.0)),
                        "metadata": it.get("metadata", {}),
                    }
                    for it in merged_items[:top_n]
                ]
                selected = mmr_select(
                    candidates=candidates,
                    query_embedding=np.asarray(query_emb, dtype=float),
                    doc_embeddings=np.vstack(docs_embs),
                    lambda_=settings.mmr_lambda,
                    out_k=min(settings.mmr_output_k, len(candidates)),
                )
                id_to_item = {it.get("id"): it for it in merged_items}
                selected_ids_in_order = [c.get("id") for c in selected]
                final_items = [
                    id_to_item[i] for i in selected_ids_in_order if i in id_to_item
                ]

            # Ререйкер (опционально)
            if settings.enable_reranker and reranker and final_items:
                rerank_top_n = min(len(final_items), settings.reranker_top_n)
                order = reranker.rerank(
                    query=request.query,
                    docs=[it["text"] for it in final_items[:rerank_top_n]],
                    top_n=rerank_top_n,
                    batch_size=settings.reranker_batch_size,
                )
                reordered = [final_items[i] for i in order]
                if len(final_items) > rerank_top_n:
                    reordered.extend(final_items[rerank_top_n:])
                final_items = reordered

            # Ответ (ограничим разумно 100 элементов для API)
            final_limit = min(len(final_items), 100)
            final_items = final_items[:final_limit]
            documents = [it["text"] for it in final_items]
            distances = [float(it.get("distance", 0.0)) for it in final_items]
            metadatas = [it.get("metadata", {}) for it in final_items]
            response = SearchResponse(
                documents=documents,
                distances=distances,
                metadatas=metadatas,
                plan=plan if request.plan_debug else None,
            )
        else:
            raw_items = retriever.search(
                request.query, k=settings.search_k_per_query_default, filters=None
            )
            items = _normalize_search_items(raw_items)
            response = SearchResponse(
                documents=[it["text"] for it in items],
                distances=[float(it.get("distance", 0.0)) for it in items],
                metadatas=[it.get("metadata", {}) for it in items],
                plan=None,
            )

        # Сохраняем в кеш
        if settings.redis_enabled:
            cache_data = response.dict()
            cache_set(redis_client, cache_key, cache_data, settings.cache_ttl)

        return response

    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при выполнении поиска: {e!s}",
        )
