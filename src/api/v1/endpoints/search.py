"""
Search endpoints: добавление маршрутов плана и выполнения поиска с Query Planner
"""

import logging
from typing import Optional, List, Dict, Tuple
from fastapi import APIRouter, Depends, HTTPException, status

from core.deps import (
    get_retriever,
    get_redis_client,
    get_query_planner,
    get_reranker,
    get_hybrid_retriever,
)
from core.settings import get_settings, Settings
from adapters.chroma.retriever import Retriever
from schemas.search import (
    SearchPlanRequest,
    SearchPlan,
    SearchRequest,
    SearchResponse,
)
from utils.ranking import rrf_merge, mmr_select, _get_item_id
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_from_cache(redis_client, cache_key: str) -> Optional[dict]:
    """Получить результат из кеша"""
    if not redis_client:
        return None
    try:
        import json

        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Найден кеш для поиска: {cache_key[:50]}...")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Ошибка чтения кеша: {e}")
    return None


async def save_to_cache(redis_client, cache_key: str, data: dict, ttl: int):
    """Сохранить результат в кеш"""
    if not redis_client:
        return
    try:
        import json

        redis_client.setex(
            cache_key, ttl, json.dumps(data, ensure_ascii=False, default=str)
        )
        logger.info(f"Результат поиска сохранен в кеш: {cache_key[:50]}...")
    except Exception as e:
        logger.warning(f"Ошибка записи в кеш: {e}")


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
    retriever: Retriever = Depends(get_retriever),
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
            cached_result = await get_from_cache(redis_client, cache_key)
            if cached_result:
                return SearchResponse(**cached_result)

        # Планирование и поиск
        if settings.enable_query_planner:
            plan = planner.make_plan(request.query)
            # Если включен гибрид и он доступен — используем единый гибридный список
            if settings.hybrid_enabled and hybrid is not None:
                try:
                    candidates = hybrid.search_with_plan(request.query, plan)
                    merged_items: List[Dict] = [
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
            results_for_fusion: List[List[Tuple[str, float, Dict]]] = []
            items_by_id: Dict[str, Dict] = {}
            if not merged_items:
                for q in plan.normalized_queries:
                    items = retriever.search(
                        q,
                        k=plan.k_per_query,
                        filters=(
                            plan.metadata_filters.dict(exclude_none=True)
                            if plan.metadata_filters
                            else None
                        ),
                    )
                    triples: List[Tuple[str, float, Dict]] = []
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
            final_items: List[Dict] = merged_items
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
                docs_embs: List[np.ndarray] = []
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
            items = retriever.search(
                request.query, k=settings.search_k_per_query_default, filters=None
            )
            response = SearchResponse(
                documents=[it["text"] for it in items],
                distances=[float(it.get("distance", 0.0)) for it in items],
                metadatas=[it.get("metadata", {}) for it in items],
                plan=None,
            )

        # Сохраняем в кеш
        if settings.redis_enabled:
            cache_data = response.dict()
            await save_to_cache(redis_client, cache_key, cache_data, settings.cache_ttl)

        return response

    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при выполнении поиска: {str(e)}",
        )
