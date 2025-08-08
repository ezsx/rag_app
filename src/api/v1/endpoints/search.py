"""
Search endpoints: добавление маршрутов плана и выполнения поиска с Query Planner
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

from core.deps import get_retriever, get_redis_client, get_query_planner
from core.settings import get_settings, Settings
from adapters.chroma.retriever import Retriever
from schemas.search import (
    SearchPlanRequest,
    SearchPlan,
    SearchRequest,
    SearchResponse,
)
from utils.ranking import rrf_merge

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
            results_for_fusion = []
            for q in plan.normalized_queries:
                docs, dists, metas = retriever.search(
                    q,
                    k=plan.k_per_query,
                    filters=(
                        plan.metadata_filters.dict(exclude_none=True)
                        if plan.metadata_filters
                        else None
                    ),
                )
                results_for_fusion.append(list(zip(docs, dists, metas)))
            merged = rrf_merge(results_for_fusion)
            documents = [d for d, _dist, _m in merged]
            distances = [float(_dist) for _d, _dist, _m in merged]
            metadatas = [m for _d, _dist, m in merged]
            response = SearchResponse(
                documents=documents,
                distances=distances,
                metadatas=metadatas,
                plan=plan if request.plan_debug else None,
            )
        else:
            docs, dists, metas = retriever.search(
                request.query, k=settings.search_k_per_query_default, filters=None
            )
            response = SearchResponse(
                documents=docs,
                distances=[float(x) for x in dists],
                metadatas=metas,
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
