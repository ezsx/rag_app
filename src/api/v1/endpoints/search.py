"""
Search endpoints: план и выполнение поиска с Query Planner.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from core.cache import cache_get, cache_set
from core.deps import (
    get_hybrid_retriever,
    get_query_planner,
    get_redis_client,
    get_reranker,
)
from core.settings import Settings, get_settings
from schemas.search import (
    SearchPlan,
    SearchPlanRequest,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


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
    planner=Depends(get_query_planner),
    redis_client=Depends(get_redis_client),
    settings: Settings = Depends(get_settings),
    reranker=Depends(get_reranker),
    hybrid=Depends(get_hybrid_retriever),
) -> SearchResponse:
    """Выполняет hybrid search с Query Planner (если включен) и опциональным rerank."""
    try:
        logger.info("Получен поисковый запрос: %s...", request.query[:100])

        cache_key = f"search:{hash(request.query + str(request.plan_debug))}"

        if settings.redis_enabled:
            cached_result = cache_get(redis_client, cache_key)
            if cached_result:
                return SearchResponse(**cached_result)

        # Построить план поиска
        if settings.enable_query_planner:
            plan = planner.make_plan(request.query)
        else:
            plan = SearchPlan(
                normalized_queries=[request.query],
                k_per_query=settings.search_k_per_query_default,
                fusion="rrf",
            )

        # search_with_plan делает BM25+Dense → RRF → ColBERT rerank внутри
        candidates = hybrid.search_with_plan(request.query, plan)
        final_items: list[dict] = [
            {
                "id": c.id,
                "text": c.text,
                "metadata": c.metadata,
                "distance": 0.0,
            }
            for c in candidates
        ]

        # Reranker (опционально)
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

        # Ограничиваем ответ (100 элементов max для API)
        final_limit = min(len(final_items), 100)
        final_items = final_items[:final_limit]

        response = SearchResponse(
            documents=[it["text"] for it in final_items],
            distances=[float(it.get("distance", 0.0)) for it in final_items],
            metadatas=[it.get("metadata", {}) for it in final_items],
            plan=plan if request.plan_debug else None,
        )

        if settings.redis_enabled:
            cache_data = response.dict()
            cache_set(redis_client, cache_key, cache_data, settings.cache_ttl)

        return response

    except Exception as e:
        logger.error("Ошибка при поиске: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при выполнении поиска: {e!s}",
        )
