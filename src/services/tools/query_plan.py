"""
Инструмент query_plan для планирования запросов
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.settings import Settings
from services.query_planner_service import QueryPlannerService

logger = logging.getLogger(__name__)


def query_plan(
    query: str,
    settings: Settings | None = None,
    query_planner: QueryPlannerService | None = None,
) -> dict[str, Any]:
    """
    Создает план поиска для заданного запроса

    Args:
        query: Текст запроса пользователя
        settings: Настройки приложения
        query_planner: Сервис планировщика запросов

    Returns:
        {
            "plan": {
                "normalized_queries": ["query1", "query2"],
                "must_phrases": ["phrase1"],
                "should_phrases": ["phrase2"],
                "metadata_filters": {...},
                "k_per_query": 10,
                "fusion": "rrf"
            },
            "cache_hit": false
        }
    """
    if not query_planner:
        logger.warning("query_plan tool: query_planner not provided")
        return {"plan": None, "error": "QueryPlannerService not provided"}

    if not query or not query.strip():
        return {"plan": None, "error": "Empty query"}

    try:
        # Создаем план поиска
        start_ts = time.perf_counter()
        search_plan = query_planner.make_plan(query)
        took_ms = int((time.perf_counter() - start_ts) * 1000)
        logger.debug(
            "query_plan tool success | took_ms=%s | normalized_queries=%s | filters=%s",
            took_ms,
            search_plan.normalized_queries,
            (
                search_plan.metadata_filters.dict(exclude_none=True)
                if search_plan.metadata_filters
                else {}
            ),
        )

        # Преобразуем в словарь
        plan_dict = {
            "normalized_queries": search_plan.normalized_queries,
            "must_phrases": search_plan.must_phrases,
            "should_phrases": search_plan.should_phrases,
            "k_per_query": search_plan.k_per_query,
            "fusion": search_plan.fusion,
            "strategy": search_plan.strategy,
        }

        # Добавляем фильтры если есть
        if search_plan.metadata_filters:
            plan_dict["metadata_filters"] = {
                "channel_usernames": search_plan.metadata_filters.channel_usernames,
                "channel_ids": search_plan.metadata_filters.channel_ids,
                "date_from": search_plan.metadata_filters.date_from,
                "date_to": search_plan.metadata_filters.date_to,
                "min_views": search_plan.metadata_filters.min_views,
                "reply_to": search_plan.metadata_filters.reply_to,
            }

        return {
            "plan": plan_dict,
            "cache_hit": False,  # Для простоты не отслеживаем кеш
        }

    except Exception as e:  # broad: tool execution safety
        logger.error("Error in query_plan tool: %s", e)
        return {"plan": None, "error": str(e)}
