"""
Инструмент search для гибридного поиска с RRF слиянием
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from schemas.search import SearchPlan, MetadataFilters
from adapters.search.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


def search(
    queries: Optional[Union[List[str], str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    k: int = 10,
    route: str = "hybrid",
    hybrid_retriever: Optional[HybridRetriever] = None,
    query: Optional[str] = None,
    search_type: Optional[str] = None,
    bm25_retriever: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Выполняет гибридный поиск по коллекции документов с RRF слиянием

    Args:
        queries: Список запросов для поиска
        filters: Фильтры метаданных (date_from, date_to, channel)
        k: Количество результатов
        route: Маршрут поиска (bm25|dense|hybrid)
        hybrid_retriever: HybridRetriever для выполнения поиска

    Returns:
        {
            "hits": [
                {
                    "id": "string",
                    "score": 11.5,
                    "snippet": "string",
                    "meta": {
                        "channel_id":"string",
                        "channel":"string",
                        "message_id":"string",
                        "date":"YYYY-MM-DDTHH:MM:SSZ",
                        "author":"string",
                        "is_forward":true,
                        "reply_to":"string|null",
                        "links":["..."],
                        "media_types":["..."],
                        "lang":"ru|en",
                        "hash":"hex"
                    }
                }
            ]
        }
    """
    if not hybrid_retriever:
        return {"hits": [], "error": "HybridRetriever not provided"}

    normalized_queries: List[str] = []

    if query and str(query).strip():
        normalized_queries.append(str(query).strip())

    if queries:
        if isinstance(queries, str):
            normalized_queries.append(queries.strip())
        elif isinstance(queries, Sequence):
            normalized_queries.extend(
                [str(item).strip() for item in queries if str(item).strip()]
            )

    # Deduplicate while preserving order
    deduped_queries: List[str] = []
    seen = set()
    for q in normalized_queries:
        if q and q not in seen:
            deduped_queries.append(q)
            seen.add(q)

    if not deduped_queries:
        return {"hits": [], "error": "No queries provided"}

    logger.debug(
        "search tool start | route=%s | search_type=%s | queries=%s | raw_filters_keys=%s",
        route,
        search_type,
        deduped_queries,
        list(filters.keys()) if isinstance(filters, dict) else None,
    )

    try:
        metadata_filters = None
        if filters:
            channel_value = filters.get("channel")
            channel_usernames = None
            if isinstance(channel_value, str) and channel_value.strip():
                channel_usernames = [channel_value.strip()]
            elif isinstance(channel_value, list):
                channel_usernames = [
                    str(x).strip() for x in channel_value if str(x).strip()
                ]

            metadata_filters = MetadataFilters(
                date_from=filters.get("date_from"),
                date_to=filters.get("date_to"),
                channel_usernames=channel_usernames,
                channel_ids=filters.get("channel_ids"),
                min_views=filters.get("min_views"),
                reply_to=filters.get("reply_to"),
            )

        logger.debug(
            "search tool filters | raw=%s | metadata=%s",
            filters,
            metadata_filters.dict(exclude_none=True) if metadata_filters else {},
        )

        search_plan = SearchPlan(
            normalized_queries=deduped_queries,
            metadata_filters=metadata_filters,
            k_per_query=max(1, int(k)),
            fusion="rrf",
        )

        candidates: List[Any] = []
        route_used = route
        hybrid_duration_ms: Optional[int] = None
        bm25_duration_ms: Optional[int] = None

        # Выполняем гибридный поиск, если доступен retriever
        if hybrid_retriever is not None:
            start_ts = time.perf_counter()
            try:
                candidates = hybrid_retriever.search_with_plan(
                    deduped_queries[0], search_plan
                )
            except AttributeError as attr_err:
                logger.error(
                    "Hybrid retriever failed with AttributeError: %s", attr_err
                )
                candidates = []
            except Exception as hybrid_err:
                logger.error("Hybrid retriever failed: %s", hybrid_err)
                candidates = []
            hybrid_duration_ms = int((time.perf_counter() - start_ts) * 1000)
            logger.debug(
                "search tool hybrid finished | took_ms=%s | results=%s",
                hybrid_duration_ms,
                len(candidates) if candidates else 0,
            )

        # Фолбэк на чистый BM25 при ошибках гибридного поиска или явном запросе
        force_bm25 = (route or "").lower() == "bm25" or (
            search_type and search_type.lower() == "bm25"
        )
        if (not candidates or force_bm25) and bm25_retriever is not None:
            start_ts = time.perf_counter()
            try:
                candidates = bm25_retriever.search(deduped_queries[0], search_plan, k)
                route_used = "bm25"
            except Exception as bm25_err:
                logger.error("BM25 fallback failed: %s", bm25_err)
                candidates = []
            bm25_duration_ms = int((time.perf_counter() - start_ts) * 1000)
            logger.debug(
                "search tool bm25 finished | took_ms=%s | results=%s",
                bm25_duration_ms,
                len(candidates) if candidates else 0,
            )

        if not candidates:
            total_ms = None
            if hybrid_duration_ms is not None or bm25_duration_ms is not None:
                total_ms = (hybrid_duration_ms or 0) + (bm25_duration_ms or 0)
            logger.warning(
                "search tool returned no candidates | total_ms=%s | route=%s",
                total_ms,
                route_used,
            )
            return {
                "hits": [],
                "error": "No results",
                "route_used": route_used,
            }

        # Ограничиваем количество результатов
        candidates = candidates[:k]

        # Преобразуем в формат ответа
        hits = []
        for i, candidate in enumerate(candidates):
            # Получаем метаданные
            meta = candidate.metadata or {}

            # Извлекаем поля метаданных в соответствии со спецификацией
            hit_meta = {
                "channel_id": meta.get("channel_id"),
                "channel": meta.get("channel"),
                "message_id": meta.get("message_id") or meta.get("msg_id"),
                "date": meta.get("date"),
                "author": meta.get("author"),
                "is_forward": meta.get("is_forward", False),
                "reply_to": meta.get("reply_to"),
                "links": meta.get("links", []),
                "media_types": meta.get("media_types", []),
                "lang": meta.get("lang"),
                "hash": meta.get("hash"),
            }

            # Убираем None значения
            hit_meta = {k: v for k, v in hit_meta.items() if v is not None}

            text_value = candidate.text or ""
            if not isinstance(text_value, str):
                text_value = str(text_value)

            hits.append(
                {
                    "id": candidate.id,
                    "score": float(i + 1),  # Используем rank как score
                    "text": text_value,
                    "snippet": (
                        text_value[:200] + "..."
                        if len(text_value) > 200
                        else text_value
                    ),
                    "meta": hit_meta,
                }
            )

        total_ms = None
        if hybrid_duration_ms is not None or bm25_duration_ms is not None:
            total_ms = (hybrid_duration_ms or 0) + (bm25_duration_ms or 0)
        logger.debug(
            "search tool success | route_used=%s | hits=%s | time_ms=%s",
            route_used,
            len(hits),
            total_ms,
        )
        return {"hits": hits, "total_found": len(hits), "route_used": route_used}

    except Exception as e:
        logger.error(f"Error in search tool: {e}")
        return {"hits": [], "error": str(e)}
