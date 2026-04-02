"""
Инструмент search для гибридного поиска с RRF слиянием
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Union

from adapters.search.hybrid_retriever import HybridRetriever
from core.observability import observe_span
from schemas.search import MetadataFilters, SearchPlan
from services.query_signals import extract_query_signals

logger = logging.getLogger(__name__)


def search(
    queries: Union[list[str], str] | None = None,
    filters: dict[str, Any] | None = None,
    k: int = 10,
    route: str = "hybrid",
    hybrid_retriever: HybridRetriever | None = None,
    query: str | None = None,
    search_type: str | None = None,
) -> dict[str, Any]:
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

    normalized_queries: list[str] = []

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
    deduped_queries: list[str] = []
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
        # --- Adaptive retrieval: extract signals и определить strategy ---
        original_query = deduped_queries[0] if deduped_queries else ""
        signals = extract_query_signals(original_query)
        strategy = "broad"
        routing_source = "default"

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

        # Rule-based strategy routing (если LLM не передал фильтры)
        if signals.strategy_hint and signals.confidence >= 0.8:
            strategy = signals.strategy_hint
            routing_source = "rules"

            if not metadata_filters:
                metadata_filters = MetadataFilters()

            # Temporal: inject dates
            if strategy == "temporal" and signals.date_from:
                if not metadata_filters.date_from:
                    metadata_filters.date_from = signals.date_from
                if not metadata_filters.date_to:
                    metadata_filters.date_to = signals.date_to

            # Channel: inject channel filter
            if strategy == "channel" and signals.channels:
                if not metadata_filters.channel_usernames:
                    metadata_filters.channel_usernames = signals.channels

        # Entity: меняем только strategy label (для логирования и будущего routing).
        # НЕ инжектируем entity names в queries — LLM уже генерирует
        # subqueries с entities, а regex inject вытесняет разнообразие из round-robin.
        if signals.entities and (not strategy or strategy == "broad"):
            strategy = "entity"
            routing_source = "rules"

        logger.info(
            "search tool adaptive | strategy=%s | routing_source=%s | signals=%s | filters=%s",
            strategy,
            routing_source,
            {
                "hint": signals.strategy_hint,
                "confidence": signals.confidence,
                "entities": signals.entities[:3],
                "channels": signals.channels,
                "date_from": signals.date_from,
            },
            metadata_filters.dict(exclude_none=True) if metadata_filters else {},
        )

        search_plan = SearchPlan(
            normalized_queries=deduped_queries,
            metadata_filters=metadata_filters,
            k_per_query=max(1, int(k)),
            fusion="rrf",
            strategy=strategy,
        )

        candidates: list[Any] = []
        route_used = route
        hybrid_duration_ms: int | None = None

        # Выполняем гибридный поиск ПО КАЖДОМУ subquery и merge результаты.
        # Ранее искали только по первому query — остальные subqueries пропадали.
        if hybrid_retriever is not None:
            start_ts = time.perf_counter()
            # Собираем результаты от каждого subquery отдельно
            per_query_results: list[list[Any]] = []
            seen_ids: set = set()
            for q in deduped_queries:
                try:
                    sub_candidates = hybrid_retriever.search_with_plan(q, search_plan)
                    per_query_results.append(sub_candidates)
                except Exception as err:
                    logger.error("Hybrid retriever failed for query '%s': %s", q[:60], err)
            # Round-robin merge: чередуем top-1 от каждого subquery, потом top-2 и т.д.
            # Сохраняет ranking от ColBERT/RRF внутри каждого subquery.
            all_candidates: list[Any] = []
            max_len = max((len(r) for r in per_query_results), default=0)
            for rank_idx in range(max_len):
                for sub_result in per_query_results:
                    if rank_idx < len(sub_result):
                        c = sub_result[rank_idx]
                        if c.id not in seen_ids:
                            all_candidates.append(c)
                            seen_ids.add(c.id)
            candidates = all_candidates
            hybrid_duration_ms = int((time.perf_counter() - start_ts) * 1000)
            logger.debug(
                "search tool hybrid finished | took_ms=%s | results=%s | queries=%d",
                hybrid_duration_ms,
                len(candidates),
                len(deduped_queries),
            )

        # Fallback: если strategy != broad и мало результатов → retry без фильтров
        if len(candidates) < 3 and strategy != "broad" and metadata_filters:
            logger.info(
                "search tool fallback | strategy=%s → broad | candidates=%d < 3",
                strategy, len(candidates),
            )
            fallback_plan = SearchPlan(
                normalized_queries=deduped_queries,
                metadata_filters=None,
                k_per_query=max(1, int(k)),
                fusion="rrf",
                strategy="broad",
            )
            routing_source = "fallback"
            fallback_results: list[Any] = []
            for q in deduped_queries:
                try:
                    sub = hybrid_retriever.search_with_plan(q, fallback_plan)
                    fallback_results.append(sub)
                except Exception as err:
                    logger.error("Fallback search failed for '%s': %s", q[:60], err)
            # Merge fallback с existing (existing first — они более targeted)
            existing_ids = {c.id for c in candidates}
            max_len_fb = max((len(r) for r in fallback_results), default=0)
            for rank_idx in range(max_len_fb):
                for sub_result in fallback_results:
                    if rank_idx < len(sub_result):
                        c = sub_result[rank_idx]
                        if c.id not in existing_ids:
                            candidates.append(c)
                            existing_ids.add(c.id)

        if not candidates:
            logger.warning(
                "search tool returned no candidates | total_ms=%s | route=%s",
                hybrid_duration_ms,
                route_used,
            )
            return {
                "hits": [],
                "error": "No results",
                "route_used": route_used,
                "strategy": strategy,
            }

        # SPEC-RAG-20d: cap total candidates для reranker, не k_per_query.
        k_total = min(k * max(1, len(deduped_queries)), 30)
        candidates = candidates[:k_total]

        # SPEC-RAG-20d: observability — search execution summary
        with observe_span("search_execution", input={
            "queries": deduped_queries[:3],
            "strategy": strategy,
            "routing_source": routing_source,
            "k_per_query": k,
            "k_total": k_total,
        }) as _search_span:
            if _search_span:
                _search_span.update(output={
                    "candidates_total": len(candidates),
                    "queries_count": len(deduped_queries),
                    "hybrid_ms": hybrid_duration_ms,
                    "strategy": strategy,
                    "routing_source": routing_source,
                    "filters": metadata_filters.dict(exclude_none=True) if metadata_filters else None,
                })

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
                    "dense_score": float(
                        getattr(candidate, "dense_score", 0.0) or 0.0
                    ),
                    "text": text_value,
                    "snippet": (
                        text_value[:200] + "..."
                        if len(text_value) > 200
                        else text_value
                    ),
                    "meta": hit_meta,
                }
            )

        logger.debug(
            "search tool success | route_used=%s | hits=%s | time_ms=%s",
            route_used,
            len(hits),
            hybrid_duration_ms,
        )
        return {
            "hits": hits,
            "total_found": len(hits),
            "route_used": route_used,
            "strategy": strategy,
            "routing_source": routing_source,
        }

    except Exception as e:
        logger.error(f"Error in search tool: {e}")
        return {"hits": [], "error": str(e)}
