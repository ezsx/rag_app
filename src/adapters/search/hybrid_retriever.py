"""
HybridRetriever Phase 1 — нативный RRF через Qdrant prefetch + FusionQuery.

Заменяет Phase 0 ChromaDB + BM25 + ручной rrf_merge.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Coroutine, Optional, TypeVar

from fastembed import SparseTextEmbedding
from qdrant_client import models

from adapters.qdrant.store import QdrantStore
from adapters.tei.embedding_client import TEIEmbeddingClient
from core.settings import Settings
from schemas.search import Candidate, MetadataFilters, SearchPlan

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class HybridRetriever:
    """Qdrant-based hybrid retriever: dense (TEI) + sparse (BM25 fastembed) → native RRF.

    Синхронный публичный интерфейс совместим как с ToolRunner/ThreadPoolExecutor,
    так и с прямыми sync-вызовами из FastAPI-эндпоинтов и QAService.

    Внутренне использует выделенный event loop в фоновом потоке для безопасного
    sync→async bridge (тот же паттерн что RerankerService). asyncio.run() не подходит —
    он создаёт и закрывает loop каждый раз, что убивает httpx connection pool.
    """

    def __init__(
        self,
        store: QdrantStore,
        embedding_client: TEIEmbeddingClient,
        sparse_encoder: SparseTextEmbedding,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embedding_client = embedding_client
        self._sparse_encoder = sparse_encoder
        self._settings = settings

        # Выделенный event loop для sync→async bridge
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="hybrid-retriever-sync-bridge",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)
        logger.info("HybridRetriever инициализирован: collection=%s", store.collection)

    def _run_event_loop(self) -> None:
        """Фоновый поток с выделенным asyncio loop."""
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _run_sync(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Выполняет coroutine в фоновом loop и синхронно возвращает результат."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def search_with_plan(self, query_text: str, plan: SearchPlan) -> list[Candidate]:
        """Выполняет hybrid search и возвращает список Candidate."""
        return self._run_sync(self._async_search(query_text, plan))

    def search(self, query: str, k: int = 10, **_kwargs) -> list[dict[str, Any]]:
        """Compatibility shim для QAService и /v1/search."""
        plan = SearchPlan(
            normalized_queries=[query],
            k_per_query=k,
            fusion="rrf",
        )
        candidates = self.search_with_plan(query, plan)
        items: list[dict[str, Any]] = []
        for candidate in candidates:
            metadata = dict(candidate.metadata or {})
            dense_vector = metadata.get("_dense_vector")
            items.append(
                {
                    "id": candidate.id,
                    "text": candidate.text,
                    "metadata": metadata,
                    "distance": 0.0,
                    "embedding": dense_vector,
                }
            )
        return items

    def get_context(self, query: str, k: int = 5) -> list[str]:
        """Compatibility shim для QAService без planner."""
        return [item["text"] for item in self.search(query, k=k)]

    def get_context_with_metadata(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Compatibility shim для QAService без planner."""
        return [
            {
                "document": item["text"],
                "metadata": item.get("metadata", {}),
                "distance": float(item.get("distance", 0.0)),
            }
            for item in self.search(query, k=k)
        ]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compatibility shim для legacy MMR-кода."""
        return self._run_sync(self._async_embed_texts(texts))

    async def _async_search(
        self, query_text: str, plan: SearchPlan
    ) -> list[Candidate]:
        """Async реализация hybrid search."""
        dense_vector: list[float] = await self._embedding_client.embed_query(query_text)

        sparse_result = next(iter(self._sparse_encoder.query_embed(query_text)))
        sparse_vector = models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )

        query_filter = self._build_filter(plan.metadata_filters)
        prefetch_limit = max(plan.k_per_query * 2, 20)

        logger.debug(
            "HybridRetriever query: collection=%s prefetch_limit=%d k=%d filter=%s",
            self._store.collection,
            prefetch_limit,
            plan.k_per_query,
            bool(query_filter),
        )

        result = await self._store.client.query_points(
            collection_name=self._store.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=QdrantStore.DENSE_VECTOR,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=QdrantStore.SPARSE_VECTOR,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=plan.k_per_query,
        )

        candidates = self._to_candidates(result.points)
        logger.debug(
            "HybridRetriever: %d результатов для '%s'", len(candidates), query_text[:60]
        )
        return candidates

    async def _async_embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Возвращает эмбеддинги без добавления префиксов."""
        return await self._embedding_client._embed_batch(texts, normalize=True)

    def _build_filter(
        self, filters: Optional[MetadataFilters]
    ) -> Optional[models.Filter]:
        """Преобразует MetadataFilters в qdrant_client.models.Filter."""
        if not filters:
            return None

        conditions: list[models.FieldCondition] = []

        if filters.channel_usernames:
            clean_names = [u.lstrip("@") for u in filters.channel_usernames]
            conditions.append(
                models.FieldCondition(
                    key="channel",
                    match=models.MatchAny(any=clean_names),
                )
            )

        if filters.date_from or filters.date_to:
            conditions.append(
                models.FieldCondition(
                    key="date",
                    range=models.DatetimeRange(
                        gte=filters.date_from or None,
                        lte=filters.date_to or None,
                    ),
                )
            )

        return models.Filter(must=conditions) if conditions else None

    def _to_candidates(self, points: list[Any]) -> list[Candidate]:
        """Конвертирует ScoredPoint из Qdrant в Candidate."""
        candidates: list[Candidate] = []
        for point in points:
            payload: dict[str, Any] = point.payload or {}

            dense_vec: list[float] | None = None
            if isinstance(point.vector, dict):
                dense_vec = point.vector.get(QdrantStore.DENSE_VECTOR)

            candidates.append(
                Candidate(
                    id=str(point.id),
                    text=payload.get("text", ""),
                    metadata={
                        "channel": payload.get("channel"),
                        "channel_id": payload.get("channel_id"),
                        "message_id": payload.get("message_id"),
                        "date": payload.get("date"),
                        "author": payload.get("author"),
                        "url": payload.get("url"),
                        "_dense_vector": dense_vec,
                    },
                    bm25_score=None,
                    dense_score=float(point.score),
                    source="hybrid",
                )
            )
        return candidates
