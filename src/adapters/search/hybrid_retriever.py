"""
HybridRetriever — текущий runtime retrieval pipeline.

Основной путь:
  1. dense (cosine) + sparse (BM25) prefetch
  2. weighted RRF fusion (BM25 weight=3, dense weight=1)
  3. ColBERT MaxSim rerank через named vector `colbert_vector` (если доступен)
  4. channel dedup: max 2 документа на канал

Fallback путь при недоступном ColBERT:
  dense + sparse → weighted RRF → channel dedup.

dense_score каждого кандидата — cosine similarity с query vector,
а не RRF score (RRF scores неинтерпретируемы для coverage metric).
Legacy MMR helper сохранён только для compatibility/testing путей.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from fastembed import SparseTextEmbedding
from qdrant_client import models

from adapters.qdrant.store import QdrantStore
from adapters.tei.embedding_client import TEIEmbeddingClient
from core.observability import observe_span
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

    # ── Public API для tools ──────────────────────────────
    def run_sync(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Публичный sync→async bridge для tools."""
        return self._run_sync(coro)

    @property
    def store(self) -> QdrantStore:
        """Публичный доступ к Qdrant store."""
        return self._store

    @property
    def embedding_client(self) -> TEIEmbeddingClient:
        """Публичный доступ к embedding client."""
        return self._embedding_client

    @property
    def sparse_encoder(self) -> SparseTextEmbedding:
        """Публичный доступ к sparse encoder."""
        return self._sparse_encoder

    def search_with_plan(self, query_text: str, plan: SearchPlan) -> list[Candidate]:
        """Выполняет hybrid search и возвращает список Candidate."""
        with observe_span(
            "hybrid_retrieval",
            input={"query": query_text[:200], "strategy": getattr(plan, "strategy", None)},
            metadata={"k_per_query": plan.k_per_query, "fusion": plan.fusion},
        ) as span:
            results = self._run_sync(self._async_search(query_text, plan))
            if span:
                # Top-3 результата для видимости в trace
                top_docs = [
                    {"id": c.id, "text": c.text[:150], "score": round(getattr(c, "score", 0.0), 4)}
                    for c in results[:3]
                ]
                span.update(output={
                    "num_results": len(results),
                    "queries": plan.normalized_queries,
                    "rerank_method": "colbert_maxsim" if getattr(self, '_settings', None) and getattr(self._settings, 'embedding_tei_url', None) else "rrf_only",
                    "top_docs": top_docs,
                })
            return results

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
        """Hybrid search: BM25 + Dense → RRF → ColBERT rerank.

        Трёхэтапный pipeline:
        1. prefetch: dense + sparse → каждый по prefetch_limit
        2. RRF fusion → расширенный набор кандидатов
        3. ColBERT MaxSim rerank → финальные top-k (если коллекция поддерживает)
        """
        dense_vector: list[float] = await self._embedding_client.embed_query(query_text)

        sparse_result = next(iter(self._sparse_encoder.query_embed(query_text)))
        sparse_vector = models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )

        query_filter = self._build_filter(plan.metadata_filters)
        rrf_limit = max(plan.k_per_query * 5, 50)

        # Асимметричный prefetch: BM25 берёт больше кандидатов (100 vs 20).
        bm25_prefetch_limit = max(plan.k_per_query * 10, 100)
        dense_prefetch_limit = max(plan.k_per_query * 2, 20)

        # ColBERT rerank: encode query через gpu_server /colbert-encode
        colbert_query_vectors = await self._get_colbert_query_vectors(query_text)

        if colbert_query_vectors:
            # Трёхэтапный: BM25+Dense → RRF → ColBERT MaxSim rerank
            logger.debug(
                "HybridRetriever ColBERT: collection=%s rrf=%d k=%d",
                self._store.collection, rrf_limit, plan.k_per_query,
            )
            result = await self._store.client.query_points(
                collection_name=self._store.collection,
                prefetch=[
                    models.Prefetch(
                        prefetch=[
                            models.Prefetch(
                                query=dense_vector,
                                using=QdrantStore.DENSE_VECTOR,
                                limit=dense_prefetch_limit,
                            ),
                            models.Prefetch(
                                query=sparse_vector,
                                using=QdrantStore.SPARSE_VECTOR,
                                limit=bm25_prefetch_limit,
                            ),
                        ],
                        query=models.RrfQuery(
                            rrf=models.Rrf(weights=[1.0, 3.0]),
                        ),
                        limit=rrf_limit,
                    ),
                ],
                query=colbert_query_vectors,
                using="colbert_vector",
                query_filter=query_filter,
                with_payload=True,
                with_vectors=True,
                limit=plan.k_per_query * 2,  # запрашиваем 2x, dedup сузит
            )
        else:
            # Fallback: BM25+Dense → RRF (без ColBERT)
            logger.debug(
                "HybridRetriever RRF-only: collection=%s rrf=%d k=%d",
                self._store.collection, rrf_limit, plan.k_per_query,
            )
            result = await self._store.client.query_points(
                collection_name=self._store.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=QdrantStore.DENSE_VECTOR,
                        limit=dense_prefetch_limit,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=QdrantStore.SPARSE_VECTOR,
                        limit=bm25_prefetch_limit,
                    ),
                ],
                query=models.RrfQuery(
                    rrf=models.Rrf(weights=[1.0, 3.0]),
                ),
                query_filter=query_filter,
                with_payload=True,
                with_vectors=True,
                limit=plan.k_per_query * 2,  # запрашиваем 2x, dedup сузит
            )

        candidates = self._to_candidates(result.points, dense_vector)

        # Channel dedup: max 2 docs per channel для diversity
        candidates = self._channel_dedup(candidates, max_per_channel=2)

        logger.debug(
            "HybridRetriever: %d результатов для '%s'", len(candidates), query_text[:60]
        )
        return candidates

    @staticmethod
    def _channel_dedup(
        candidates: list[Candidate], max_per_channel: int = 2
    ) -> list[Candidate]:
        """Ограничить количество документов из одного канала.
        Сохраняет порядок (по score), но пропускает лишние из того же канала.
        """
        channel_counts: dict[str, int] = {}
        result = []
        for c in candidates:
            ch = c.metadata.get("channel", "")
            count = channel_counts.get(ch, 0)
            if count < max_per_channel:
                result.append(c)
                channel_counts[ch] = count + 1
        return result

    async def _get_colbert_query_vectors(self, query_text: str) -> list[list[float]] | None:
        """Encode query через ColBERT (gpu_server /colbert-encode).
        Возвращает list of token vectors [N_tokens × 128] или None при ошибке.
        """
        import urllib.request
        try:
            colbert_url = self._settings.embedding_tei_url.replace("/embed", "").rstrip("/")
            body = json.dumps({"texts": [query_text], "is_query": True}).encode()
            req = urllib.request.Request(
                f"{colbert_url}/colbert-encode",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=15)
            result = json.loads(resp.read())
            if result and result[0]:
                return result[0]  # list of 128-dim token vectors
        except Exception as e:
            logger.warning("ColBERT query encoding failed: %s", e)
        return None

    @staticmethod
    def _mmr_rerank(
        candidates: list[Candidate],
        query_vector: list[float],
        k: int,
        lambda_: float,
    ) -> list[Candidate]:
        """MMR post-processing: выбирает k кандидатов, балансируя relevance и diversity.

        Использует dense_score (cosine с query) как relevance,
        и cosine между документами как similarity для diversity penalty.
        """
        import numpy as np

        from utils.ranking import mmr_select

        # Подготовка данных для mmr_select
        query_emb = np.asarray(query_vector, dtype=np.float32)
        doc_embs = []
        mmr_candidates = []

        for c in candidates:
            dense_vec = (c.metadata or {}).get("_dense_vector")
            if dense_vec is None:
                continue
            doc_embs.append(dense_vec)
            mmr_candidates.append({
                "id": c.id,
                "text": c.text,
                "score": c.dense_score,
                "metadata": c.metadata,
                "bm25_score": c.bm25_score,
                "dense_score": c.dense_score,
                "source": c.source,
            })

        if len(mmr_candidates) <= k:
            return candidates[:k]

        doc_embs_np = np.asarray(doc_embs, dtype=np.float32)
        selected = mmr_select(
            mmr_candidates, query_emb, doc_embs_np,
            lambda_=lambda_, out_k=k,
        )

        # Конвертируем обратно в Candidate
        return [
            Candidate(
                id=s["id"],
                text=s["text"],
                metadata=s["metadata"],
                bm25_score=s.get("bm25_score"),
                dense_score=s.get("dense_score", 0.0),
                source=s.get("source", "hybrid"),
            )
            for s in selected
        ]

    async def _async_embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Возвращает эмбеддинги без добавления префиксов."""
        return await self._embedding_client._embed_batch(texts, normalize=True)

    def _build_filter(
        self, filters: MetadataFilters | None
    ) -> models.Filter | None:
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

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity между двумя векторами."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _to_candidates(
        self,
        points: list[Any],
        query_vector: list[float] | None = None,
    ) -> list[Candidate]:
        """Конвертирует ScoredPoint из Qdrant в Candidate.

        dense_score вычисляется как cosine similarity между query_vector и
        document dense_vector. Если query_vector не передан или у документа
        нет dense vector — fallback на point.score.
        """
        candidates: list[Candidate] = []
        for point in points:
            payload: dict[str, Any] = point.payload or {}

            dense_vec: list[float] | None = None
            if isinstance(point.vector, dict):
                dense_vec = point.vector.get(QdrantStore.DENSE_VECTOR)

            # Cosine similarity вместо RRF/MMR score для coverage metric
            if query_vector and dense_vec:
                cosine_sim = self._cosine_similarity(query_vector, dense_vec)
            else:
                cosine_sim = float(point.score)

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
                    dense_score=cosine_sim,
                    source="hybrid",
                )
            )
        return candidates
