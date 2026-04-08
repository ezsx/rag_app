"""
HybridRetriever -- runtime retrieval pipeline.

Main path: dense + sparse prefetch -> weighted RRF (BM25 3:1) -> ColBERT MaxSim rerank -> channel dedup.
Fallback (no ColBERT): dense + sparse -> weighted RRF -> channel dedup.
dense_score per candidate is cosine similarity (not RRF score) for coverage metric.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections.abc import Coroutine
from pathlib import Path
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


# ── Standalone pure functions (extracted from HybridRetriever) ──────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity между двумя векторами."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_filter(filters: MetadataFilters | None) -> models.Filter | None:
    """Convert MetadataFilters to qdrant_client.models.Filter."""
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


def _to_candidates(
    points: list[Any],
    query_vector: list[float] | None = None,
) -> list[Candidate]:
    """Convert Qdrant ScoredPoints to Candidates with cosine dense_score."""
    candidates: list[Candidate] = []
    for point in points:
        payload: dict[str, Any] = point.payload or {}

        dense_vec: list[float] | None = None
        if isinstance(point.vector, dict):
            dense_vec = point.vector.get(QdrantStore.DENSE_VECTOR)

        if query_vector and dense_vec:
            cosine_sim = _cosine_similarity(query_vector, dense_vec)
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


class HybridRetriever:
    """Qdrant hybrid retriever: dense (TEI) + sparse (BM25 fastembed) -> native RRF.

    Sync public API with dedicated background event loop for sync->async bridge
    (same pattern as RerankerService). Compatible with ToolRunner and FastAPI endpoints.
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
        self._sparse_lexicon = self._load_lexicon()

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
        logger.info(
            "HybridRetriever инициализирован: collection=%s, lexicon=%d entries",
            store.collection,
            sum(len(v) for v in self._sparse_lexicon.values() if isinstance(v, dict)),
        )

    @staticmethod
    def _load_lexicon() -> dict:
        """Загрузить lexicon для BM25 нормализации (R2 ablation: sparse-only normalization)."""
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "datasets" / "query_normalization_lexicon.json",
            Path("datasets/query_normalization_lexicon.json"),
        ]
        for p in candidates:
            if p.is_file():
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
        return {}

    def _normalize_for_sparse(self, query: str) -> str:
        """Нормализация query для BM25: добавляет синонимы из lexicon, не заменяет оригинал.

        Ablation R2: sparse-only normalization даёт +0.009 R@1, +0.005 MRR.
        Dense query остаётся raw — нормализация dense ВРЕДИТ (R1: −4.2% R@5).
        """
        if not self._sparse_lexicon:
            return query
        additions = []
        query_lower = query.lower()
        for category in self._sparse_lexicon.values():
            if not isinstance(category, dict):
                continue
            for key, replacements in category.items():
                if key.lower() in query_lower:
                    additions.extend(replacements)
        if additions:
            return query + " " + " ".join(additions)
        return query

    def _run_event_loop(self) -> None:
        """Background thread running dedicated asyncio loop."""
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _run_sync(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run coroutine in background loop, return result synchronously."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ── Public API для tools ──────────────────────────────
    def run_sync(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Public sync->async bridge for tools."""
        return self._run_sync(coro)

    @property
    def store(self) -> QdrantStore:
        """Public access to Qdrant store."""
        return self._store

    @property
    def embedding_client(self) -> TEIEmbeddingClient:
        """Public access to embedding client."""
        return self._embedding_client

    @property
    def sparse_encoder(self) -> SparseTextEmbedding:
        """Public access to sparse encoder."""
        return self._sparse_encoder

    def search_with_plan(self, query_text: str, plan: SearchPlan) -> list[Candidate]:
        """Execute hybrid search and return Candidates."""
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
        """Convenience wrapper used by QAService and /v1/search."""
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

    async def _async_search(
        self, query_text: str, plan: SearchPlan
    ) -> list[Candidate]:
        """BM25 + Dense -> RRF -> ColBERT MaxSim rerank -> channel dedup."""
        dense_vector: list[float] = await self._embedding_client.embed_query(query_text)

        # R2 sparse-only normalization: BM25 получает нормализованный query, dense — raw.
        sparse_query = self._normalize_for_sparse(query_text)
        sparse_result = next(iter(self._sparse_encoder.query_embed(sparse_query)))
        sparse_vector = models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )

        query_filter = _build_filter(plan.metadata_filters)
        rrf_limit = max(plan.k_per_query * 5, 50)

        # Асимметричный prefetch: BM25 берёт больше кандидатов (100 vs 40).
        # Dense limit 20→40: ablation study показал +3.4% R@5 (больше кандидатов для ColBERT).
        bm25_prefetch_limit = max(plan.k_per_query * 10, 100)
        dense_prefetch_limit = max(plan.k_per_query * 4, 40)

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

        candidates = _to_candidates(result.points, dense_vector)

        # Channel dedup: max 2 docs per channel для diversity
        candidates = self._channel_dedup(candidates, max_per_channel=3)

        logger.debug(
            "HybridRetriever: %d результатов для '%s'", len(candidates), query_text[:60]
        )
        return candidates

    @staticmethod
    def _channel_dedup(
        candidates: list[Candidate], max_per_channel: int = 2
    ) -> list[Candidate]:
        """Limit docs per channel for diversity. Preserves score order."""
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
        except Exception as e:  # broad: adapter boundary
            logger.warning("ColBERT query encoding failed: %s", e)
        return None


