from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# Namespace для детерминированного UUID5 из строковых point_id.
# Одна и та же строка всегда даёт один UUID → idempotent upsert.
_POINT_ID_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

logger = logging.getLogger(__name__)

# Канонический список payload indexes (SPEC-RAG-20a).
# Единый source of truth — используется в store.py и migrate_collection.py.
PAYLOAD_INDEXES = [
    # Critical — используются почти во всех запросах
    ("channel", models.KeywordIndexParams(type="keyword", is_tenant=True)),
    ("date", models.DatetimeIndexParams(type="datetime", is_principal=True)),
    ("entities", models.PayloadSchemaType.KEYWORD),
    ("year_week", models.PayloadSchemaType.KEYWORD),
    # Secondary — для специализированных tools (entity_tracker, arxiv_tracker)
    ("entity_orgs", models.PayloadSchemaType.KEYWORD),
    ("entity_models", models.PayloadSchemaType.KEYWORD),
    ("arxiv_ids", models.PayloadSchemaType.KEYWORD),
    ("hashtags", models.PayloadSchemaType.KEYWORD),
    ("url_domains", models.PayloadSchemaType.KEYWORD),
    ("lang", models.PayloadSchemaType.KEYWORD),
    ("forwarded_from_id", models.PayloadSchemaType.KEYWORD),
    ("year_month", models.PayloadSchemaType.KEYWORD),
    ("root_message_id", models.PayloadSchemaType.KEYWORD),
    ("author", models.PayloadSchemaType.KEYWORD),
    ("message_id", models.IntegerIndexParams(type="integer", lookup=True, range=True)),
    # Range
    ("text_length", models.IntegerIndexParams(type="integer", lookup=False, range=True)),
]


@dataclass
class PointDocument:
    """Transfer object for upserting a single document into Qdrant.

    Vector fields are pre-computed externally (TEIEmbeddingClient, fastembed).
    """

    point_id: str
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    payload: dict[str, Any]
    colbert_vectors: list[list[float]] | None = None  # per-token 128-dim


class QdrantStore:
    """Thin adapter over AsyncQdrantClient for collection CRUD.

    Search queries (query_points) are executed via self.client in HybridRetriever.
    """

    DENSE_VECTOR: str = "dense_vector"
    SPARSE_VECTOR: str = "sparse_vector"
    DENSE_DIM: int = 1024

    def __init__(self, url: str, collection: str) -> None:
        self._url = url
        self._collection = collection
        self._client = AsyncQdrantClient(url=url)
        logger.info(
            "QdrantStore инициализирован: url=%s collection=%s", url, collection
        )

    @property
    def client(self) -> AsyncQdrantClient:
        """Direct access to AsyncQdrantClient for HybridRetriever."""
        return self._client

    @property
    def collection(self) -> str:
        """Qdrant collection name."""
        return self._collection

    async def ensure_collection(self) -> None:
        """Create collection with named vectors + payload indexes if it doesn't exist.

        Idempotent -- safe to call on every app startup. Handles race conditions.
        """
        try:
            exists = await self._client.collection_exists(self._collection)
        except Exception as exc:  # broad: qdrant adapter boundary
            logger.error(
                "Qdrant: ошибка проверки коллекции '%s': %s", self._collection, exc
            )
            raise

        if exists:
            logger.info(
                "Qdrant: коллекция '%s' уже существует, пропуск создания",
                self._collection,
            )
            return

        logger.info("Qdrant: создание коллекции '%s'", self._collection)
        try:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    self.DENSE_VECTOR: models.VectorParams(
                        size=self.DENSE_DIM,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                        index=models.SparseIndexParams(on_disk=False),
                    ),
                },
            )
        except UnexpectedResponse as exc:
            if "already exists" in str(exc).lower():
                logger.info(
                    "Qdrant: коллекция '%s' создана параллельным процессом",
                    self._collection,
                )
                return
            logger.error(
                "Qdrant: ошибка создания коллекции '%s': %s", self._collection, exc
            )
            raise

        await self._create_payload_indices()
        logger.info(
            "Qdrant: коллекция '%s' создана с payload-индексами", self._collection
        )

    async def _create_payload_indices(self) -> None:
        """Create all 16 payload indexes from PAYLOAD_INDEXES.

        Called on collection creation. Without indexes, analytics tools get Qdrant 400 on Facet API.
        """
        failed = []
        for field_name, field_schema in PAYLOAD_INDEXES:
            try:
                await self._client.create_payload_index(
                    self._collection,
                    field_name,
                    field_schema=field_schema,  # type: ignore[arg-type]
                )
                logger.debug("Payload index '%s' создан", field_name)
            except Exception as exc:  # broad: qdrant adapter boundary
                logger.error("Payload index '%s' FAILED: %s", field_name, exc)
                failed.append(field_name)
        if failed:
            logger.error(
                "Не удалось создать %d payload index(es): %s", len(failed), failed
            )
        else:
            logger.info("Все %d payload indexes созданы", len(PAYLOAD_INDEXES))

    async def aclose(self) -> None:
        """Close HTTP connection. Call during lifespan shutdown."""
        await self._client.close()
        logger.info("QdrantStore: соединение закрыто")

    async def upsert(
        self, documents: list[PointDocument], batch_size: int = 8
    ) -> int:
        """Upsert documents into Qdrant in batches. wait=True ensures immediate visibility."""
        if not documents:
            return 0

        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]

            points = []
            for doc in batch:
                qdrant_id = str(uuid.uuid5(_POINT_ID_NS, doc.point_id))
                payload = {**doc.payload, "point_id": doc.point_id}
                vector_data = {
                    self.DENSE_VECTOR: doc.dense_vector,
                    self.SPARSE_VECTOR: models.SparseVector(
                        indices=doc.sparse_indices,
                        values=doc.sparse_values,
                    ),
                }
                if doc.colbert_vectors:
                    vector_data["colbert_vector"] = doc.colbert_vectors
                points.append(
                    models.PointStruct(
                        id=qdrant_id,
                        vector=vector_data,
                        payload=payload,
                    )
                )

            try:
                await self._client.upsert(
                    collection_name=self._collection,
                    points=points,
                    wait=True,
                )
                total += len(batch)
                logger.info(
                    "Qdrant upsert: %d/%d точек (коллекция=%s)",
                    total,
                    len(documents),
                    self._collection,
                )
            except Exception as exc:  # broad: qdrant adapter boundary
                logger.error(
                    "Qdrant upsert ошибка (batch start=%d, size=%d): %s",
                    start,
                    len(batch),
                    exc,
                )
                raise

        return total

    async def delete(self, point_ids: list[str]) -> None:
        """Delete points by string IDs."""
        if not point_ids:
            return

        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=point_ids),
                wait=True,
            )
            logger.info(
                "Qdrant delete: удалено %d точек из '%s'",
                len(point_ids),
                self._collection,
            )
        except Exception as exc:  # broad: qdrant adapter boundary
            logger.error("Qdrant delete ошибка: %s", exc)
            raise

    async def get_by_ids(self, point_ids: list[str]) -> list[Any]:
        """Retrieve points by ID (payload only, no vectors)."""
        if not point_ids:
            return []

        try:
            points = await self._client.retrieve(
                collection_name=self._collection,
                ids=point_ids,
                with_payload=True,
                with_vectors=False,
            )
            return points
        except Exception as exc:  # broad: qdrant adapter boundary
            logger.error("Qdrant retrieve ошибка: %s", exc)
            raise

    async def collection_info(self) -> dict[str, Any]:
        """Return basic collection stats for diagnostics."""
        try:
            info = await self._client.get_collection(self._collection)
            return {
                "name": self._collection,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(info.status),
            }
        except Exception as exc:  # broad: qdrant adapter boundary
            logger.error("Qdrant collection_info ошибка: %s", exc)
            raise
