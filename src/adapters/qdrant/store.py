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


@dataclass
class PointDocument:
    """Transfer object для загрузки одного документа в Qdrant.

    Поля векторов передаются уже сгенерированными снаружи:
      - dense_vector: список float из TEIEmbeddingClient.embed_documents()
      - sparse_indices / sparse_values: из fastembed SparseTextEmbedding.embed()
    """

    point_id: str
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    payload: dict[str, Any]


class QdrantStore:
    """Тонкий адаптер над AsyncQdrantClient.

    Инкапсулирует создание коллекции и CRUD-операции.
    Поисковые запросы (query_points) выполняются через self.client в HybridRetriever.

    Использование:
        store = QdrantStore(url="http://qdrant:6333", collection="news")
        await store.ensure_collection()
        await store.upsert(documents)
        await store.aclose()
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
        """Прямой доступ к AsyncQdrantClient для HybridRetriever."""
        return self._client

    @property
    def collection(self) -> str:
        """Имя коллекции Qdrant."""
        return self._collection

    async def ensure_collection(self) -> None:
        """Создаёт коллекцию с named vectors и payload-индексами, если не существует.

        Идемпотентен — безопасно вызывать при каждом старте приложения.
        Обрабатывает race condition: если параллельный процесс уже создал коллекцию,
        молча продолжает работу.
        """
        try:
            exists = await self._client.collection_exists(self._collection)
        except Exception as exc:
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
        """Создаёт payload-индексы для фильтрации по channel, date, author, message_id.

        Вызывается однократно при создании коллекции.
        Параметры is_tenant / is_principal оптимизируют HNSW граф.
        """
        await self._client.create_payload_index(
            self._collection,
            "channel",
            field_schema=models.KeywordIndexParams(type="keyword", is_tenant=True),
        )
        await self._client.create_payload_index(
            self._collection,
            "date",
            field_schema=models.DatetimeIndexParams(
                type="datetime", is_principal=True
            ),
        )
        await self._client.create_payload_index(
            self._collection,
            "author",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            self._collection,
            "message_id",
            field_schema=models.IntegerIndexParams(
                type="integer", lookup=True, range=True
            ),
        )

    async def aclose(self) -> None:
        """Закрывает HTTP-соединение. Вызывать в lifespan shutdown."""
        await self._client.close()
        logger.info("QdrantStore: соединение закрыто")

    async def upsert(
        self, documents: list[PointDocument], batch_size: int = 64
    ) -> int:
        """Загружает документы в Qdrant батчами.

        Каждый документ содержит оба вектора (dense + sparse) и payload.
        wait=True гарантирует видимость данных сразу после возврата.
        """
        if not documents:
            return 0

        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]

            points = []
            for doc in batch:
                qdrant_id = str(uuid.uuid5(_POINT_ID_NS, doc.point_id))
                payload = {**doc.payload, "point_id": doc.point_id}
                points.append(
                    models.PointStruct(
                        id=qdrant_id,
                        vector={
                            self.DENSE_VECTOR: doc.dense_vector,
                            self.SPARSE_VECTOR: models.SparseVector(
                                indices=doc.sparse_indices,
                                values=doc.sparse_values,
                            ),
                        },
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
            except Exception as exc:
                logger.error(
                    "Qdrant upsert ошибка (batch start=%d, size=%d): %s",
                    start,
                    len(batch),
                    exc,
                )
                raise

        return total

    async def delete(self, point_ids: list[str]) -> None:
        """Удаляет точки по списку string ID."""
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
        except Exception as exc:
            logger.error("Qdrant delete ошибка: %s", exc)
            raise

    async def get_by_ids(self, point_ids: list[str]) -> list[Any]:
        """Извлекает точки по ID (только payload, без векторов)."""
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
        except Exception as exc:
            logger.error("Qdrant retrieve ошибка: %s", exc)
            raise

    async def collection_info(self) -> dict[str, Any]:
        """Возвращает базовую статистику коллекции для диагностики."""
        try:
            info = await self._client.get_collection(self._collection)
            return {
                "name": self._collection,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(info.status),
            }
        except Exception as exc:
            logger.error("Qdrant collection_info ошибка: %s", exc)
            raise
