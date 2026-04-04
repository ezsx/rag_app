# SPEC-RAG-03: Qdrant Store Adapter

> **Версия:** 1.0
> **Дата:** 2026-03-17
> **Статус:** Draft
> **Цель:** Создать `src/adapters/qdrant/store.py` — тонкий адаптер над `AsyncQdrantClient`,
> инкапсулирующий создание коллекции с named vectors, upsert, delete и точечное извлечение.
>
> **Источники:**
> - `docs/specifications/arch-brief.md` (DEC-0015, схема коллекции)
> - `docs/research/reports/R01-qdrant-hybrid-rag.md`
> - `docs/architecture/07-data-model/data-model.md`
> - `docs/architecture/03-invariants/invariants.md` (INV-06: атомарный ingest)

---

## 0. Implementation Pointers

### 0.1 Текущие файлы (что есть сейчас)

| Файл | Текущее поведение | После SPEC-RAG-03 |
|------|------------------|-------------------|
| `src/adapters/chroma/retriever.py` | ChromaDB-клиент, SentenceTransformer | **Не трогать** — удаляется в SPEC-RAG-04 |
| `src/adapters/search/bm25_index.py` | Disk-based BM25 индекс | **Не трогать** — удаляется в SPEC-RAG-04 |
| `src/core/deps.py` | Фабрики для chroma/bm25 | **Добавить** `get_qdrant_store()` фрагментом (полная замена в SPEC-RAG-04) |

### 0.2 Новые файлы (создать)

```
src/adapters/qdrant/
    __init__.py       — экспортирует QdrantStore, PointDocument
    store.py          — QdrantStore + PointDocument
```

### 0.3 Что НЕ удаляется здесь

Весь `src/adapters/chroma/` и `src/adapters/search/` **остаются нетронутыми** — удаляются только
в SPEC-RAG-04, когда будет готов HybridRetriever. До этого момента приложение продолжает работать
с ChromaDB-стеком.

---

## 1. Обзор

### 1.1 Задача

1. Создать `src/adapters/qdrant/__init__.py` с реэкспортом.
2. Создать `src/adapters/qdrant/store.py` с классом `QdrantStore` и dataclass `PointDocument`.
3. `QdrantStore.ensure_collection()` — идемпотентное создание коллекции `news` с named vectors
   `dense_vector` (1024-dim, Cosine) и `sparse_vector` (Qdrant/bm25, modifier=IDF) + 4 payload-индекса.
4. `QdrantStore.upsert()` — батч-загрузка `PointDocument` с dense + sparse векторами.
5. `QdrantStore.delete()` — удаление точек по string ID.
6. `QdrantStore.get_by_ids()` — точечное извлечение payload без векторов (для инструмента `verify`).
7. `QdrantStore.collection_info()` — диагностическая статистика коллекции.
8. Добавить `get_qdrant_store()` фабрику в `src/core/deps.py` (фрагмент — без удаления старых фабрик).
9. Добавить `ensure_collection()` и `aclose()` в lifespan `src/main.py`.

### 1.2 Контекст

DEC-0015: `QdrantStore` заменяет ~400 строк ChromaDB + BM25IndexManager. Qdrant хранит dense и
sparse-векторы одной точки атомарно, серверный IDF обновляется автоматически при каждом upsert.

Разделение ответственности:
- `QdrantStore` — **только** lifecycle коллекции + CRUD (этот спек).
- `HybridRetriever` (SPEC-RAG-04) — поисковые запросы через `store.client.query_points()`.
- Ingest pipeline (SPEC-RAG-06) — использует `QdrantStore.upsert()`.

`AsyncQdrantClient` используется вместо синхронного для совместимости с FastAPI event loop.
Ingest-скрипт оборачивает вызовы в `asyncio.run()`.

### 1.3 Ключевые решения

| Решение | Выбор | Обоснование |
|---------|-------|-------------|
| Async vs sync | `AsyncQdrantClient` | FastAPI async; ingest использует `asyncio.run()` |
| Имена векторов | `dense_vector`, `sparse_vector` | Закреплено в arch-brief.md и data-model.md |
| `modifier=IDF` | Да, серверный | Автообновление IDF при upsert/delete — кастомный BM25 не нужен |
| `on_disk=False` для sparse | Да | Коллекция ~60 МБ, помещается в RAM |
| `is_tenant=True` на `channel` | Да | Qdrant оптимизирует HNSW граф по tenant-полю |
| `is_principal=True` на `date` | Да | Оптимизация time-range фильтров |
| `wait=True` в upsert/delete | Да | Гарантирует видимость данных сразу (INV-06) |
| String Point ID | `"{channel}:{message_id}"` | Уникальность + читаемость; Qdrant v1.x поддерживает string ID |

### 1.4 Что НЕ делать

- **Не реализовывать** `query_points()` / `hybrid_search()` в `QdrantStore` — это задача SPEC-RAG-04.
- **Не хранить** `SparseTextEmbedding` (fastembed) внутри `QdrantStore` — sparse-векторы
  генерируются снаружи и передаются в `PointDocument`.
- **Не создавать** ABC или базовый класс — один конкретный класс достаточен.
- **Не использовать** sync `QdrantClient` — только `AsyncQdrantClient`.
- **Не удалять** chroma/bm25 адаптеры — это делает SPEC-RAG-04.
- **Не хардкодить** URL и имя коллекции — только через `Settings`.
- **Не вызывать** `recreate_collection()` — потеря данных. Только `create_collection` с проверкой.

---

## 2. PointDocument — Transfer Object

Dataclass для передачи данных между слоями (ingest → QdrantStore).

```python
# src/adapters/qdrant/store.py (верх файла, перед QdrantStore)

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


@dataclass
class PointDocument:
    """Transfer object для загрузки одного документа в Qdrant.

    Поля векторов передаются уже сгенерированными снаружи:
      - dense_vector: список float из TEIEmbeddingClient.embed_documents()
      - sparse_indices / sparse_values: из fastembed SparseTextEmbedding.embed()
    """

    point_id: str               # "{channel_name}:{message_id}" — уникальный ключ
    dense_vector: list[float]   # 1024-dim, L2-нормированный (TEI normalize=True)
    sparse_indices: list[int]   # ненулевые индексы BM25
    sparse_values: list[float]  # соответствующие веса
    payload: dict[str, Any]     # text, channel, channel_id, message_id, date, author, url
```

**Инварианты:**
- `len(dense_vector) == 1024` — проверяется логом, не исключением (производительность).
- `len(sparse_indices) == len(sparse_values)` — нарушение → Qdrant вернёт 422.
- `payload` **обязан** содержать: `text`, `channel`, `message_id`, `date`.
  Поля `author`, `url`, `channel_id` — опциональны.

---

## 3. QdrantStore — Полная реализация

```python
class QdrantStore:
    """Тонкий адаптер над AsyncQdrantClient.

    Инкапсулирует создание коллекции и CRUD-операции.
    Поисковые запросы (query_points) выполняются через self.client в HybridRetriever.

    Использование:
        store = QdrantStore(url="http://qdrant:6333", collection="news")
        await store.ensure_collection()   # при старте приложения
        await store.upsert(documents)     # в ingest pipeline
        await store.aclose()             # при shutdown
    """

    # Имена векторов — закреплены в arch-brief.md / data-model.md
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

    # ------------------------------------------------------------------
    # Публичные свойства
    # ------------------------------------------------------------------

    @property
    def client(self) -> AsyncQdrantClient:
        """Прямой доступ к AsyncQdrantClient для HybridRetriever."""
        return self._client

    @property
    def collection(self) -> str:
        """Имя коллекции Qdrant."""
        return self._collection

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
            # Race condition: другой процесс успел создать коллекцию первым
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
        Параметры is_tenant / is_principal оптимизируют HNSW граф (R01).
        """
        # channel: tenant-поле — Qdrant группирует сегменты по нему
        await self._client.create_payload_index(
            self._collection,
            "channel",
            field_schema=models.KeywordIndexParams(
                type="keyword", is_tenant=True
            ),
        )
        # date: principal-поле — оптимизирует time-range фильтры
        await self._client.create_payload_index(
            self._collection,
            "date",
            field_schema=models.DatetimeIndexParams(
                type="datetime", is_principal=True
            ),
        )
        # author: keyword для фильтрации по автору
        await self._client.create_payload_index(
            self._collection,
            "author",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        # message_id: integer с lookup + range для точечного и диапазонного поиска
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

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def upsert(
        self, documents: list[PointDocument], batch_size: int = 64
    ) -> int:
        """Загружает документы в Qdrant батчами.

        Каждый документ содержит оба вектора (dense + sparse) и payload.
        wait=True гарантирует видимость данных сразу после возврата (INV-06).

        Args:
            documents:  список PointDocument для загрузки.
            batch_size: размер батча. По умолчанию 64 (баланс скорости и памяти).

        Returns:
            Количество успешно загруженных точек.

        Raises:
            UnexpectedResponse: при ошибке Qdrant API.
            Exception: при сетевой ошибке.
        """
        if not documents:
            return 0

        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]

            points = [
                models.PointStruct(
                    id=doc.point_id,
                    vector={
                        self.DENSE_VECTOR: doc.dense_vector,
                        self.SPARSE_VECTOR: models.SparseVector(
                            indices=doc.sparse_indices,
                            values=doc.sparse_values,
                        ),
                    },
                    payload=doc.payload,
                )
                for doc in batch
            ]

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
        """Удаляет точки по списку string ID.

        Идемпотентен — отсутствующие в коллекции ID игнорируются Qdrant молча.
        wait=True гарантирует, что удалённые точки не возвращаются в поиске (INV-06).

        Args:
            point_ids: список ID вида "{channel}:{message_id}".
        """
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
        """Извлекает точки по ID (только payload, без векторов).

        Используется инструментом verify агента для проверки конкретных документов.

        Args:
            point_ids: список ID вида "{channel}:{message_id}".

        Returns:
            Список Record (qdrant_client.models.Record) с полем payload.
            Порядок не гарантирован — может отличаться от порядка point_ids.
        """
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
        """Возвращает базовую статистику коллекции для диагностики.

        Returns:
            dict с ключами: name, points_count, indexed_vectors_count, status.

        Raises:
            Exception: если коллекция не существует или недоступна.
        """
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
```

---

## 4. `__init__.py`

```python
# src/adapters/qdrant/__init__.py

from .store import PointDocument, QdrantStore

__all__ = ["QdrantStore", "PointDocument"]
```

---

## 5. Интеграция

### 5.1 Фрагмент для `src/core/deps.py`

Добавить рядом с существующими фабриками (не удалять chroma/bm25 — удаляет SPEC-RAG-04):

```python
from adapters.qdrant.store import QdrantStore


@lru_cache
def get_qdrant_store() -> QdrantStore:
    """Синглтон QdrantStore. Смена URL требует cache_clear() через settings.update_*()."""
    settings = get_settings()
    return QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
    )
```

### 5.2 Фрагменты для `src/main.py`

Добавить в lifespan context manager:

```python
from core.deps import get_qdrant_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    store = get_qdrant_store()
    await store.ensure_collection()          # создаёт коллекцию при первом запуске
    # ... (остальной startup из SPEC-RAG-02: tei clients)

    yield

    # --- shutdown ---
    await store.aclose()
    # ... (остальной shutdown из SPEC-RAG-02: tei clients aclose)
```

> **Порядок startup:** ensure_collection → остальные инициализации.
> Если Qdrant недоступен при старте — приложение упадёт с исключением, что правильно.

---

## 6. Тесты

Файл: `src/tests/test_qdrant_store.py`

Все тесты используют `unittest.mock.AsyncMock` для `AsyncQdrantClient`.

```python
# src/tests/test_qdrant_store.py

import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client import models

# путь к QdrantStore зависит от PYTHONPATH: обычно `adapters.qdrant.store`
from adapters.qdrant.store import PointDocument, QdrantStore

MOCK_URL = "http://localhost:6333"
MOCK_COLLECTION = "news"


@pytest.fixture
def mock_qdrant_client() -> AsyncMock:
    """Замокированный AsyncQdrantClient."""
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.retrieve = AsyncMock(return_value=[])
    client.get_collection = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def store(mock_qdrant_client: AsyncMock) -> QdrantStore:
    """QdrantStore с замокированным клиентом."""
    with patch(
        "adapters.qdrant.store.AsyncQdrantClient",
        return_value=mock_qdrant_client,
    ):
        return QdrantStore(url=MOCK_URL, collection=MOCK_COLLECTION)


def make_point_doc(idx: int = 0) -> PointDocument:
    return PointDocument(
        point_id=f"channel:{idx}",
        dense_vector=[0.1] * 1024,
        sparse_indices=[1, 5, 10],
        sparse_values=[0.5, 0.3, 0.2],
        payload={"text": f"text {idx}", "channel": "channel", "message_id": idx, "date": "2026-01-01T00:00:00"},
    )


# ------------------------------------------------------------------
# ensure_collection
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """Коллекция создаётся со строгой схемой: dense_vector 1024-dim Cosine + sparse_vector IDF."""
    mock_qdrant_client.collection_exists.return_value = False

    await store.ensure_collection()

    mock_qdrant_client.create_collection.assert_called_once()
    call_kwargs = mock_qdrant_client.create_collection.call_args.kwargs

    assert call_kwargs["collection_name"] == MOCK_COLLECTION
    dense = call_kwargs["vectors_config"]["dense_vector"]
    assert dense.size == 1024
    assert dense.distance == models.Distance.COSINE

    sparse = call_kwargs["sparse_vectors_config"]["sparse_vector"]
    assert sparse.modifier == models.Modifier.IDF


@pytest.mark.asyncio
async def test_ensure_collection_skips_if_exists(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """Если коллекция уже существует — create_collection не вызывается."""
    mock_qdrant_client.collection_exists.return_value = True

    await store.ensure_collection()

    mock_qdrant_client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_collection_handles_race_condition(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """Race condition: другой процесс создал коллекцию — исключение 'already exists' глотается."""
    mock_qdrant_client.collection_exists.return_value = False
    exc = UnexpectedResponse(
        status_code=400,
        reason_phrase="Bad Request",
        content=b'{"status": {"error": "Collection already exists"}}',
        headers={},
    )
    mock_qdrant_client.create_collection.side_effect = exc

    # Не должно бросить исключение
    await store.ensure_collection()


@pytest.mark.asyncio
async def test_ensure_collection_creates_payload_indices(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """После create_collection создаются 4 payload-индекса."""
    mock_qdrant_client.collection_exists.return_value = False

    await store.ensure_collection()

    assert mock_qdrant_client.create_payload_index.call_count == 4
    # Проверяем наличие channel и date (ключевые поля)
    calls_fields = [
        c.args[1] for c in mock_qdrant_client.create_payload_index.call_args_list
    ]
    assert "channel" in calls_fields
    assert "date" in calls_fields
    assert "author" in calls_fields
    assert "message_id" in calls_fields


# ------------------------------------------------------------------
# upsert
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_empty_returns_zero(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """Пустой список — client.upsert не вызывается, возвращает 0."""
    result = await store.upsert([])
    assert result == 0
    mock_qdrant_client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_single_batch(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """2 документа → 1 вызов upsert с PointStruct с dense + sparse векторами."""
    docs = [make_point_doc(0), make_point_doc(1)]

    result = await store.upsert(docs)

    assert result == 2
    mock_qdrant_client.upsert.assert_called_once()
    call_kwargs = mock_qdrant_client.upsert.call_args.kwargs
    assert call_kwargs["collection_name"] == MOCK_COLLECTION
    assert call_kwargs["wait"] is True
    points = call_kwargs["points"]
    assert len(points) == 2
    # Проверяем структуру вектора первой точки
    assert "dense_vector" in points[0].vector
    assert "sparse_vector" in points[0].vector
    assert points[0].id == "channel:0"


@pytest.mark.asyncio
async def test_upsert_multiple_batches(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """130 документов с batch_size=64 → 3 вызова upsert (64+64+2)."""
    docs = [make_point_doc(i) for i in range(130)]

    result = await store.upsert(docs, batch_size=64)

    assert result == 130
    assert mock_qdrant_client.upsert.call_count == 3


# ------------------------------------------------------------------
# delete
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_calls_client_correctly(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """delete передаёт PointIdsList с правильными ID и wait=True."""
    ids = ["channel:1", "channel:2"]

    await store.delete(ids)

    mock_qdrant_client.delete.assert_called_once()
    call_kwargs = mock_qdrant_client.delete.call_args.kwargs
    assert call_kwargs["collection_name"] == MOCK_COLLECTION
    assert call_kwargs["wait"] is True
    assert set(call_kwargs["points_selector"].points) == set(ids)


@pytest.mark.asyncio
async def test_delete_empty_list_noop(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """Пустой список → client.delete не вызывается."""
    await store.delete([])
    mock_qdrant_client.delete.assert_not_called()


# ------------------------------------------------------------------
# get_by_ids
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_by_ids_passes_correct_params(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """retrieve вызывается с with_payload=True, with_vectors=False."""
    mock_record = MagicMock()
    mock_record.payload = {"text": "hello", "channel": "ch"}
    mock_qdrant_client.retrieve.return_value = [mock_record]

    result = await store.get_by_ids(["channel:1"])

    mock_qdrant_client.retrieve.assert_called_once_with(
        collection_name=MOCK_COLLECTION,
        ids=["channel:1"],
        with_payload=True,
        with_vectors=False,
    )
    assert result == [mock_record]


@pytest.mark.asyncio
async def test_get_by_ids_empty_returns_empty(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    result = await store.get_by_ids([])
    assert result == []
    mock_qdrant_client.retrieve.assert_not_called()
```

---

## 7. Чеклист реализации

- [ ] `src/adapters/qdrant/` — каталог создан
- [ ] `src/adapters/qdrant/__init__.py` — экспортирует `QdrantStore`, `PointDocument`
- [ ] `src/adapters/qdrant/store.py` — `PointDocument` dataclass реализован
- [ ] `QdrantStore.__init__` — инициализирует `AsyncQdrantClient`, логирует
- [ ] `QdrantStore.ensure_collection` — проверяет `collection_exists`, создаёт с named vectors
  - [ ] `vectors_config["dense_vector"]`: size=1024, distance=Cosine
  - [ ] `sparse_vectors_config["sparse_vector"]`: modifier=IDF, on_disk=False
  - [ ] Race condition (`already exists`) — не бросает исключение
- [ ] `QdrantStore._create_payload_indices` — 4 индекса: channel (tenant), date (principal), author, message_id
- [ ] `QdrantStore.upsert` — батчи по 64, wait=True, логирует прогресс, возвращает count
- [ ] `QdrantStore.delete` — PointIdsList, wait=True, idempotent
- [ ] `QdrantStore.get_by_ids` — retrieve без векторов
- [ ] `QdrantStore.collection_info` — возвращает points_count, status
- [ ] `QdrantStore.aclose` — закрывает AsyncQdrantClient
- [ ] `QdrantStore.client` property — доступ для HybridRetriever
- [ ] `src/core/deps.py` — добавлен `get_qdrant_store()` с `@lru_cache`
- [ ] `src/main.py` lifespan — `ensure_collection()` при startup, `aclose()` при shutdown
- [ ] `qdrant-client>=1.9.0` добавлен в `requirements.txt`
- [ ] `src/tests/test_qdrant_store.py` — 9 тестов реализованы, `pytest` проходит
- [ ] Старые адаптеры (`chroma/`, `search/`) **НЕ тронуты** — удаляются в SPEC-RAG-04
