from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

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
        payload={
            "text": f"text {idx}",
            "channel": "channel",
            "message_id": idx,
            "date": "2026-01-01T00:00:00",
        },
    )


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

    await store.ensure_collection()


@pytest.mark.asyncio
async def test_ensure_collection_creates_payload_indices(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """После create_collection создаются payload-индексы (16 штук)."""
    mock_qdrant_client.collection_exists.return_value = False

    await store.ensure_collection()

    assert mock_qdrant_client.create_payload_index.call_count == 16
    calls_fields = [
        c.args[1] for c in mock_qdrant_client.create_payload_index.call_args_list
    ]
    assert "channel" in calls_fields
    assert "date" in calls_fields
    assert "entities" in calls_fields
    assert "year_week" in calls_fields


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
    assert "dense_vector" in points[0].vector
    assert "sparse_vector" in points[0].vector
    # ID теперь UUID5 от point_id, проверяем что он не пустой
    assert points[0].id is not None


@pytest.mark.asyncio
async def test_upsert_multiple_batches(
    store: QdrantStore, mock_qdrant_client: AsyncMock
) -> None:
    """130 документов с batch_size=64 → 3 вызова upsert (64+64+2)."""
    docs = [make_point_doc(i) for i in range(130)]

    result = await store.upsert(docs, batch_size=64)

    assert result == 130
    assert mock_qdrant_client.upsert.call_count == 3


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
