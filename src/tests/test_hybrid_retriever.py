from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models as qdrant_models

from adapters.search.hybrid_retriever import HybridRetriever
from schemas.search import MetadataFilters, SearchPlan


def make_retriever(
    collection: str = "news",
) -> tuple[HybridRetriever, MagicMock, MagicMock, MagicMock]:
    """Возвращает (retriever, mock_store, mock_tei, mock_sparse)."""
    mock_store = MagicMock()
    mock_store.collection = collection
    mock_store.client = AsyncMock()

    mock_result = MagicMock()
    mock_result.points = []
    mock_store.client.query_points = AsyncMock(return_value=mock_result)

    mock_tei = MagicMock()
    mock_tei.embed_query = AsyncMock(return_value=[0.1] * 1024)

    mock_sparse_enc = MagicMock()
    mock_sparse_result = MagicMock()
    mock_sparse_result.indices = MagicMock(tolist=lambda: [1, 5, 10])
    mock_sparse_result.values = MagicMock(tolist=lambda: [0.5, 0.3, 0.2])
    mock_sparse_enc.query_embed = MagicMock(return_value=iter([mock_sparse_result]))

    settings = MagicMock()
    settings.hybrid_enabled = True

    retriever = HybridRetriever(
        store=mock_store,
        embedding_client=mock_tei,
        sparse_encoder=mock_sparse_enc,
        settings=settings,
    )
    return retriever, mock_store, mock_tei, mock_sparse_enc


def make_plan(k: int = 10, filters: MetadataFilters | None = None) -> SearchPlan:
    return SearchPlan(
        normalized_queries=["test query"],
        metadata_filters=filters,
        k_per_query=k,
        fusion="rrf",
    )


@pytest.mark.asyncio
async def test_async_search_calls_embed_and_query_points() -> None:
    """_async_search вызывает embed_query и query_points."""
    retriever, mock_store, mock_tei, _ = make_retriever()

    # Патчим ColBERT чтоб получить RRF-only path
    with patch.object(retriever, "_get_colbert_query_vectors", new_callable=AsyncMock, return_value=None):
        await retriever._async_search("курс рубля", make_plan(k=5))

    mock_tei.embed_query.assert_awaited_once_with("курс рубля")
    mock_store.client.query_points.assert_awaited_once()
    call_kwargs = mock_store.client.query_points.call_args.kwargs
    assert call_kwargs["collection_name"] == "news"
    assert call_kwargs["with_vectors"] is True
    # RRF-only path: 2 prefetch (dense + sparse)
    prefetch = call_kwargs["prefetch"]
    assert len(prefetch) == 2


@pytest.mark.asyncio
async def test_async_search_uses_weighted_rrf() -> None:
    """query RrfQuery с весами (BM25 weight=3, dense=1)."""
    retriever, mock_store, _, _ = make_retriever()

    with patch.object(retriever, "_get_colbert_query_vectors", new_callable=AsyncMock, return_value=None):
        await retriever._async_search("test", make_plan())

    call_kwargs = mock_store.client.query_points.call_args.kwargs
    assert isinstance(call_kwargs["query"], qdrant_models.RrfQuery)


def test_build_filter_none_when_no_filters() -> None:
    """Без фильтров возвращает None."""
    retriever, *_ = make_retriever()
    result = retriever._build_filter(None)
    assert result is None


def test_build_filter_channel_usernames() -> None:
    """channel_usernames с @ → MatchAny без @."""
    retriever, *_ = make_retriever()
    filters = MetadataFilters(channel_usernames=["@news", "@finance"])
    result = retriever._build_filter(filters)
    assert result is not None
    cond = result.must[0]
    assert cond.key == "channel"
    assert set(cond.match.any) == {"news", "finance"}


def test_build_filter_date_range() -> None:
    """date_from/date_to конвертируется в DatetimeRange."""
    retriever, *_ = make_retriever()
    filters = MetadataFilters(
        date_from="2026-01-01T00:00:00", date_to="2026-03-01T00:00:00"
    )
    result = retriever._build_filter(filters)
    assert result is not None
    cond = result.must[0]
    assert cond.key == "date"
    # DatetimeRange — gte через .range
    assert "2026-01-01" in str(cond.range.gte)


def test_to_candidates_extracts_fields() -> None:
    """ScoredPoint → Candidate с id, text, metadata, dense_score."""
    retriever, *_ = make_retriever()

    point = MagicMock()
    point.id = "channel:123"
    point.score = 0.42
    point.payload = {
        "text": "Новость",
        "channel": "@news",
        "message_id": 123,
        "date": "2026-01-01T00:00:00",
    }
    point.vector = {"dense_vector": [0.1] * 1024}

    query_vector = [0.1] * 1024
    candidates = retriever._to_candidates([point], query_vector)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.id == "channel:123"
    assert c.text == "Новость"
    assert c.metadata["channel"] == "@news"


def test_search_with_plan_is_sync() -> None:
    """search_with_plan возвращает список, не корутину."""
    retriever, *_ = make_retriever()
    plan = make_plan()
    result = retriever.search_with_plan("test", plan)
    assert isinstance(result, list)
