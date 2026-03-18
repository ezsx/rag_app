"""
Unit-тесты для ключевых функций scripts/ingest_telegram.py (Phase 1).
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest


def _load_ingest_module():
    """Загружает scripts/ingest_telegram.py как обычный модуль."""
    spec = importlib.util.spec_from_file_location(
        "ingest_telegram_module",
        Path(__file__).parent.parent.parent / "scripts" / "ingest_telegram.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_message(msg_id: int, text: str, chat_id: int = 100) -> MagicMock:
    """Создаёт mock Telegram Message."""
    msg = MagicMock()
    msg.id = msg_id
    msg.chat_id = chat_id
    msg.message = text
    msg.date = datetime(2024, 6, 1, tzinfo=timezone.utc)
    msg.sender = None
    msg.reply_to_msg_id = None
    msg.views = None
    return msg


def _make_sparse_result(n_tokens: int = 5) -> MagicMock:
    """Создаёт mock EmbeddingResult от fastembed."""
    result = MagicMock()
    result.indices = np.array(list(range(n_tokens)))
    result.values = np.array([1.0] * n_tokens)
    return result


class TestBuildPointDocsFlat:
    """Тесты helper-функции построения PointDocument."""

    def test_single_message_no_chunk(self):
        mod = _load_ingest_module()
        msg = _make_message(42, "hello world")
        dense = [[0.1] * 1024]
        sparse = [_make_sparse_result()]

        docs = mod._build_point_docs_flat(
            [msg], ["hello world"], dense, sparse, "chan", 0
        )

        assert len(docs) == 1
        doc = docs[0]
        assert doc.point_id == "chan:42"
        assert doc.payload["message_id"] == 42
        assert doc.payload["channel"] == "chan"
        assert doc.payload["text"] == "hello world"
        assert len(doc.dense_vector) == 1024

    def test_two_chunks_same_message(self):
        mod = _load_ingest_module()
        msg = _make_message(7, "text")
        dense = [[0.1] * 1024, [0.2] * 1024]
        sparse = [_make_sparse_result(), _make_sparse_result()]

        docs = mod._build_point_docs_flat(
            [msg, msg], ["part1", "part2"], dense, sparse, "chan", 100
        )

        assert len(docs) == 2
        assert docs[0].point_id == "chan:7:0"
        assert docs[1].point_id == "chan:7:1"

    def test_two_chunks_same_message_with_smart_chunk_ids(self):
        mod = _load_ingest_module()
        msg = _make_message(8, "text")
        dense = [[0.1] * 1024, [0.2] * 1024]
        sparse = [_make_sparse_result(), _make_sparse_result()]

        docs = mod._build_point_docs_flat(
            [msg, msg], ["part1", "part2"], dense, sparse, "chan", 0
        )

        assert docs[0].point_id == "chan:8:0"
        assert docs[1].point_id == "chan:8:1"

    def test_payload_fields(self):
        mod = _load_ingest_module()
        msg = _make_message(1, "text", chat_id=999)

        docs = mod._build_point_docs_flat(
            [msg], ["text"], [[0.0] * 1024], [_make_sparse_result()], "mychan", 0
        )

        payload = docs[0].payload
        assert payload["channel"] == "mychan"
        assert payload["channel_id"] == 999
        assert payload["message_id"] == 1
        assert "date" in payload
        assert "text" in payload
        assert "author" not in payload

    def test_sparse_vectors_are_lists(self):
        mod = _load_ingest_module()
        msg = _make_message(1, "text")

        docs = mod._build_point_docs_flat(
            [msg], ["text"], [[0.0] * 1024], [_make_sparse_result(3)], "chan", 0
        )

        assert isinstance(docs[0].sparse_indices, list)
        assert isinstance(docs[0].sparse_values, list)


class TestIngestBatches:
    """Тесты батчевого ingest с замоканными зависимостями."""

    @pytest.mark.asyncio
    async def test_calls_embed_and_upsert(self):
        mod = _load_ingest_module()
        messages = [_make_message(i, f"text {i}") for i in range(3)]

        embedding_client = AsyncMock()
        embedding_client.embed_documents = AsyncMock(
            return_value=[[0.1] * 1024 for _ in range(3)]
        )

        sparse_results = [_make_sparse_result() for _ in range(3)]
        sparse_encoder = MagicMock()
        sparse_encoder.embed = MagicMock(return_value=iter(sparse_results))

        qdrant_store = AsyncMock()
        qdrant_store.upsert = AsyncMock(return_value=3)

        stats = await mod.ingest_batches(
            messages=messages,
            batch_size=10,
            embedding_client=embedding_client,
            sparse_encoder=sparse_encoder,
            qdrant_store=qdrant_store,
            channel_hint="@testchan",
        )

        embedding_client.embed_documents.assert_called_once_with(
            ["text 0", "text 1", "text 2"]
        )
        sparse_encoder.embed.assert_called_once_with(["text 0", "text 1", "text 2"])
        qdrant_store.upsert.assert_called_once()
        assert stats["written_qdrant"] == 3
        assert stats["processed_in_channel"] == 3

    @pytest.mark.asyncio
    async def test_skips_empty_messages(self):
        mod = _load_ingest_module()
        messages = [_make_message(1, ""), _make_message(2, "  ")]

        embedding_client = AsyncMock()
        sparse_encoder = MagicMock()
        qdrant_store = AsyncMock()
        qdrant_store.upsert = AsyncMock(return_value=0)

        stats = await mod.ingest_batches(
            messages=messages,
            batch_size=10,
            embedding_client=embedding_client,
            sparse_encoder=sparse_encoder,
            qdrant_store=qdrant_store,
        )

        embedding_client.embed_documents.assert_not_called()
        qdrant_store.upsert.assert_not_called()
        assert stats["written_qdrant"] == 0

    @pytest.mark.asyncio
    async def test_uses_chat_id_when_channel_hint_not_username(self):
        mod = _load_ingest_module()
        messages = [_make_message(10, "text", chat_id=555)]

        embedding_client = AsyncMock()
        embedding_client.embed_documents = AsyncMock(return_value=[[0.1] * 1024])

        sparse_encoder = MagicMock()
        sparse_encoder.embed = MagicMock(return_value=iter([_make_sparse_result()]))

        qdrant_store = AsyncMock()
        qdrant_store.upsert = AsyncMock(return_value=1)

        await mod.ingest_batches(
            messages=messages,
            batch_size=10,
            embedding_client=embedding_client,
            sparse_encoder=sparse_encoder,
            qdrant_store=qdrant_store,
            channel_hint="555",
        )

        point_docs = qdrant_store.upsert.await_args.args[0]
        assert point_docs[0].point_id == "555:10"
        assert point_docs[0].payload["channel"] == "555"

    @pytest.mark.asyncio
    async def test_retries_embed_timeout_then_succeeds(self, monkeypatch):
        mod = _load_ingest_module()
        messages = [_make_message(i, f"text {i}") for i in range(2)]

        sleep_mock = AsyncMock()
        monkeypatch.setattr(mod.asyncio, "sleep", sleep_mock)

        embedding_client = AsyncMock()
        embedding_client.embed_documents = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("timeout"),
                [[0.1] * 1024 for _ in range(2)],
            ]
        )

        sparse_encoder = MagicMock()
        sparse_encoder.embed = MagicMock(
            return_value=iter([_make_sparse_result(), _make_sparse_result()])
        )

        qdrant_store = AsyncMock()
        qdrant_store.upsert = AsyncMock(return_value=2)

        stats = await mod.ingest_batches(
            messages=messages,
            batch_size=10,
            embedding_client=embedding_client,
            sparse_encoder=sparse_encoder,
            qdrant_store=qdrant_store,
            channel_hint="@retrychan",
        )

        assert embedding_client.embed_documents.await_count == 2
        sleep_mock.assert_awaited_once()
        qdrant_store.upsert.assert_called_once()
        assert stats["written_qdrant"] == 2

    @pytest.mark.asyncio
    async def test_raises_after_embed_retry_exhausted(self, monkeypatch):
        mod = _load_ingest_module()
        messages = [_make_message(1, "text")]

        sleep_mock = AsyncMock()
        monkeypatch.setattr(mod.asyncio, "sleep", sleep_mock)

        embedding_client = AsyncMock()
        embedding_client.embed_documents = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("timeout")
                for _ in range(mod.EMBED_RETRY_ATTEMPTS)
            ]
        )

        sparse_encoder = MagicMock()
        sparse_encoder.embed = MagicMock(return_value=iter([_make_sparse_result()]))

        qdrant_store = AsyncMock()
        qdrant_store.upsert = AsyncMock(return_value=1)

        with pytest.raises(httpx.ConnectTimeout):
            await mod.ingest_batches(
                messages=messages,
                batch_size=10,
                embedding_client=embedding_client,
                sparse_encoder=sparse_encoder,
                qdrant_store=qdrant_store,
                channel_hint="@retrychan",
            )

        assert embedding_client.embed_documents.await_count == mod.EMBED_RETRY_ATTEMPTS
        assert sleep_mock.await_count == mod.EMBED_RETRY_ATTEMPTS - 1
        qdrant_store.upsert.assert_not_called()
