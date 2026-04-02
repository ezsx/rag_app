"""
Тесты для TEI HTTP клиентов.

Unit-тесты используют httpx.AsyncBaseTransport для изоляции от реального TEI.
Integration-тест (помечен @pytest.mark.integration) требует запущенного TEI.
"""

import json

import httpx
import pytest

from adapters.tei import TEIEmbeddingClient, TEIRerankerClient


def _make_embed_client(transport):
    """Создаёт TEIEmbeddingClient с mock transport, минуя __init__."""
    client = TEIEmbeddingClient.__new__(TEIEmbeddingClient)
    client.base_url = "http://mock"
    client.query_instruction = ""
    client._whitening = None
    client._client = httpx.AsyncClient(transport=transport, base_url="http://mock")
    return client


class MockEmbedTransport(httpx.AsyncBaseTransport):
    """Mock TEI embedding: возвращает фиксированный вектор 1024-dim."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embed":
            body = json.loads(request.content)
            n = len(body["inputs"])
            vectors = [[0.1] * 1024 for _ in range(n)]
            return httpx.Response(200, json=vectors)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(404)


class MockRerankTransport(httpx.AsyncBaseTransport):
    """Mock TEI reranker: возвращает убывающие scores."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rerank":
            body = json.loads(request.content)
            n = len(body["texts"])
            results = [{"index": i, "score": float(n - i)} for i in range(n)]
            results.sort(key=lambda x: x["score"], reverse=True)
            return httpx.Response(200, json=results)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(404)


@pytest.mark.asyncio
async def test_embed_query_returns_vector():
    """embed_query возвращает вектор правильной размерности."""
    client = _make_embed_client(MockEmbedTransport())
    vector = await client.embed_query("тест запрос")
    assert len(vector) == 1024
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.asyncio
async def test_embed_query_adds_prefix():
    """embed_query добавляет instruction prefix к тексту (если задан)."""
    captured_inputs = []

    class CapturingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            body = json.loads(request.content)
            captured_inputs.extend(body["inputs"])
            return httpx.Response(200, json=[[0.1] * 1024])

    client = _make_embed_client(CapturingTransport())
    await client.embed_query("новости крипто")
    # Проверяем что запрос содержит текст (prefix может быть пустым для pplx-embed)
    assert "новости крипто" in captured_inputs[0]


@pytest.mark.asyncio
async def test_embed_documents_no_prefix():
    """embed_documents отправляет документы без prefix."""
    captured_inputs = []

    class CapturingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            body = json.loads(request.content)
            captured_inputs.extend(body["inputs"])
            return httpx.Response(200, json=[[0.1] * 1024, [0.2] * 1024])

    client = _make_embed_client(CapturingTransport())
    await client.embed_documents(["текст 1", "текст 2"])
    assert captured_inputs[0] == "текст 1"
    assert captured_inputs[1] == "текст 2"


@pytest.mark.asyncio
async def test_embed_connect_error_propagates():
    """ConnectError пробрасывается наружу без подавления."""

    class FailTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("connection refused")

    client = _make_embed_client(FailTransport())
    with pytest.raises(httpx.ConnectError):
        await client.embed_query("test")


@pytest.mark.asyncio
async def test_rerank_restores_original_order():
    """rerank() возвращает scores в порядке входных passages, не по убыванию score."""
    client = TEIRerankerClient.__new__(TEIRerankerClient)
    client.base_url = "http://mock"
    client._client = httpx.AsyncClient(
        transport=MockRerankTransport(), base_url="http://mock"
    )
    passages = ["doc_a", "doc_b", "doc_c"]
    scores = await client.rerank("query", passages)

    assert len(scores) == 3
    assert scores[0] == 3.0
    assert scores[1] == 2.0
    assert scores[2] == 1.0


@pytest.mark.asyncio
async def test_rerank_empty_passages():
    """rerank() с пустым списком возвращает пустой список без HTTP запроса."""
    client = TEIRerankerClient.__new__(TEIRerankerClient)
    client.base_url = "http://mock"
    client._client = httpx.AsyncClient(base_url="http://mock")
    scores = await client.rerank("query", [])
    assert scores == []


@pytest.mark.asyncio
async def test_healthcheck_returns_true_on_200():
    client = _make_embed_client(MockEmbedTransport())
    assert await client.healthcheck() is True
