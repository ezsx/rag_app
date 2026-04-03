"""
Sync wrapper over async TEIRerankerClient.

Provides backward-compatible sync API for QAService and tools.
Async calls execute in a dedicated background event loop to avoid
RuntimeError in nested event loops.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections.abc import Coroutine
from typing import TypeVar

from adapters.tei.reranker_client import TEIRerankerClient

logger = logging.getLogger(__name__)
T = TypeVar("T")


class RerankerService:
    """Sync wrapper over TEIRerankerClient with backward-compatible API."""

    def __init__(self, reranker_client: TEIRerankerClient) -> None:
        self._client = reranker_client
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="tei-reranker-sync-bridge",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)
        logger.info(
            "RerankerService инициализирован как sync-обёртка над TEI client: %s",
            self._client.base_url,
        )

    def rerank(
        self, query: str, docs: list[str], top_n: int, batch_size: int = 16
    ) -> list[int]:
        """Return document indices sorted by descending relevance to query."""
        if not docs:
            return []
        try:
            scores = self._get_raw_scores(query, docs)
            order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
            if top_n and top_n > 0:
                order = order[: min(top_n, len(order))]
            return order
        except Exception as e:  # broad: adapter boundary
            logger.error("Ошибка ререйкера: %s", e)
            return list(range(min(len(docs), top_n)))

    def rerank_with_scores(
        self, query: str, docs: list[str], top_n: int, batch_size: int = 16
    ) -> tuple[list[int], list[float]]:
        """Rerank documents and return (indices, normalized_scores)."""
        if not docs:
            return [], []
        try:
            raw_scores = self._get_raw_scores(query, docs)
            order = sorted(range(len(docs)), key=lambda i: raw_scores[i], reverse=True)
            if top_n and top_n > 0:
                order = order[: min(top_n, len(order))]
            norm_scores = [self._sigmoid(raw_scores[i]) for i in order]
            return order, norm_scores
        except Exception as exc:  # broad: adapter boundary
            logger.error("Ошибка ререйкера (with_scores): %s", exc)
            return list(range(min(len(docs), top_n or len(docs)))), []

    def healthcheck(self) -> bool:
        """Proxy TEI reranker healthcheck via sync API."""
        try:
            return self._run_async(self._client.healthcheck())
        except Exception as exc:  # broad: adapter boundary
            logger.warning("TEI reranker healthcheck failed: %s", exc)
            return False

    def close(self) -> None:
        """Stop the background event loop. TEIRerankerClient lifecycle is managed separately."""
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()

    def _get_raw_scores(self, query: str, passages: list[str]) -> list[float]:
        """Return raw relevance scores in input passage order."""
        return self._run_async(self._client.rerank(query, passages))

    def _run_event_loop(self) -> None:
        """Background thread running dedicated asyncio loop for sync->async bridge."""
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _run_async(self, coro: Coroutine[object, object, T]) -> T:
        """Run coroutine in background loop, return result synchronously."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid for normalizing raw logit score to [0, 1]."""
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
