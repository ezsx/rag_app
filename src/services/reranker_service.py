"""
Синхронная обёртка над async TEI reranker client.

Phase 1 использует общий `TEIRerankerClient` из DI, а этот сервис даёт
совместимый sync API для `qa_service.py`, инструментов и legacy-кода.
Чтобы не падать с `RuntimeError: This event loop is already running`,
async вызовы исполняются в выделенном event loop внутри фонового потока.
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
    """
    Sync-обёртка над `TEIRerankerClient` с backward-compatible API.
    """

    def __init__(self, reranker_client: TEIRerankerClient) -> None:
        """
        Args:
            reranker_client: общий async-клиент TEI reranker из `deps.py`
        """
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
        """
        Возвращает индексы документов, отсортированные по убыванию релевантности к запросу.
        """
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
        """Переранжирует документы и возвращает нормализованные scores."""
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
        """Проксирует healthcheck TEI reranker через sync API."""
        try:
            return self._run_async(self._client.healthcheck())
        except Exception as exc:  # broad: adapter boundary
            logger.warning("TEI reranker healthcheck failed: %s", exc)
            return False

    def close(self) -> None:
        """
        Останавливает фоновой event loop.

        Сам `TEIRerankerClient` не закрывается здесь: его lifecycle управляется
        отдельно через `get_tei_reranker_client().aclose()` в `main.py`.
        """
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()

    def _get_raw_scores(self, query: str, passages: list[str]) -> list[float]:
        """
        Возвращает raw relevance scores в порядке входных passages.

        TEI-клиент уже восстанавливает исходный порядок по `index`, поэтому
        здесь остаётся только синхронно дождаться async результата.
        """
        return self._run_async(self._client.rerank(query, passages))

    def _run_event_loop(self) -> None:
        """Фоновый поток с выделенным asyncio loop для sync→async bridge."""
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _run_async(self, coro: Coroutine[object, object, T]) -> T:
        """Выполняет coroutine в фоновом loop и синхронно возвращает результат."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Сигмоида для нормализации raw logit score в [0, 1]."""
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
