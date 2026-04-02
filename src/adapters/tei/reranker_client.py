"""
HTTP-клиент для TEI reranker service (BAAI/bge-reranker-v2-m3).

Обёртка над TEI REST API:
  POST /rerank → list[{index: int, score: float}]  (отсортировано по score desc)

bge-reranker-v2-m3 НЕ требует instruction prefix.
Возвращаем scores в исходном порядке passages (по index), не по score.
"""

from __future__ import annotations

import logging

import httpx

from core.observability import observe_span

logger = logging.getLogger(__name__)


class TEIRerankerClient:
    """
    Async HTTP-клиент для TEI reranker service.

    Принимает query + список passages, возвращает relevance scores
    в том же порядке, что и входные passages.

    Создаётся через deps.get_tei_reranker_client().
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        """
        Args:
            base_url: URL TEI service, например "http://host.docker.internal:8083"
            timeout: таймаут HTTP запроса в секундах
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
        logger.info("TEIRerankerClient инициализирован: %s", self.base_url)

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        """
        Переранжирует passages по релевантности к query.

        TEI /rerank возвращает результаты отсортированными по убыванию score.
        Этот метод восстанавливает исходный порядок passages: scores[i]
        соответствует passages[i].

        Args:
            query: поисковый запрос
            passages: список текстов для ранжирования (обычно 20–80 штук)

        Returns:
            list[float] длиной len(passages): score[i] для passages[i].
            Score в диапазоне (обычно) -10..10, не нормализован.
            Для нормализации в [0,1] вызывающий код использует sigmoid или min-max.

        Raises:
            httpx.ConnectError: TEI service недоступен
            httpx.HTTPStatusError: TEI вернул ошибку
        """
        if not passages:
            return []

        try:
            with observe_span(
                "rerank",
                input={"query": query[:200], "num_passages": len(passages)},
                metadata={"reranker_type": "cross_encoder", "model": "Qwen3-Reranker-0.6B"},
            ) as span:
                response = await self._client.post(
                    "/rerank",
                    json={
                        "query": query,
                        "texts": passages,
                        "raw_scores": True,
                        "truncate": True,
                    },
                )
                response.raise_for_status()

                # TEI возвращает [{"index": i, "score": f}, ...] sorted by score desc.
                # Восстанавливаем порядок по index, чтобы scores[i] ↔ passages[i].
                results = response.json()
                scores = [0.0] * len(passages)
                for item in results:
                    scores[item["index"]] = item["score"]

                if span:
                    span.update(output={
                        "top_score": max(scores) if scores else 0.0,
                        "num_passages": len(passages),
                    })

                logger.debug(
                    "TEI rerank: query=%r, %d passages → scores [%.3f..%.3f]",
                    query[:50],
                    len(passages),
                    min(scores),
                    max(scores),
                )
                return scores

        except httpx.ConnectError as exc:
            logger.error("TEI reranker недоступен (%s): %s", self.base_url, exc)
            raise
        except httpx.HTTPStatusError as exc:
            logger.error(
                "TEI reranker вернул ошибку %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise

    async def healthcheck(self) -> bool:
        """Проверяет доступность TEI reranker service."""
        try:
            response = await self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("TEI reranker healthcheck failed: %s", exc)
            return False

    async def aclose(self) -> None:
        """Закрывает HTTP connection pool."""
        await self._client.aclose()
