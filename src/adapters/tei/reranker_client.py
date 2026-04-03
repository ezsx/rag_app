"""
HTTP client for cross-encoder reranker service (Qwen3-Reranker-0.6B-seq-cls via gpu_server.py).

POST /rerank -> list[{index, score}] sorted by score desc.
Returns scores in original passage order (by index).
"""

from __future__ import annotations

import logging

import httpx

from core.observability import observe_span

logger = logging.getLogger(__name__)


class TEIRerankerClient:
    """Async HTTP client for reranker service. Singleton via deps.get_tei_reranker_client().

    Accepts query + passages, returns relevance scores in original passage order.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        """
        Args:
            base_url: reranker service URL, e.g. "http://host.docker.internal:8082"
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
        logger.info("TEIRerankerClient инициализирован: %s", self.base_url)

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Rerank passages by relevance to query.

        Returns raw scores in original passage order: scores[i] corresponds to passages[i].
        Scores are unnormalized (typically -10..10); callers apply sigmoid or min-max.
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
        """Check if reranker service is reachable."""
        try:
            response = await self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("TEI reranker healthcheck failed: %s", exc)
            return False

    async def aclose(self) -> None:
        """Close HTTP connection pool."""
        await self._client.aclose()
