"""
HTTP-клиент для TEI embedding service (intfloat/multilingual-e5-large).

Обёртка над TEI REST API:
  POST /embed  → list[list[float]]  (normalize=True, 1024-dim)

Instruction prefix для e5-large (обязателен для корректного retrieval):
  query-текст:    "query: {text}"
  document-текст: "passage: {text}"
"""

from __future__ import annotations

import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)

# Instruction prefixes для intfloat/multilingual-e5-large.
# Без префикса качество retrieval снижается ~5–8% NDCG@10.
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class TEIEmbeddingClient:
    """
    Async HTTP-клиент для TEI embedding service.

    Используется для:
    - embed_query: встраивание поискового запроса (с prefix "query: ")
    - embed_documents: батчевое встраивание документов при ingest (с prefix "passage: ")

    Connection pool переиспользуется между вызовами — инстанс должен быть singleton.
    Создаётся через deps.get_tei_embedding_client().
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        """
        Args:
            base_url: URL TEI service, например "http://host.docker.internal:8082"
            timeout: таймаут HTTP запроса в секундах (default 30s)
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            # Лимит соединений: embedding вызывается последовательно в pipeline,
            # но при параллельных subquery-запросах может быть несколько concurrent
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        logger.info("TEIEmbeddingClient инициализирован: %s", self.base_url)

    async def embed_query(self, text: str) -> List[float]:
        """
        Встраивает один поисковый запрос.

        Применяет prefix "query: " согласно спецификации multilingual-e5-large.
        Возвращает L2-нормализованный вектор 1024-dim.

        Args:
            text: поисковый запрос на естественном языке

        Returns:
            list[float] длиной 1024

        Raises:
            httpx.ConnectError: TEI service недоступен
            httpx.HTTPStatusError: TEI вернул ошибку (4xx/5xx)
        """
        prefixed = _QUERY_PREFIX + text
        vectors = await self._embed_batch([prefixed], normalize=True)
        return vectors[0]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Батчевое встраивание документов для ingest.

        Применяет prefix "passage: " к каждому тексту.
        Возвращает L2-нормализованные векторы 1024-dim.

        Args:
            texts: список текстов документов

        Returns:
            list[list[float]], каждый вектор длиной 1024

        Raises:
            httpx.ConnectError: TEI service недоступен
            httpx.HTTPStatusError: TEI вернул ошибку (4xx/5xx)
        """
        prefixed = [_PASSAGE_PREFIX + t for t in texts]
        return await self._embed_batch(prefixed, normalize=True)

    async def _embed_batch(
        self, texts: List[str], normalize: bool = True, max_batch: int = 32
    ) -> List[List[float]]:
        """Внутренний метод: POST /embed с автоматическим разбиением на суб-батчи.

        TEI имеет серверный лимит на размер батча (обычно 32).
        Если texts длиннее max_batch — разбиваем и склеиваем результат.
        """
        if not texts:
            return []

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), max_batch):
            chunk = texts[i : i + max_batch]
            try:
                response = await self._client.post(
                    "/embed",
                    json={"inputs": chunk, "normalize": normalize},
                )
                response.raise_for_status()
                vectors: List[List[float]] = response.json()
                all_vectors.extend(vectors)
                logger.debug(
                    "TEI embed: sub-batch %d-%d → %d векторов",
                    i, i + len(chunk), len(vectors),
                )
            except httpx.ConnectError as exc:
                logger.error("TEI embedding недоступен (%s): %s", self.base_url, exc)
                raise
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "TEI embedding вернул ошибку %d: %s",
                    exc.response.status_code,
                    exc.response.text[:200],
                )
                raise

        return all_vectors

    async def healthcheck(self) -> bool:
        """
        Проверяет доступность TEI service.

        Returns:
            True если service отвечает на GET /health, False иначе
        """
        try:
            response = await self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("TEI embedding healthcheck failed: %s", exc)
            return False

    async def aclose(self) -> None:
        """Закрывает HTTP connection pool. Вызывать при shutdown приложения."""
        await self._client.aclose()
