"""
HTTP-клиент для embedding service (pplx-embed-v1-0.6B).

Обёртка над gpu_server.py REST API:
  POST /embed  → list[list[float]]  (normalize=True, 1024-dim)

pplx-embed-v1-0.6B НЕ использует instruction prefix —
ни для query, ни для document. Mean pooling + L2 normalize.
"""

from __future__ import annotations

import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)

# pplx-embed не требует instruction prefix
DEFAULT_QUERY_INSTRUCTION = ""
_PASSAGE_PREFIX = ""


class TEIEmbeddingClient:
    """
    Async HTTP-клиент для TEI embedding service.

    Используется для:
    - embed_query: встраивание поискового запроса с instruction prefix
    - embed_documents: батчевое встраивание документов без prefix

    Connection pool переиспользуется между вызовами — инстанс должен быть singleton.
    Создаётся через deps.get_tei_embedding_client().
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        whitening_params_path: str = "",
    ) -> None:
        """
        Args:
            base_url: URL TEI service, например "http://host.docker.internal:8082"
            timeout: таймаут HTTP запроса в секундах (default 30s)
            query_instruction: instruction prefix для query embedding
            whitening_params_path: путь к .npz с PCA whitening (mean, components, explained_variance)
        """
        self.base_url = base_url.rstrip("/")
        self.query_instruction = query_instruction
        self._whitening = None
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        # Загрузка PCA whitening transform (если указан)
        if whitening_params_path:
            try:
                import numpy as np
                params = np.load(whitening_params_path)
                self._whitening = {
                    "mean": params["mean"],
                    "components": params["components"],
                    "scale": 1.0 / np.sqrt(params["explained_variance"] + 1e-4),
                }
                out_dim = self._whitening["components"].shape[0]
                logger.info(
                    "PCA whitening загружен: %s (1024→%d dim)",
                    whitening_params_path, out_dim,
                )
            except Exception as exc:
                logger.warning("Не удалось загрузить whitening: %s", exc)

        logger.info("TEIEmbeddingClient инициализирован: %s", self.base_url)

    async def embed_query(self, text: str) -> List[float]:
        """
        Встраивает один поисковый запрос.

        Применяет instruction prefix, затем PCA whitening (если загружен).
        Возвращает L2-нормализованный вектор (512-dim с whitening, 1024 без).
        """
        prefixed = self.query_instruction + text
        vectors = await self._embed_batch([prefixed], normalize=True)
        vec = vectors[0]

        if self._whitening is not None:
            vec = self._apply_whitening(vec)

        return vec

    def _apply_whitening(self, vec: List[float]) -> List[float]:
        """PCA whitening: center → project → scale → normalize."""
        import numpy as np
        x = np.array(vec, dtype=np.float32)
        centered = x - self._whitening["mean"]
        projected = centered @ self._whitening["components"].T
        whitened = projected * self._whitening["scale"]
        norm = np.linalg.norm(whitened)
        if norm > 0:
            whitened = whitened / norm
        return whitened.tolist()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Батчевое встраивание документов для ingest.

        Qwen3-Embedding не использует prefix для документов.
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
