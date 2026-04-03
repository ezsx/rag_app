"""
HTTP client for embedding service (pplx-embed-v1-0.6B via gpu_server.py).

POST /embed -> list[list[float]] (normalize=True, 1024-dim).
No instruction prefix for either query or document. Mean pooling + L2 normalize.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# pplx-embed не требует instruction prefix
DEFAULT_QUERY_INSTRUCTION = ""
_PASSAGE_PREFIX = ""


class TEIEmbeddingClient:
    """Async HTTP client for embedding service. Singleton via deps.get_tei_embedding_client().

    - embed_query: query embedding with optional instruction prefix
    - embed_documents: batch document embedding without prefix
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
            base_url: embedding service URL, e.g. "http://host.docker.internal:8082"
            timeout: HTTP request timeout in seconds
            query_instruction: instruction prefix for query embedding
            whitening_params_path: path to .npz with PCA whitening params
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
            except Exception as exc:  # broad: adapter boundary
                logger.warning("Не удалось загрузить whitening: %s", exc)

        logger.info("TEIEmbeddingClient инициализирован: %s", self.base_url)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Applies instruction prefix + optional PCA whitening."""
        prefixed = self.query_instruction + text
        vectors = await self._embed_batch([prefixed], normalize=True)
        vec = vectors[0]

        if self._whitening is not None:
            vec = self._apply_whitening(vec)

        return vec

    def _apply_whitening(self, vec: list[float]) -> list[float]:
        """PCA whitening: center → project → scale → normalize."""
        import numpy as np
        assert self._whitening is not None
        x = np.array(vec, dtype=np.float32)
        centered = x - self._whitening["mean"]
        projected = centered @ self._whitening["components"].T
        whitened = projected * self._whitening["scale"]
        norm = np.linalg.norm(whitened)
        if norm > 0:
            whitened = whitened / norm
        return whitened.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed documents for ingest. Returns L2-normalized 1024-dim vectors."""
        prefixed = [_PASSAGE_PREFIX + t for t in texts]
        return await self._embed_batch(prefixed, normalize=True)

    async def _embed_batch(
        self, texts: list[str], normalize: bool = True, max_batch: int = 32
    ) -> list[list[float]]:
        """POST /embed with automatic sub-batching (server limit is typically 32)."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), max_batch):
            chunk = texts[i : i + max_batch]
            try:
                response = await self._client.post(
                    "/embed",
                    json={"inputs": chunk, "normalize": normalize},
                )
                response.raise_for_status()
                vectors: list[list[float]] = response.json()
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
        """Check if embedding service is reachable via GET /health."""
        try:
            response = await self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("TEI embedding healthcheck failed: %s", exc)
            return False

    async def aclose(self) -> None:
        """Close HTTP connection pool. Call during application shutdown."""
        await self._client.aclose()
