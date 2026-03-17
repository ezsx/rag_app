"""TEI HTTP адаптеры для embedding и reranking."""

from .embedding_client import TEIEmbeddingClient
from .reranker_client import TEIRerankerClient

__all__ = ["TEIEmbeddingClient", "TEIRerankerClient"]
