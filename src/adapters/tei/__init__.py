"""TEI HTTP adapters for embedding and reranking."""

from .embedding_client import TEIEmbeddingClient
from .reranker_client import TEIRerankerClient

__all__ = ["TEIEmbeddingClient", "TEIRerankerClient"]
