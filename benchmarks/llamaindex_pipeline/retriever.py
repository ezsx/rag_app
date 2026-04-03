"""LlamaIndex retriever — two configs: stock and maxed.

LI-stock: QdrantVectorStore hybrid, default fusion, no reranker.
LI-maxed: QdrantVectorStore hybrid, weighted RRF (hybrid_fusion_fn), QwenReranker.

Подключается к существующей коллекции news_colbert_v2 без переиндексации.
ColBERT vector (colbert_vector) невидим для LlamaIndex — нет API для multivector query.
"""

from __future__ import annotations

from benchmarks.config import (
    COLLECTION,
    DENSE_VECTOR_NAME,
    FINAL_TOP_K,
    QDRANT_URL,
    SPARSE_MODEL_NAME,
    SPARSE_VECTOR_NAME,
)
from benchmarks.llamaindex_pipeline.embedding import PplxEmbedding
from benchmarks.llamaindex_pipeline.fusion import weighted_rrf_fusion
from benchmarks.llamaindex_pipeline.reranker import QwenReranker
from benchmarks.protocols import RetrievalResult
from fastembed import SparseTextEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# Lazy singleton — fastembed BM25 модель загружается один раз
_sparse_model: SparseTextEmbedding | None = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    return _sparse_model


def _sparse_query_fn(query: str) -> dict:
    """Encode query для BM25 sparse vector — тот же fastembed что и в production."""
    model = _get_sparse_model()
    result = next(iter(model.query_embed(query)))
    return {
        "indices": result.indices.tolist(),
        "values": result.values.tolist(),
    }


def _build_vector_store(*, use_weighted_rrf: bool = False) -> QdrantVectorStore:
    """Создаёт QdrantVectorStore подключённый к существующей коллекции."""
    client = QdrantClient(url=QDRANT_URL, timeout=30)

    kwargs: dict = {
        "collection_name": COLLECTION,
        "client": client,
        "dense_vector_name": DENSE_VECTOR_NAME,
        "sparse_vector_name": SPARSE_VECTOR_NAME,
        "enable_hybrid": True,
        "sparse_query_fn": _sparse_query_fn,
    }

    if use_weighted_rrf:
        kwargs["hybrid_fusion_fn"] = weighted_rrf_fusion

    return QdrantVectorStore(**kwargs)


def _nodes_to_results(nodes) -> list[RetrievalResult]:
    """Конвертирует LlamaIndex NodeWithScore → RetrievalResult."""
    results = []
    for node in nodes:
        metadata = node.node.metadata or {}
        channel = metadata.get("channel", "")
        msg_id = metadata.get("message_id", 0)
        results.append(RetrievalResult(
            doc_id=f"{channel}:{msg_id}",
            score=node.score or 0.0,
            channel=channel,
            message_id=msg_id,
            text=node.node.get_content(),
        ))
    return results


class LlamaIndexRetrieverStock:
    """LI-stock: hybrid search, default fusion, без reranker."""

    def __init__(self):
        self._embed = PplxEmbedding()
        store = _build_vector_store(use_weighted_rrf=False)
        index = VectorStoreIndex.from_vector_store(
            vector_store=store, embed_model=self._embed,
        )
        self._retriever = index.as_retriever(similarity_top_k=FINAL_TOP_K)

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[RetrievalResult]:
        nodes = self._retriever.retrieve(query)
        return _nodes_to_results(nodes)[:top_k]


class LlamaIndexRetrieverMaxed:
    """LI-maxed: hybrid search, weighted RRF, QwenReranker."""

    def __init__(self):
        self._embed = PplxEmbedding()
        store = _build_vector_store(use_weighted_rrf=True)
        index = VectorStoreIndex.from_vector_store(
            vector_store=store, embed_model=self._embed,
        )
        self._retriever = index.as_retriever(similarity_top_k=FINAL_TOP_K)
        self._reranker = QwenReranker()

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[RetrievalResult]:
        nodes = self._retriever.retrieve(query)
        reranked = self._reranker.postprocess_nodes(
            nodes, query_bundle=QueryBundle(query_str=query),
        )
        return _nodes_to_results(reranked)[:top_k]
