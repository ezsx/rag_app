"""Custom pipeline adapter — BM25+Dense → RRF → ColBERT MaxSim via Qdrant HTTP.

Самостоятельная реализация, не импортирует evaluate_retrieval.py.
Embedding без instruction prefix (DEC-0042).
ColBERT через Qdrant native multivector query (prefetch + rescore).
"""

from __future__ import annotations

import json
import urllib.request

from fastembed import SparseTextEmbedding

from benchmarks.config import (
    COLLECTION,
    COLBERT_VECTOR_NAME,
    DENSE_VECTOR_NAME,
    EMBEDDING_URL,
    FINAL_TOP_K,
    QDRANT_URL,
    SPARSE_MODEL_NAME,
    SPARSE_VECTOR_NAME,
)
from benchmarks.protocols import RetrievalResult

# Lazy singleton
_sparse_model: SparseTextEmbedding | None = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    return _sparse_model


def _embed_query(text: str) -> list[float]:
    """Dense embedding через gpu_server.py (без instruction prefix, DEC-0042)."""
    body = json.dumps({"inputs": [text], "normalize": True}).encode()
    req = urllib.request.Request(
        f"{EMBEDDING_URL}/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=15).read())[0]


def _colbert_encode(text: str) -> list[list[float]] | None:
    """ColBERT per-token encoding через gpu_server.py."""
    try:
        body = json.dumps({"texts": [text], "is_query": True}).encode()
        req = urllib.request.Request(
            f"{EMBEDDING_URL}/colbert-encode",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        result = json.loads(urllib.request.urlopen(req, timeout=15).read())
        return result[0] if result else None
    except Exception:
        return None


def _search_qdrant(query_text: str, top_k: int) -> list[dict]:
    """3-stage Qdrant query: BM25+Dense → RRF → ColBERT MaxSim."""
    dense_vec = _embed_query(query_text)

    sparse_model = _get_sparse_model()
    sparse_result = next(iter(sparse_model.query_embed(query_text)))
    sparse_q = {
        "indices": sparse_result.indices.tolist(),
        "values": sparse_result.values.tolist(),
    }

    colbert_vecs = _colbert_encode(query_text)

    if colbert_vecs:
        # 3-stage: BM25+Dense → RRF → ColBERT MaxSim rerank
        body = {
            "prefetch": [
                {
                    "prefetch": [
                        {"query": dense_vec, "using": DENSE_VECTOR_NAME, "limit": 20},
                        {"query": sparse_q, "using": SPARSE_VECTOR_NAME, "limit": 100},
                    ],
                    "query": {"fusion": "rrf"},
                    "limit": max(top_k * 3, 30),
                }
            ],
            "query": colbert_vecs,
            "using": COLBERT_VECTOR_NAME,
            "limit": top_k,
            "with_payload": ["channel", "message_id", "text"],
        }
    else:
        # Fallback 2-stage: BM25+Dense → RRF (без ColBERT)
        body = {
            "prefetch": [
                {"query": dense_vec, "using": DENSE_VECTOR_NAME, "limit": 20},
                {"query": sparse_q, "using": SPARSE_VECTOR_NAME, "limit": 100},
            ],
            "query": {"fusion": "rrf"},
            "limit": top_k,
            "with_payload": ["channel", "message_id", "text"],
        }

    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return resp["result"]["points"]


class CustomRetriever:
    """Production pipeline: BM25+Dense → RRF → ColBERT MaxSim via Qdrant."""

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[RetrievalResult]:
        points = _search_qdrant(query, top_k)
        results = []
        for p in points:
            pay = p.get("payload", {})
            channel = pay.get("channel", "")
            msg_id = pay.get("message_id", 0)
            results.append(RetrievalResult(
                doc_id=f"{channel}:{msg_id}",
                score=p.get("score", 0.0),
                channel=channel,
                message_id=msg_id,
                text=pay.get("text"),
            ))
        return results
