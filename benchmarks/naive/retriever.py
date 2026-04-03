"""Naive retriever — dense-only search via Qdrant HTTP API.

Без BM25, без reranking, без ColBERT.
Показывает baseline: что даёт один dense vector search.
"""

from __future__ import annotations

import json
import urllib.request

from benchmarks.config import (
    COLLECTION,
    DENSE_VECTOR_NAME,
    EMBEDDING_URL,
    FINAL_TOP_K,
    QDRANT_URL,
)
from benchmarks.protocols import RetrievalResult


def _embed_query(text: str) -> list[float]:
    """Dense embedding через gpu_server.py (pplx-embed-v1, без instruction prefix)."""
    body = json.dumps({"inputs": [text], "normalize": True}).encode()
    req = urllib.request.Request(
        f"{EMBEDDING_URL}/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())[0]


def _search_dense(query_vector: list[float], top_k: int) -> list[dict]:
    """Qdrant dense-only search через HTTP API."""
    body = json.dumps({
        "query": query_vector,
        "using": DENSE_VECTOR_NAME,
        "limit": top_k,
        "with_payload": ["channel", "message_id", "text"],
    }).encode()
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())["result"]["points"]


class NaiveRetriever:
    """Dense-only retrieval — embed query → Qdrant dense search → top-K."""

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[RetrievalResult]:
        vec = _embed_query(query)
        points = _search_dense(vec, top_k)
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
