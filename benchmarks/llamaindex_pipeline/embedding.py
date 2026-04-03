"""Custom embedding for LlamaIndex — wraps gpu_server.py pplx-embed-v1.

DEC-0042: pplx-embed-v1, bf16, mean pooling, без instruction prefix.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from llama_index.core.embeddings import BaseEmbedding

from benchmarks.config import EMBEDDING_URL


class PplxEmbedding(BaseEmbedding):
    """pplx-embed-v1 via gpu_server.py HTTP API."""

    model_name: str = "pplx-embed-v1"
    embed_batch_size: int = 10

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """HTTP POST к gpu_server.py/embed."""
        body = json.dumps({"inputs": texts, "normalize": True}).encode()
        req = urllib.request.Request(
            f"{EMBEDDING_URL}/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._call_api([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        # Sync fallback — benchmark не требует async performance
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._call_api([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._call_api(texts)

    @classmethod
    def class_name(cls) -> str:
        return "PplxEmbedding"
