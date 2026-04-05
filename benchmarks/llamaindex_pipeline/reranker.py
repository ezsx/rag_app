"""Custom reranker for LlamaIndex — wraps gpu_server.py Qwen3-Reranker.

DEC-0043: Qwen3-Reranker-0.6B, seq-cls, chat template, padding_side=left, logit scoring.
Стандартный SentenceTransformerRerank не подходит — Qwen3 используе�� chat template формат.
"""

from __future__ import annotations

import json
import urllib.request

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from benchmarks.config import EMBEDDING_URL, RERANK_TOP_N


class QwenReranker(BaseNodePostprocessor):
    """Qwen3-Reranker via gpu_server.py /v1/rerank endpoint."""

    top_n: int = RERANK_TOP_N

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes[:self.top_n]

        query = query_bundle.query_str

        # Собираем тексты для reranking
        documents = []
        for node in nodes:
            text = node.node.get_content()
            if text:
                documents.append(text)
            else:
                documents.append("")

        # HTTP POST к gpu_server.py/rerank
        # gpu_server.py ожидает "texts", не "documents"; возвращает "score", не "relevance_score"
        body = json.dumps({
            "query": query,
            "texts": documents,
        }).encode()
        req = urllib.request.Request(
            f"{EMBEDDING_URL}/rerank",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        rerank_results = json.loads(resp.read())

        # gpu_server.py возвращает список {"index": i, "score": float}
        scored_nodes = []
        for result in rerank_results:
            idx = result["index"]
            score = result["score"]
            if idx < len(nodes):
                node_copy = NodeWithScore(
                    node=nodes[idx].node,
                    score=score,
                )
                scored_nodes.append(node_copy)

        # Сортируем по score descending, берём top_n
        scored_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_nodes[:self.top_n]

    @classmethod
    def class_name(cls) -> str:
        return "QwenReranker"
