"""Weighted RRF fusion callable for LlamaIndex hybrid_fusion_fn.

Production pipeline использует BM25 weight=3.0, dense weight=1.0.
Стандартный LlamaIndex RRF (relative_score_fusion) использует равные веса.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeWithScore


def weighted_rrf_fusion(
    dense_nodes: list[NodeWithScore],
    sparse_nodes: list[NodeWithScore],
    *,
    dense_weight: float = 1.0,
    sparse_weight: float = 3.0,
    k: int = 60,
    **kwargs: Any,
) -> list[NodeWithScore]:
    """Weighted Reciprocal Rank Fusion — BM25-heavy для русского тек��та.

    BM25 weight 3:1 vs dense — русский текст с аббревиатурами (LLM, SSM, MoE)
    плохо ложится на dense-only embedding, BM25 ловит точные термины.
    """
    scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    for rank, node in enumerate(sparse_nodes):
        node_id = node.node.node_id
        scores[node_id] = scores.get(node_id, 0.0) + sparse_weight / (k + rank + 1)
        node_map[node_id] = node

    for rank, node in enumerate(dense_nodes):
        node_id = node.node.node_id
        scores[node_id] = scores.get(node_id, 0.0) + dense_weight / (k + rank + 1)
        if node_id not in node_map:
            node_map[node_id] = node

    # Сортируем по RRF score
    sorted_ids = sorted(scores, key=lambda nid: scores[nid], reverse=True)

    result = []
    for node_id in sorted_ids:
        node = node_map[node_id]
        result.append(NodeWithScore(node=node.node, score=scores[node_id]))

    return result
