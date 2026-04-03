"""
Protocol interfaces -- explicit contracts between layers.

runtime_checkable for isinstance checks; mypy enforcement is optional
(duck typing + @lru_cache singletons).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from schemas.search import Candidate, SearchPlan


@runtime_checkable
class Retriever(Protocol):
    """Retriever layer contract (HybridRetriever)."""

    def search_with_plan(self, query_text: str, plan: SearchPlan) -> list[Candidate]: ...


class EmbeddingClient(Protocol):
    """Embedding service contract (TEIEmbeddingClient)."""

    async def embed_query(self, text: str) -> list[float]: ...


class RerankerClient(Protocol):
    """Reranker service contract (TEIRerankerClient)."""

    async def rerank(self, query: str, passages: list[str]) -> list[float]: ...


class LLMClient(Protocol):
    """LLM client contract (LlamaServerClient)."""

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
