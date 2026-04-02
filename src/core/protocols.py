"""
Protocol interfaces — explicit contracts между слоями.

Для документации архитектуры в коде. runtime_checkable для isinstance проверок,
но mypy enforcement — необязателен (duck typing + @lru_cache singletons).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from schemas.search import Candidate, SearchPlan


@runtime_checkable
class Retriever(Protocol):
    """Контракт retriever-слоя (HybridRetriever)."""

    def search_with_plan(self, query_text: str, plan: SearchPlan) -> list[Candidate]: ...


class EmbeddingClient(Protocol):
    """Контракт embedding service (TEIEmbeddingClient)."""

    async def embed_query(self, text: str) -> list[float]: ...


class RerankerClient(Protocol):
    """Контракт reranker service (TEIRerankerClient)."""

    async def rerank(self, query: str, passages: list[str]) -> list[float]: ...


class LLMClient(Protocol):
    """Контракт LLM (LlamaServerClient)."""

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
