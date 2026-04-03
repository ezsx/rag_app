"""Unified interfaces for all benchmark pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RetrievalResult:
    """Единый формат результата retrieval."""

    doc_id: str          # "channel:message_id"
    score: float
    channel: str
    message_id: int
    text: str | None = None


@dataclass
class AgentResult:
    """Единый формат результата agent."""

    answer: str
    docs: list[RetrievalResult] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    latency: float = 0.0


class RetrieverProtocol(Protocol):
    """Interface for all retrieval pipelines."""

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        ...


class AgentProtocol(Protocol):
    """Interface for all agent pipelines."""

    def run(self, query: str) -> AgentResult:
        ...
