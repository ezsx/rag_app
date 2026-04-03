"""Test factories and fixtures for agent tests."""

from __future__ import annotations

from typing import Any

import pytest

from schemas.agent import AgentAction, ToolMeta, ToolResponse
from services.agent.state import RequestContext


def make_ctx(query: str = "test query", **overrides: Any) -> RequestContext:
    """Factory for RequestContext with sensible defaults."""
    defaults: dict[str, Any] = dict(
        request_id="test-req", query=query, original_query=query,
    )
    defaults.update(overrides)
    return RequestContext(**defaults)


def make_action(tool: str, data: dict[str, Any], ok: bool = True, step: int = 1) -> AgentAction:
    """Factory for AgentAction with minimal required fields."""
    return AgentAction(
        step=step,
        tool=tool,
        input=data,
        output=ToolResponse(
            ok=ok,
            data=data if ok else {},
            meta=ToolMeta(took_ms=10, error=None if ok else "test_error"),
        ),
    )


@pytest.fixture(autouse=True)
def _mock_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal routing data so guards/visibility don't depend on datasets/tool_keywords.json."""
    monkeypatch.setattr("services.agent.routing._ROUTING_DATA", {
        "tool_keywords": {
            "entity_tracker": {"keywords": ["популярн", "тренды", "упомина"]},
            "arxiv_tracker": {"keywords": ["arxiv", "статей", "paper"]},
            "hot_topics": {"keywords": ["дайджест", "горяч", "тренд"]},
            "channel_expertise": {"keywords": ["экспертиза", "о чём пишет"]},
            "temporal_search": {"keywords": ["вчера", "неделю", "месяц"]},
            "channel_search": {"keywords": ["канал", "@"]},
            "cross_channel_compare": {"keywords": ["сравни", "versus", "vs"]},
            "summarize_channel": {"keywords": ["резюме канала", "обзор канала"]},
            "list_channels": {"keywords": ["какие каналы", "сколько постов"]},
        },
        "agent_policies": {
            "refusal_markers": {"values": ["нет в базе", "не найден", "отсутствует"]},
            "negative_intent_markers": {"values": ["существует ли", "был ли", "выходила ли"]},
            "eviction_order": {"values": [
                "arxiv_tracker", "entity_tracker", "hot_topics",
                "channel_expertise", "cross_channel_compare",
                "summarize_channel", "list_channels",
            ]},
        },
    })
