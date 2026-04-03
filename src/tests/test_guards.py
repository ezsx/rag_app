"""Tests for agent guards: forced search, analytics short-circuit, repeat blocker."""

from __future__ import annotations

from unittest.mock import patch

from services.agent.guards import (
    check_analytics_shortcircuit,
    check_forced_search,
    should_block_repeat,
)
from services.agent.state import AgentState
from tests.conftest import make_ctx

# ---------------------------------------------------------------------------
# check_forced_search
# ---------------------------------------------------------------------------

def test_forced_search_when_no_tools_no_search():
    """LLM returned no tools, search_count=0 → forced search injected."""
    state = AgentState()
    calls, msg = check_forced_search([], state, "some text", "что нового в AI?", step=1)
    assert len(calls) == 1
    assert calls[0]["name"] == "search"
    assert msg is not None
    assert msg["role"] == "assistant"


def test_no_forced_search_when_tools_present():
    state = AgentState()
    existing = [{"id": "c1", "name": "query_plan", "arguments": {}}]
    calls, msg = check_forced_search(existing, state, "", "query", step=1)
    assert calls is existing
    assert msg is None


def test_no_forced_search_when_search_already_done():
    state = AgentState()
    state.search_count = 1
    calls, msg = check_forced_search([], state, "", "query", step=2)
    assert calls == []
    assert msg is None


def test_no_forced_search_when_analytics_done():
    state = AgentState()
    state.analytics_done = True
    calls, msg = check_forced_search([], state, "", "query", step=1)
    assert calls == []
    assert msg is None


def test_no_forced_search_when_navigation_answered():
    state = AgentState()
    state.navigation_answered = True
    calls, msg = check_forced_search([], state, "", "query", step=1)
    assert calls == []
    assert msg is None


def test_forced_search_bypass_on_refusal_and_negative_intent():
    """Both refusal marker in content AND negative intent in query → bypass."""
    state = AgentState()
    refusal_markers = ["нет в базе"]
    negative_markers = ["существует ли"]
    with patch("services.agent.guards._load_policy", side_effect=lambda name: {
        "refusal_markers": refusal_markers,
        "negative_intent_markers": negative_markers,
    }[name]):
        calls, msg = check_forced_search(
            [], state,
            content="Информация нет в базе данных",
            query="существует ли статья X?",
            step=1,
        )
    assert calls == []
    assert msg is None


def test_forced_search_not_bypassed_on_refusal_only():
    """Refusal without negative intent → still forced."""
    state = AgentState()
    with patch("services.agent.guards._load_policy", side_effect=lambda name: {
        "refusal_markers": ["нет в базе"],
        "negative_intent_markers": ["существует ли"],
    }[name]):
        calls, msg = check_forced_search(
            [], state,
            content="нет в базе",
            query="расскажи про GPT-5",  # no negative intent
            step=1,
        )
    assert len(calls) == 1
    assert msg is not None


# ---------------------------------------------------------------------------
# check_analytics_shortcircuit
# ---------------------------------------------------------------------------

def test_analytics_shortcircuit_returns_payload():
    """analytics_done + no tools + content → final payload."""
    state = AgentState()
    state.analytics_done = True
    ctx = make_ctx()
    result = check_analytics_shortcircuit(
        tool_calls=[],
        agent_state=state,
        content="Top entities: GPT-5",
        last_analytics_obs="obs data",
        request_id="r1",
        step=2,
        ctx=ctx,
    )
    assert result is not None
    assert "answer" in result
    assert "Top entities: GPT-5" in result["answer"]


def test_analytics_shortcircuit_none_when_tools():
    state = AgentState()
    state.analytics_done = True
    ctx = make_ctx()
    result = check_analytics_shortcircuit(
        tool_calls=[{"id": "c", "name": "search", "arguments": {}}],
        agent_state=state,
        content="text",
        last_analytics_obs=None,
        request_id="r1",
        step=2,
        ctx=ctx,
    )
    assert result is None


def test_analytics_shortcircuit_none_when_not_done():
    state = AgentState()
    ctx = make_ctx()
    result = check_analytics_shortcircuit(
        tool_calls=[],
        agent_state=state,
        content="text",
        last_analytics_obs=None,
        request_id="r1",
        step=1,
        ctx=ctx,
    )
    assert result is None


def test_analytics_shortcircuit_uses_obs_as_fallback():
    """When content is empty, falls back to last_analytics_obs."""
    state = AgentState()
    state.analytics_done = True
    ctx = make_ctx()
    result = check_analytics_shortcircuit(
        tool_calls=[],
        agent_state=state,
        content="",
        last_analytics_obs="fallback observation",
        request_id="r1",
        step=2,
        ctx=ctx,
    )
    assert result is not None
    assert "fallback observation" in result["answer"]


def test_analytics_shortcircuit_none_when_no_content():
    """Both content and obs empty → None."""
    state = AgentState()
    state.analytics_done = True
    ctx = make_ctx()
    result = check_analytics_shortcircuit(
        tool_calls=[],
        agent_state=state,
        content="",
        last_analytics_obs=None,
        request_id="r1",
        step=2,
        ctx=ctx,
    )
    assert result is None


# ---------------------------------------------------------------------------
# should_block_repeat
# ---------------------------------------------------------------------------

def test_entity_tracker_blocked_after_one():
    counts: dict[str, int] = {}
    assert should_block_repeat("entity_tracker", counts) is False  # 1st
    assert should_block_repeat("entity_tracker", counts) is True   # 2nd


def test_search_allowed_twice():
    counts: dict[str, int] = {}
    assert should_block_repeat("search", counts) is False  # 1st
    assert should_block_repeat("search", counts) is False  # 2nd
    assert should_block_repeat("search", counts) is True   # 3rd


def test_list_channels_blocked_after_one():
    counts: dict[str, int] = {}
    assert should_block_repeat("list_channels", counts) is False
    assert should_block_repeat("list_channels", counts) is True


def test_rerank_allowed_twice():
    counts: dict[str, int] = {}
    assert should_block_repeat("rerank", counts) is False
    assert should_block_repeat("rerank", counts) is False
    assert should_block_repeat("rerank", counts) is True
