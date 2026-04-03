"""Tests for phase-based tool visibility (get_step_tools)."""

from __future__ import annotations

from services.agent.state import AgentState
from services.agent.visibility import get_step_tools
from tests.conftest import make_ctx


def _tool_names(tools: list[dict]) -> set[str]:
    return {t["function"]["name"] for t in tools}


# ---------------------------------------------------------------------------
# PRE-SEARCH phase
# ---------------------------------------------------------------------------

def test_pre_search_base_tools():
    """Default phase: query_plan + search always visible."""
    state = AgentState()
    ctx = make_ctx("что нового в AI?")
    names = _tool_names(get_step_tools(state, ctx))
    assert "query_plan" in names
    assert "search" in names


# ---------------------------------------------------------------------------
# POST-SEARCH phase
# ---------------------------------------------------------------------------

def test_post_search_shows_rerank_compose_final():
    state = AgentState()
    state.search_count = 1
    ctx = make_ctx()
    names = _tool_names(get_step_tools(state, ctx))
    assert "rerank" in names
    assert "compose_context" in names
    assert "final_answer" in names
    assert "query_plan" not in names
    assert "search" not in names


# ---------------------------------------------------------------------------
# NAV-COMPLETE phase
# ---------------------------------------------------------------------------

def test_nav_complete_only_final_answer():
    state = AgentState()
    state.navigation_answered = True
    ctx = make_ctx()
    names = _tool_names(get_step_tools(state, ctx))
    assert names == {"final_answer"}


# ---------------------------------------------------------------------------
# ANALYTICS-COMPLETE phase
# ---------------------------------------------------------------------------

def test_analytics_complete_only_final_answer():
    state = AgentState()
    state.analytics_done = True
    ctx = make_ctx()
    names = _tool_names(get_step_tools(state, ctx))
    assert names == {"final_answer"}


# ---------------------------------------------------------------------------
# Cap at 5
# ---------------------------------------------------------------------------

def test_max_five_tools():
    state = AgentState()
    ctx = make_ctx("что нового в AI?")
    tools = get_step_tools(state, ctx)
    assert len(tools) <= 5


# ---------------------------------------------------------------------------
# Keyword routing (requires properly structured tool_keywords mock)
# ---------------------------------------------------------------------------

def test_keyword_routing_entity_tracker(monkeypatch):
    """Query with 'тренды' → entity_tracker becomes visible (PRE-SEARCH phase)."""
    # Override mock with properly structured tool_keywords for load_tool_keywords()
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
            "eviction_order": {"values": [
                "arxiv_tracker", "entity_tracker", "hot_topics",
                "channel_expertise", "cross_channel_compare",
                "summarize_channel", "list_channels",
            ]},
        },
    })
    state = AgentState()
    ctx = make_ctx("какие тренды в AI?")
    names = _tool_names(get_step_tools(state, ctx))
    assert "entity_tracker" in names


def test_analytics_done_and_search_done_shows_post_search():
    """If both analytics and search done → POST-SEARCH phase (search_done wins)."""
    state = AgentState()
    state.analytics_done = True
    state.search_count = 1
    ctx = make_ctx()
    names = _tool_names(get_step_tools(state, ctx))
    assert "rerank" in names
    assert "compose_context" in names
    assert "final_answer" in names
