"""Tests for apply_action_state — pure reducer, no IO."""

from __future__ import annotations

from services.agent.state import apply_action_state
from tests.conftest import make_action, make_ctx

# --- query_plan ---

def test_query_plan_saves_plan():
    ctx = make_ctx()
    action = make_action("query_plan", {"plan": {"normalized_queries": ["q1", "q2"]}})
    apply_action_state(ctx, action)
    assert ctx.plan_summary == {"normalized_queries": ["q1", "q2"]}


def test_query_plan_empty_plan():
    ctx = make_ctx()
    action = make_action("query_plan", {"plan": None})
    apply_action_state(ctx, action)
    assert ctx.plan_summary == {}


# --- search ---

def test_search_increments_count_and_saves_hits():
    ctx = make_ctx()
    hits = [{"id": "1", "score": 0.9}]
    action = make_action("search", {"hits": hits, "route_used": "hybrid"})
    apply_action_state(ctx, action)
    assert ctx.agent_state.search_count == 1
    assert ctx.search_hits == hits
    assert ctx.search_route == "hybrid"


def test_temporal_search_is_search_like():
    ctx = make_ctx()
    action = make_action("temporal_search", {"hits": [{"id": "t1"}], "route_used": "temporal"})
    apply_action_state(ctx, action)
    assert ctx.agent_state.search_count == 1
    assert ctx.search_route == "temporal"


def test_channel_search_is_search_like():
    ctx = make_ctx()
    action = make_action("channel_search", {"hits": [], "route_used": "channel"})
    apply_action_state(ctx, action)
    assert ctx.agent_state.search_count == 1


def test_search_saves_strategy():
    ctx = make_ctx()
    action = make_action("search", {"hits": [], "strategy": "focused", "routing_source": "signal"})
    apply_action_state(ctx, action)
    assert ctx.agent_state.strategy == "focused"
    assert ctx.agent_state.routing_source == "signal"


def test_search_implicit_nuggets_from_multi_query():
    """When query_plan was not called, search subqueries become implicit nuggets."""
    ctx = make_ctx()
    action = make_action("search", {"hits": [{"id": "1"}]})
    action.input = {"queries": ["q1", "q2", "q3"]}
    apply_action_state(ctx, action)
    assert ctx.plan_summary is not None
    assert ctx.plan_summary["normalized_queries"] == ["q1", "q2", "q3"]


def test_search_no_implicit_nuggets_when_plan_exists():
    ctx = make_ctx()
    ctx.plan_summary = {"normalized_queries": ["existing"]}
    action = make_action("search", {"hits": []})
    action.input = {"queries": ["q1", "q2"]}
    apply_action_state(ctx, action)
    assert ctx.plan_summary == {"normalized_queries": ["existing"]}


# --- list_channels ---

def test_list_channels_sets_navigation():
    ctx = make_ctx()
    action = make_action("list_channels", {"channels": ["@ch1"]})
    apply_action_state(ctx, action)
    assert ctx.agent_state.navigation_answered is True


# --- analytics tools ---

def test_entity_tracker_sets_analytics_done():
    ctx = make_ctx()
    action = make_action("entity_tracker", {"top": [{"name": "GPT-5"}]})
    apply_action_state(ctx, action)
    assert ctx.agent_state.analytics_done is True
    assert ctx.agent_state.search_count == 0  # not search-like


def test_hot_topics_sets_analytics_done():
    ctx = make_ctx()
    action = make_action("hot_topics", {"topics": ["LLM agents"]})
    apply_action_state(ctx, action)
    assert ctx.agent_state.analytics_done is True


def test_channel_expertise_sets_analytics_done():
    ctx = make_ctx()
    action = make_action("channel_expertise", {"expertise": "ML"})
    apply_action_state(ctx, action)
    assert ctx.agent_state.analytics_done is True


def test_arxiv_tracker_with_hits_is_search_like():
    """arxiv_tracker(lookup) with hits → analytics_done AND search_count++."""
    ctx = make_ctx()
    hits = [{"id": "arxiv:2401.001"}]
    action = make_action("arxiv_tracker", {"hits": hits})
    apply_action_state(ctx, action)
    assert ctx.agent_state.analytics_done is True
    assert ctx.agent_state.search_count == 1
    assert ctx.search_hits == hits


def test_arxiv_tracker_without_hits_no_search_count():
    ctx = make_ctx()
    action = make_action("arxiv_tracker", {"top": [{"paper": "x"}]})
    apply_action_state(ctx, action)
    assert ctx.agent_state.analytics_done is True
    assert ctx.agent_state.search_count == 0


# --- rerank ---

def test_rerank_filters_and_preserves_order():
    ctx = make_ctx()
    ctx.search_hits = [
        {"id": "a", "score": 0.9},
        {"id": "b", "score": 0.7},
        {"id": "c", "score": 0.5},
    ]
    # CE returns indices 0 and 2 (skipping 1), order from CE: [0, 2]
    action = make_action("rerank", {
        "indices": [0, 2],
        "scores": [0.95, 0.60],
        "filtered_out": 1,
    })
    apply_action_state(ctx, action)
    # ColBERT order preserved: a first, then c
    assert len(ctx.search_hits) == 2
    assert ctx.search_hits[0]["id"] == "a"
    assert ctx.search_hits[0]["rerank_score"] == 0.95
    assert ctx.search_hits[1]["id"] == "c"
    assert ctx.search_hits[1]["rerank_score"] == 0.60


def test_rerank_empty_indices_keeps_hits():
    ctx = make_ctx()
    ctx.search_hits = [{"id": "a"}]
    action = make_action("rerank", {"indices": [], "scores": []})
    apply_action_state(ctx, action)
    # No kept_hits → original preserved (guard: `if kept_hits:`)
    assert len(ctx.search_hits) == 1


# --- compose_context ---

def test_compose_context_saves_citations():
    ctx = make_ctx()
    citations = [{"id": "1", "title": "Post"}]
    action = make_action("compose_context", {
        "citations": citations,
        "citation_coverage": 0.85,
    })
    apply_action_state(ctx, action)
    assert ctx.compose_citations == citations
    assert ctx.coverage_score == 0.85


# --- failed action ---

def test_failed_action_no_state_change():
    ctx = make_ctx()
    action = make_action("search", {"hits": [{"id": "1"}]}, ok=False)
    apply_action_state(ctx, action)
    assert ctx.agent_state.search_count == 0
    assert ctx.search_hits == []
