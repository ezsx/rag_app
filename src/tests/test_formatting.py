"""Tests for format_observation() dispatch table and error handling."""

from __future__ import annotations

import pytest

from schemas.agent import ToolMeta, ToolResponse
from services.agent.formatting import format_observation


def _ok_resp(data: dict) -> ToolResponse:
    return ToolResponse(ok=True, data=data, meta=ToolMeta(took_ms=10))


def _err_resp(error: str = "timeout") -> ToolResponse:
    return ToolResponse(ok=False, data={}, meta=ToolMeta(took_ms=0, error=error))


# ── Parametrized dispatch ────────────────────────────────────────

@pytest.mark.parametrize("tool,data,expected_substr", [
    # _fmt_search: "Found N documents"
    ("search", {"hits": [{"id": "abc"}], "total_found": 1, "route_used": "hybrid"}, "found 1 documents"),
    ("temporal_search", {"hits": [], "total_found": 0, "route_used": "bm25"}, "found 0 documents"),
    ("channel_search", {"hits": [{"id": "x"}], "total_found": 5}, "found 1 documents"),
    # _fmt_rerank: "Reranked ... documents"
    ("rerank", {"scores": [0.95, 0.8], "indices": [0, 1], "kept": 2, "filtered_out": 0}, "reranked"),
    # _fmt_compose: "Composed context with N citations"
    ("compose_context", {"citations": [{"id": "c1"}], "citation_coverage": 0.85, "contexts": ["a"]}, "composed context"),
    # _fmt_verify: "Verification:"
    ("verify", {"verified": True, "confidence": 0.9, "threshold": 0.6, "documents_found": 3}, "verification"),
    # _fmt_query_plan: "Plan: N queries"
    ("query_plan", {"plan": {"normalized_queries": ["q1", "q2"], "k_per_query": 5, "fusion": "rrf"}}, "plan: 2 queries"),
    # _fmt_final_answer: "Final answer prepared (N chars)"
    ("final_answer", {"answer": "test answer"}, "final answer prepared"),
    # _fmt_entity_tracker: "entity_tracker(mode):"
    ("entity_tracker", {"mode": "top", "results": [], "data": [1, 2], "summary": ""}, "entity_tracker(top)"),
    # _fmt_arxiv_tracker: "arxiv_tracker(mode):"
    ("arxiv_tracker", {"mode": "lookup", "data": [], "summary": "found 5 papers"}, "arxiv_tracker(lookup)"),
    # _fmt_hot_topics: "period:"
    ("hot_topics", {"period": "2026-W14", "post_count": 42, "topics": []}, "period: 2026-w14"),
    # _fmt_channel_expertise: "Channel ..."
    ("channel_expertise", {"channel": "@test", "authority_score": 0.9, "profile_summary": "ML news"}, "channel @test"),
    # _fmt_list_channels: "N channels:"
    ("list_channels", {"channels": [{"channel": "@a", "count": 10}], "total": 1}, "1 channels"),
    # _fmt_related_posts: "Found N related posts"
    ("related_posts", {"hits": [{"id": "r1"}], "source_id": "s1"}, "found 1 related posts"),
    # _fmt_cross_channel: "Compared N channels"
    ("cross_channel_compare", {"groups": [{"channel": "@a"}, {"channel": "@b"}], "topic": "LLM", "total_found": 10}, "compared 2 channels"),
    # _fmt_summarize_channel: "Channel ... posts"
    ("summarize_channel", {"channel": "@news", "period": "2026-W14", "post_count": 7}, "channel @news"),
    # _fmt_default fallback for unknown tools
    ("unknown_tool", {"key": "val"}, "key: val"),
])
def test_format_observation_dispatch(tool: str, data: dict, expected_substr: str) -> None:
    result = format_observation(_ok_resp(data), tool)
    assert expected_substr in result.lower(), f"Expected '{expected_substr}' in: {result}"


# ── Error responses ──────────────────────────────────────────────

def test_format_observation_error_with_message() -> None:
    result = format_observation(_err_resp("timeout"), "search")
    assert "ошибка" in result.lower()
    assert "timeout" in result.lower()


def test_format_observation_error_no_message() -> None:
    result = format_observation(_err_resp(None), "search")  # type: ignore[arg-type]
    assert "ошибка" in result.lower() or "неизвестная" in result.lower()


# ── Edge cases ───────────────────────────────────────────────────

def test_format_observation_empty_data() -> None:
    resp = ToolResponse(ok=True, data={}, meta=ToolMeta(took_ms=0))
    result = format_observation(resp, "search")
    assert "пустые данные" in result.lower()


def test_format_observation_entity_tracker_with_summary() -> None:
    data = {"mode": "compare", "summary": "GPT vs Claude: 42 mentions each"}
    result = format_observation(_ok_resp(data), "entity_tracker")
    assert "entity_tracker(compare)" in result
    assert "GPT vs Claude" in result


def test_format_observation_rerank_empty_scores() -> None:
    data = {"scores": [], "indices": [], "kept": 0, "filtered_out": 0}
    result = format_observation(_ok_resp(data), "rerank")
    assert "reranked" in result.lower()


def test_format_observation_default_handles_nested_types() -> None:
    data = {"items": [1, 2, 3], "nested": {"a": 1}, "scalar": 42}
    result = format_observation(_ok_resp(data), "nonexistent_tool")
    assert "items: 3" in result  # list → len
    assert "nested: object" in result  # dict → "object"
    assert "scalar: 42" in result
