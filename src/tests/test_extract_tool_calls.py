"""Tests for extract_tool_calls() parsing and filtering logic."""

from __future__ import annotations

from services.agent.formatting import extract_tool_calls


def test_single_tool_call_with_json_arguments() -> None:
    msg = {
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": '{"queries": ["test"], "k": 5}',
                },
            }
        ]
    }
    result = extract_tool_calls(msg)
    assert len(result) == 1
    assert result[0]["name"] == "search"
    assert result[0]["arguments"] == {"queries": ["test"], "k": 5}
    assert result[0]["id"] == "call_1"


def test_filters_invisible_tools() -> None:
    msg = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "search", "arguments": "{}"}},
            {"id": "c2", "function": {"name": "final_answer", "arguments": "{}"}},
            {"id": "c3", "function": {"name": "rerank", "arguments": "{}"}},
        ]
    }
    result = extract_tool_calls(msg, visible_tools={"search", "rerank"})
    assert len(result) == 2
    names = [r["name"] for r in result]
    assert "search" in names
    assert "rerank" in names
    assert "final_answer" not in names


def test_malformed_json_arguments_no_crash() -> None:
    msg = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "search", "arguments": "{invalid json!!"}}
        ]
    }
    result = extract_tool_calls(msg)
    assert len(result) == 1
    assert result[0]["name"] == "search"
    assert result[0]["arguments"] == {"raw_input": "{invalid json!!"}


def test_empty_tool_calls() -> None:
    assert extract_tool_calls({}) == []
    assert extract_tool_calls({"tool_calls": []}) == []
    assert extract_tool_calls({"tool_calls": None}) == []


def test_multiple_tool_calls() -> None:
    msg = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "query_plan", "arguments": {"k": 3}}},
            {"id": "c2", "function": {"name": "search", "arguments": {"queries": ["a"]}}},
        ]
    }
    result = extract_tool_calls(msg)
    assert len(result) == 2
    assert result[0]["name"] == "query_plan"
    assert result[1]["name"] == "search"


def test_dict_arguments_passed_through() -> None:
    msg = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "search", "arguments": {"q": "hello"}}}
        ]
    }
    result = extract_tool_calls(msg)
    assert result[0]["arguments"] == {"q": "hello"}


def test_non_dict_items_skipped() -> None:
    msg = {"tool_calls": ["not a dict", 42, None]}
    result = extract_tool_calls(msg)
    assert result == []


def test_missing_tool_name_skipped() -> None:
    msg = {
        "tool_calls": [
            {"id": "c1", "function": {"name": None, "arguments": "{}"}},
            {"id": "c2", "function": {"arguments": "{}"}},
        ]
    }
    result = extract_tool_calls(msg)
    assert result == []


def test_visible_tools_none_allows_all() -> None:
    """When visible_tools is None, no filtering applied."""
    msg = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "search", "arguments": "{}"}},
            {"id": "c2", "function": {"name": "final_answer", "arguments": "{}"}},
        ]
    }
    result = extract_tool_calls(msg, visible_tools=None)
    assert len(result) == 2
