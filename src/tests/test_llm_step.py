"""Tests for call_llm_step() — LLM call + response parsing."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from services.agent.llm_step import LLMStepResult, call_llm_step
from services.agent.state import AgentState


def _make_settings(**overrides: Any) -> MagicMock:
    defaults = dict(
        agent_tool_max_tokens=384,
        agent_final_max_tokens=1024,
        agent_tool_temp=0.7,
        agent_tool_top_p=0.8,
        agent_tool_top_k=20,
        agent_tool_presence_penalty=1.5,
    )
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _make_llm_response(
    content: str = "",
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


@patch("services.agent.llm_step.observe_span")
def test_happy_path_tool_call(mock_span: MagicMock) -> None:
    mock_span.return_value.__enter__ = MagicMock(return_value=None)
    mock_span.return_value.__exit__ = MagicMock(return_value=False)

    llm = MagicMock()
    llm.chat_completion.return_value = _make_llm_response(
        tool_calls=[
            {"id": "tc1", "function": {"name": "search", "arguments": '{"queries": ["test"]}'}}
        ],
        finish_reason="tool_calls",
    )
    state = AgentState()
    result = call_llm_step(
        llm=llm,
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        step_tools=[{"type": "function", "function": {"name": "search"}}],
        visible_tool_names=["search"],
        step=1,
        settings=_make_settings(),
        agent_state=state,
    )

    assert isinstance(result, LLMStepResult)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search"
    assert result.finish_reason == "tool_calls"
    llm.chat_completion.assert_called_once()


@patch("services.agent.llm_step.observe_span")
def test_content_extraction(mock_span: MagicMock) -> None:
    mock_span.return_value.__enter__ = MagicMock(return_value=None)
    mock_span.return_value.__exit__ = MagicMock(return_value=False)

    llm = MagicMock()
    llm.chat_completion.return_value = _make_llm_response(
        content="Here is your answer about AI trends.",
        finish_reason="stop",
    )
    state = AgentState()
    state.coverage = 0.8  # triggers expect_final → agent_final_max_tokens
    result = call_llm_step(
        llm=llm,
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        step_tools=[],
        visible_tool_names=[],
        step=2,
        settings=_make_settings(),
        agent_state=state,
    )

    assert result.content == "Here is your answer about AI trends."
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    # Should use agent_final_max_tokens because coverage > 0
    call_kwargs = llm.chat_completion.call_args
    assert call_kwargs.kwargs.get("max_tokens") == 1024


@patch("services.agent.llm_step.observe_span")
def test_retry_on_first_failure(mock_span: MagicMock) -> None:
    mock_span.return_value.__enter__ = MagicMock(return_value=None)
    mock_span.return_value.__exit__ = MagicMock(return_value=False)

    llm = MagicMock()
    llm.chat_completion.side_effect = [
        RuntimeError("context too long"),
        _make_llm_response(content="recovered", finish_reason="stop"),
    ]

    state = AgentState()
    result = call_llm_step(
        llm=llm,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "mid1"},
            {"role": "tool", "name": "search", "content": "{}"},
        ],
        step_tools=[],
        visible_tool_names=[],
        step=3,
        settings=_make_settings(),
        agent_state=state,
    )

    assert result.content == "recovered"
    assert llm.chat_completion.call_count == 2


@patch("services.agent.llm_step.observe_span")
def test_both_retries_fail_raises(mock_span: MagicMock) -> None:
    mock_span.return_value.__enter__ = MagicMock(return_value=None)
    mock_span.return_value.__exit__ = MagicMock(return_value=False)

    original_error = RuntimeError("original error")
    llm = MagicMock()
    llm.chat_completion.side_effect = [original_error, RuntimeError("retry also failed")]

    state = AgentState()
    with pytest.raises(RuntimeError, match="original error"):
        call_llm_step(
            llm=llm,
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
            step_tools=[],
            visible_tool_names=[],
            step=1,
            settings=_make_settings(),
            agent_state=state,
        )
