"""
LLM step: вызов модели + парсинг ответа.

Извлечено из agent_service.py (SPEC-RAG-27 Phase 2).
Helper возвращает data, yield points остаются в main loop.
"""

from __future__ import annotations

import logging
from typing import Any

from core.observability import observe_span
from core.settings import Settings
from services.agent.formatting import extract_tool_calls, trim_messages
from services.agent.state import AgentState

logger = logging.getLogger(__name__)


class LLMStepResult:
    """Результат одного LLM step."""

    __slots__ = ("assistant_message", "content", "finish_reason", "tool_calls")

    def __init__(
        self,
        assistant_message: dict[str, Any],
        finish_reason: str,
        content: str,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        self.assistant_message = assistant_message
        self.finish_reason = finish_reason
        self.content = content
        self.tool_calls = tool_calls


def call_llm_step(
    llm: Any,
    messages: list[dict[str, Any]],
    step_tools: list[dict[str, Any]],
    visible_tool_names: list[str],
    step: int,
    settings: Settings,
    agent_state: AgentState,
) -> LLMStepResult:
    """Вызывает LLM chat/completions и парсит ответ.

    Включает retry с обрезанной историей при ошибке.
    Observability span создаётся внутри.
    """
    trimmed = trim_messages(messages)
    expect_final = (
        agent_state.coverage > 0
        or agent_state.analytics_done
        or agent_state.navigation_answered
    )
    max_tokens = (
        settings.agent_final_max_tokens if expect_final
        else settings.agent_tool_max_tokens
    )
    phase = (
        "final" if expect_final
        else ("post_search" if agent_state.search_count > 0 else "pre_search")
    )

    response = _do_llm_call(
        llm, trimmed, messages, step_tools, max_tokens, step, phase, settings, agent_state,
    )

    choice = (response.get("choices") or [{}])[0]
    assistant_message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason", "unknown")
    content = (assistant_message.get("content") or "").strip()
    tool_calls = extract_tool_calls(
        assistant_message, visible_tools=set(visible_tool_names),
    )

    logger.debug(
        "Agent step %d finish=%s content_len=%d tool_calls=%d",
        step, finish_reason, len(content), len(tool_calls),
    )

    return LLMStepResult(
        assistant_message=assistant_message,
        finish_reason=finish_reason,
        content=content,
        tool_calls=tool_calls,
    )


def _do_llm_call(
    llm: Any,
    trimmed: list[dict],
    original_messages: list[dict],
    step_tools: list[dict],
    max_tokens: int,
    step: int,
    phase: str,
    settings: Settings,
    agent_state: AgentState,
) -> dict[str, Any]:
    """LLM вызов с retry и observability span."""
    with observe_span(f"llm_step_{step}_{phase}", metadata={
        "step": step,
        "phase": phase,
        "search_count": agent_state.search_count,
        "refinement_count": agent_state.refinement_count,
        "coverage": agent_state.coverage,
        "visible_tools": [
            t["function"]["name"] for t in step_tools if "function" in t
        ],
    }) as span:
        sent_messages = trimmed
        try:
            response = llm.chat_completion(
                messages=trimmed,
                tools=step_tools,
                max_tokens=max_tokens,
                temperature=settings.agent_tool_temp,
                top_p=settings.agent_tool_top_p,
                top_k=settings.agent_tool_top_k,
                presence_penalty=settings.agent_tool_presence_penalty,
                seed=42,
            )
        except Exception as llm_exc:  # broad: adapter boundary
            logger.warning(
                "LLM call failed at step %d: %s — retrying with trimmed history",
                step, llm_exc,
            )
            sent_messages = original_messages[:2] + original_messages[-2:]
            try:
                response = llm.chat_completion(
                    messages=sent_messages,
                    tools=step_tools,
                    max_tokens=max_tokens,
                    temperature=settings.agent_tool_temp,
                    seed=42,
                )
            except Exception:  # broad: adapter boundary
                raise llm_exc from None

        if span:
            choice = (response.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            span.update(
                input={"message_count": len(sent_messages)},
                output={
                    "finish_reason": choice.get("finish_reason", "unknown"),
                    "content_len": len(msg.get("content") or ""),
                    "tool_calls": [
                        tc.get("function", {}).get("name", "?")
                        for tc in (msg.get("tool_calls") or [])
                    ],
                },
            )
        return response
