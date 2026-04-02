"""
Agent guards: forced search, analytics short-circuit, tool repeat.

Извлечено из agent_service.py (SPEC-RAG-27 Phase 2).
Каждая функция возвращает data — yield points остаются в main loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from services.agent.finalization import build_final_payload
from services.agent.routing import load_policy as _load_policy
from services.agent.state import AgentState, RequestContext

logger = logging.getLogger(__name__)

# Tools, для которых повтор блокируется после 1 вызова
NO_REPEAT_TOOLS = frozenset({
    "entity_tracker", "arxiv_tracker", "hot_topics",
    "channel_expertise", "list_channels", "related_posts",
})

# search/rerank можно 2 раза (refinement), остальные — 1
MAX_REPEAT = 2


def check_forced_search(
    tool_calls: list[dict[str, Any]],
    agent_state: AgentState,
    content: str,
    query: str,
    step: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Проверяет нужен ли принудительный search.

    Если LLM не вызвала tools и search ещё не был — форсируем search.
    Bypass только для negative intent + refusal markers.

    Returns:
        (tool_calls, assistant_message) — обновлённые если forced search нужен.
        assistant_message будет None если forced search не нужен.
    """
    if tool_calls or agent_state.search_count > 0:
        return tool_calls, None
    if agent_state.navigation_answered or agent_state.analytics_done:
        return tool_calls, None

    refusal_markers = _load_policy("refusal_markers")
    negative_intent_markers = _load_policy("negative_intent_markers")
    is_refusal = content and any(m in content.lower() for m in refusal_markers)
    is_negative_intent = any(m in query.lower() for m in negative_intent_markers)

    if is_refusal and is_negative_intent:
        return tool_calls, None

    logger.warning(
        "Agent step %d: LLM не вызвала tools, search_count=0 → forced search",
        step,
    )
    forced_calls = [{
        "id": f"forced_search_{step}",
        "name": "search",
        "arguments": {"queries": [query]},
    }]
    forced_msg = {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [{
            "id": forced_calls[0]["id"],
            "type": "function",
            "function": {
                "name": "search",
                "arguments": json.dumps(forced_calls[0]["arguments"]),
            },
        }],
    }
    return forced_calls, forced_msg


def check_analytics_shortcircuit(
    tool_calls: list[dict[str, Any]],
    agent_state: AgentState,
    content: str,
    last_analytics_obs: str | None,
    request_id: str,
    step: int,
    ctx: RequestContext,
) -> dict[str, Any] | None:
    """Проверяет нужен ли analytics forced completion.

    Если analytics done, модель не вызвала tools, и есть ответ → финализируем.

    Returns:
        final_payload dict если нужна принудительная финализация, иначе None.
    """
    if tool_calls or not agent_state.analytics_done:
        return None

    direct_answer = content or last_analytics_obs or ""
    if not direct_answer:
        return None

    logger.info(
        "Analytics forced completion at step %d (content=%d, obs=%s)",
        step, len(content), bool(last_analytics_obs),
    )
    return build_final_payload(
        base_payload={"answer": direct_answer},
        answer=direct_answer,
        verify_res={},
        agent_state=agent_state,
        request_id=request_id,
        step=step,
        ctx=ctx,
    )


def should_block_repeat(
    tool_name: str,
    call_counts: dict[str, int],
) -> bool:
    """Проверяет, заблокирован ли повторный вызов tool.

    Обновляет call_counts in-place.
    """
    call_counts[tool_name] = call_counts.get(tool_name, 0) + 1
    max_allowed = 1 if tool_name in NO_REPEAT_TOOLS else MAX_REPEAT
    return call_counts[tool_name] > max_allowed


def build_analytics_repeat_payload(
    last_analytics_obs: str | None,
    agent_state: AgentState,
    request_id: str,
    step: int,
    ctx: RequestContext,
) -> dict[str, Any] | None:
    """Если analytics done и модель зациклилась — возвращает final payload.

    Returns None если analytics guard не сработал.
    """
    if not (agent_state.analytics_done and last_analytics_obs):
        return None

    logger.info(
        "Analytics guard forced completion: repeat after analytics_done",
    )
    return build_final_payload(
        base_payload={"answer": last_analytics_obs},
        answer=last_analytics_obs,
        verify_res={},
        agent_state=agent_state,
        request_id=request_id,
        step=step,
        ctx=ctx,
    )
