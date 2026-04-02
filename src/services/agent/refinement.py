"""
Refinement flow — targeted search по непокрытым nuggets.

Извлечено из agent_service.py (SPEC-RAG-27 Phase 2).
Единый refinement flow вместо двух идентичных блоков.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.settings import Settings
from schemas.agent import AgentAction, AgentStepEvent
from services.agent.executor import execute_action
from services.agent.formatting import format_observation, tool_message_for_history
from services.agent.state import AgentState, RequestContext, apply_action_state
from services.tools.tool_runner import ToolRunner

logger = logging.getLogger(__name__)


async def run_refinement(
    query: str,
    agent_state: AgentState,
    request_id: str,
    step: int,
    messages: list[dict[str, Any]],
    conversation_history: list[str],
    ctx: RequestContext,
    tool_runner: ToolRunner,
    settings: Settings,
    label: str = "refinement",
) -> tuple[list[AgentStepEvent], bool]:
    """Единый refinement flow — дедупликация двух одинаковых блоков.

    Выполняет targeted refinement search и возвращает список событий.
    yield points остаются в stream_agent_response.

    Returns:
        (events, should_repeat_step)
    """
    agent_state.refinement_count += 1
    events: list[AgentStepEvent] = []

    refine_thought = (
        "Покрытие контекста недостаточно. Выполняю дополнительный поиск."
        if label == "refinement"
        else "Ответ требует дополнительной проверки. Выполняю ещё один поиск перед финализацией."
    )
    events.append(AgentStepEvent(
        type="thought",
        data={
            "content": refine_thought,
            "step": step,
            "request_id": request_id,
            "system_generated": True,
            label: True,
            "refinement_count": agent_state.refinement_count,
        },
    ))
    conversation_history.append(f"Thought: {refine_thought}")

    refinement_actions = await perform_refinement(
        query, agent_state, request_id, step, ctx, tool_runner, settings,
    )

    for action in refinement_actions:
        events.append(AgentStepEvent(
            type="tool_invoked",
            data={
                "tool": action.tool,
                "input": action.input,
                "step": step,
                "request_id": request_id,
                "system_generated": True,
                label: True,
            },
        ))

        apply_action_state(ctx, action)
        obs = format_observation(action.output, action.tool)

        events.append(AgentStepEvent(
            type="observation",
            data={
                "content": obs,
                "success": action.output.ok,
                "step": step,
                "request_id": request_id,
                "took_ms": action.output.meta.took_ms,
                "system_generated": True,
                label: True,
            },
        ))
        conversation_history.append(
            f"Action: {action.tool} {json.dumps(action.input, ensure_ascii=False)}"
        )
        conversation_history.append(f"Observation: {obs}")
        messages.append(
            tool_message_for_history(
                {"id": f"{label}-{step}-{action.tool}",
                 "name": action.tool, "arguments": action.input},
                action.tool,
                action.output.data if action.output.ok
                else {"error": action.output.meta.error},
            )
        )

        if action.tool == "compose_context" and action.output.ok:
            refined_coverage = float(
                action.output.data.get("citation_coverage", 0.0) or 0.0
            )
            agent_state.coverage = refined_coverage
            ctx.coverage_score = refined_coverage
            events.append(AgentStepEvent(
                type="citations",
                data={
                    "citations": action.output.data.get("citations", []),
                    "coverage": refined_coverage,
                    "step": step,
                    "request_id": request_id,
                    "system_generated": True,
                    label: True,
                },
            ))
            if (refined_coverage < 0.50
                    and agent_state.refinement_count >= settings.max_refinements):
                agent_state.low_coverage_disclaimer = True

    return events, True


async def perform_refinement(
    query: str,
    agent_state: AgentState,
    request_id: str,
    step: int,
    ctx: RequestContext,
    tool_runner: ToolRunner,
    settings: Settings,
) -> list[AgentAction]:
    """Targeted refinement: ищет конкретно то чего не хватает (SEAL-RAG + LANCER).

    Использует uncovered_nuggets из nugget coverage вместо повторного
    поиска по всему запросу. Если nuggets нет — fallback на plan subqueries.
    """
    plan = ctx.plan_summary or {}
    uncovered = ctx.uncovered_nuggets
    if uncovered:
        queries = list(uncovered)
        logger.info(
            "Targeted refinement: %d uncovered nuggets: %s",
            len(uncovered), [n[:50] for n in uncovered],
        )
    else:
        queries = list(plan.get("normalized_queries") or [query])
    if query not in queries:
        queries.insert(0, query)

    filters = {}
    plan_filters = plan.get("metadata_filters")
    if isinstance(plan_filters, dict):
        filters = {k: v for k, v in plan_filters.items() if v is not None}

    actions: list[AgentAction] = []
    search_action = await execute_action(
        tool_name="search",
        params={"queries": queries, "filters": filters, "k": 20, "route": "hybrid"},
        request_id=request_id,
        step=step,
        ctx=ctx,
        tool_runner=tool_runner,
        settings=settings,
    )
    if search_action is None:
        return actions

    actions.append(search_action)
    if not search_action.output.ok:
        return actions

    apply_action_state(ctx, search_action)
    compose_action = await execute_action(
        tool_name="compose_context",
        params={"hit_ids": []},
        request_id=request_id,
        step=step,
        ctx=ctx,
        tool_runner=tool_runner,
        settings=settings,
    )
    if compose_action is not None:
        actions.append(compose_action)
    return actions
