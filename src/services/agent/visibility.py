"""
SPEC-RAG-20c Step 4: Phase-based tool visibility.

Determines which tools the LLM sees at each step:
PRE-SEARCH → POST-SEARCH → NAV-COMPLETE → ANALYTICS-COMPLETE.
"""

from __future__ import annotations

from typing import Any

from services.agent.prompts import AGENT_TOOLS
from services.agent.routing import load_policy, load_tool_keywords
from services.agent.state import AgentState, RequestContext


def get_step_tools(agent_state: AgentState, ctx: RequestContext) -> list[dict[str, Any]]:
    """Phase-based visibility — фиксированные наборы по фазе агента.

    SPEC-RAG-13 + SPEC-RAG-15: 15 tools, max 5 видимых.
    Фазы:
    1. PRE-SEARCH: planning + search + analytics (keyword-triggered)
    2. POST-SEARCH: enrichment + synthesis
    3. NAV-COMPLETE: only final_answer
    4. ANALYTICS-COMPLETE: only final_answer (ранее показывали search+analytics → loop)
    """
    search_done = agent_state.search_count > 0
    nav_done = agent_state.navigation_answered
    analytics_done = agent_state.analytics_done
    signals = ctx.query_signals
    original_query = ctx.original_query or ""
    query_lower = original_query.lower()

    if nav_done and not search_done:
        # NAV-COMPLETE: list_channels ответил, search не нужен → сразу final
        visible_names = {"final_answer"}
    elif analytics_done and not search_done:
        # ANALYTICS-COMPLETE: данные получены → только final_answer
        # (зеркалит NAV-COMPLETE; ранее показывали search+analytics
        #  и модель пыталась повторно вызвать analytics → loop)
        visible_names = {"final_answer"}
    elif search_done:
        # POST-SEARCH: rerank, compose_context, final_answer + enrichment
        visible_names = {
            "rerank", "compose_context", "final_answer",
            "related_posts",
        }
    else:
        # PRE-SEARCH: query_plan + fallback search (всегда)
        visible_names = {"query_plan", "search"}

        # Signal-based: добавляем релевантный specialized search
        if signals:
            if signals.date_from or signals.strategy_hint == "temporal":
                visible_names.add("temporal_search")
            if signals.channels or signals.strategy_hint == "channel":
                visible_names.add("channel_search")
                visible_names.add("summarize_channel")

        # Keyword-based routing из datasets/tool_keywords.json
        tool_kw = load_tool_keywords()
        for tool_name, keywords in tool_kw.items():
            if any(kw in query_lower for kw in keywords):
                visible_names.add(tool_name)

        # Conditional gating: убираем tools добавленные keywords если нет нужного signal
        if not (signals and (signals.channels or signals.strategy_hint == "channel")):
            # summarize_channel без channel signal — noise от "дайджест"
            visible_names.discard("summarize_channel")
        # cross_channel_compare без явного compare intent — noise от "обсуждали"
        compare_intents = {"сравни", "compare", "vs", "разных канал", "между канал"}
        if not any(ci in query_lower for ci in compare_intents):
            visible_names.discard("cross_channel_compare")

    # Hard cap at 5: deterministic eviction order (lowest priority first)
    eviction_order = load_policy("eviction_order") or [
        "arxiv_tracker", "entity_tracker", "list_channels",
        "summarize_channel", "search",
    ]
    for tool_name in eviction_order:
        if len(visible_names) <= 5:
            break
        if tool_name in visible_names:
            # Не убираем search если нет specialized альтернатив
            if tool_name == "search":
                specialized = visible_names - {"search", "query_plan"}
                if len(specialized) < 2:
                    continue
            visible_names.discard(tool_name)

    return [t for t in AGENT_TOOLS if t["function"]["name"] in visible_names]


def get_available_tools() -> dict[str, Any]:
    """Полный обзор LLM-visible и системных инструментов агента."""
    tools_info: dict[str, Any] = {}
    for tool in AGENT_TOOLS:
        function = tool.get("function", {})
        name = function.get("name")
        if not name:
            continue
        tools_info[name] = {
            "description": function.get("description", ""),
            "parameters_schema": function.get("parameters", {}),
        }

    system_tools = {
        "fetch_docs": "Системная догрузка полных текстов по id",
        "evidence_support_check": "Системный retrieval-backed support check финального ответа",
        "verify": "Legacy alias for evidence_support_check",
    }

    return {
        "tools": tools_info,
        "system_tools": system_tools,
        "total": len(tools_info),
        "max_visible_per_step": 5,
        "note": "AGENT_TOOLS содержит полный набор из 15 LLM-visible schema; get_step_tools показывает не более 5 на шаг.",
    }
