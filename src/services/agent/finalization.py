"""
SPEC-RAG-20c Step 7: Final answer preparation — build payload, refusal trim, verify.

Ответственность: подготовка и проверка финального ответа перед SSE yield.
"""

from __future__ import annotations

import logging
from typing import Any

from schemas.agent import ToolRequest
from services.agent.routing import load_policy
from services.agent.state import AgentState, RequestContext

logger = logging.getLogger(__name__)


def trim_refusal_alternatives(answer: str) -> str:
    """Обрезает альтернативы после refusal (q19 fix).

    Если ответ содержит refusal pattern ("не найден", "нет в базе")
    и потом предлагает альтернативы ("Однако", "Но в базе есть") —
    обрезаем до чистого refusal. Refusal policy запрещает альтернативы.
    """
    answer_lower = answer.lower()
    _refusal_patterns = load_policy("refusal_markers")
    _alt_patterns = load_policy("refusal_alt_patterns")

    has_refusal = any(p in answer_lower for p in _refusal_patterns)
    if not has_refusal:
        return answer

    for alt in _alt_patterns:
        idx = answer_lower.find(alt)
        if idx > 0:
            trimmed = answer[:idx].rstrip(" \n.,;")
            if len(trimmed) > 20:
                logger.debug("Refusal trim: cut %d chars of alternatives", len(answer) - len(trimmed))
                return trimmed

    return answer


async def verify_answer(
    final_answer: str,
    conversation_history: list[str],
    ctx: RequestContext,
    tool_runner,
) -> dict[str, Any]:
    """Проверяет финальный ответ через verify tool."""
    try:
        original_query = conversation_history[0] if conversation_history else ""
        if not original_query.startswith("Human: "):
            original_query = "Human: " + original_query

        result = tool_runner.run(
            ctx.request_id or "verify",
            ctx.step,
            ToolRequest(
                tool="verify",
                input={
                    "query": original_query,
                    "claim": final_answer,
                    "top_k": 3,
                },
            ),
            deadline=ctx.deadline,
        )

        if result.output.ok:
            return result.output.data

        return {
            "verified": False,
            "confidence": 0.0,
            "error": result.output.meta.error or "Tool execution failed",
        }
    except Exception as exc:
        logger.error("Error in verify_answer: %s", exc, exc_info=True)
        return {"verified": False, "confidence": 0.0, "error": str(exc)}


def build_final_payload(
    base_payload: dict[str, Any],
    answer: str,
    verify_res: dict[str, Any],
    agent_state: AgentState,
    request_id: str,
    step: int,
    ctx: RequestContext,
) -> dict[str, Any]:
    """Собирает финальный SSE payload без изменения публичной схемы."""
    final_answer = answer.strip()

    if agent_state.low_coverage_disclaimer:
        disclaimer = (
            "[Примечание: найдено ограниченное количество релевантной информации. "
            "Ответ может быть неполным.]"
        )
        final_answer = (
            f"{disclaimer}\n\n{final_answer}" if final_answer else disclaimer
        )

    final_payload = dict(base_payload)
    final_payload["answer"] = final_answer
    final_payload.setdefault("citations", ctx.compose_citations)
    final_payload["coverage"] = agent_state.coverage
    final_payload["refinements"] = agent_state.refinement_count
    final_payload["route"] = ctx.search_route
    final_payload["plan"] = ctx.plan_summary
    if verify_res:
        final_payload["verification"] = {
            "verified": verify_res.get("verified", False),
            "confidence": verify_res.get("confidence", 0.0),
            "documents_found": verify_res.get("documents_found", 0),
        }
    final_payload.update(
        {
            "step": step,
            "total_steps": step,
            "request_id": request_id,
        }
    )
    # SPEC-RAG-20b: сохраняем answer для trace output
    ctx.final_answer_text = final_answer
    return final_payload
