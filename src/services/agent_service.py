"""
ReAct Agent Service на native function calling через /v1/chat/completions.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from core.observability import get_langfuse, observe_trace
from core.security import sanitize_for_logging, security_manager
from core.settings import Settings
from schemas.agent import (
    AgentRequest,
    AgentStepEvent,
)
from services.query_signals import extract_query_signals
from services.tools.tool_runner import ToolRunner

logger = logging.getLogger(__name__)

from services.agent.coverage import compute_nugget_coverage
from services.agent.executor import execute_action
from services.agent.finalization import (
    build_final_payload,
    trim_refusal_alternatives,
    verify_answer,
)
from services.agent.formatting import (
    assistant_message_for_history,
    format_observation,
    tool_error_action,
    tool_message_for_history,
)
from services.agent.guards import (
    build_analytics_repeat_payload,
    check_analytics_shortcircuit,
    check_forced_search,
    should_block_repeat,
)
from services.agent.llm_step import call_llm_step
from services.agent.prompts import SYSTEM_PROMPT
from services.agent.refinement import run_refinement
from services.agent.state import (
    RequestContext,
    _request_ctx,
    apply_action_state,
)
from services.agent.visibility import get_available_tools, get_step_tools


class AgentService:
    """Агент с native function calling и SSE-наблюдаемостью."""

    def __init__(
        self,
        llm_factory: Callable,
        tool_runner: ToolRunner,
        settings: Settings,
    ) -> None:
        self.llm_factory = llm_factory
        self.tool_runner = tool_runner
        self.settings = settings
        self.system_prompt = SYSTEM_PROMPT

    def get_available_tools(self) -> dict[str, Any]:
        """Обёртка для API endpoint /v1/agent/tools."""
        return get_available_tools()

    @property
    def _ctx(self) -> RequestContext:
        """Текущий request context из ContextVar. Raises если вне запроса."""
        ctx = _request_ctx.get()
        if ctx is None:
            raise RuntimeError("No active RequestContext — called outside request scope")
        return ctx

    async def stream_agent_response(
        self, request: AgentRequest
    ) -> AsyncIterator[AgentStepEvent]:
        """Основной цикл агента на chat/completions + tool_calls."""
        request_id = str(uuid.uuid4())
        requested_steps = request.max_steps or self.settings.agent_default_steps
        max_steps = min(max(requested_steps, 1), self.settings.agent_max_steps)
        step = 1
        conversation_history = [f"Human: {request.query}"]

        # FIX-01: per-request state через ContextVar
        ctx = RequestContext(
            request_id=request_id,
            query=request.query,
            original_query=request.query,
            query_signals=extract_query_signals(request.query),
        )
        _ctx_token = _request_ctx.set(ctx)

        # Langfuse root trace — explicit enter/exit для SSE async safety
        _langfuse, _root_trace_cm, _root_span = get_langfuse(), None, None
        if _langfuse:
            try:
                _root_trace_cm = observe_trace(
                    name=request.trace_name or "agent_request",
                    session_id=request.session_id, tags=request.tags,
                    input_data={"query": request.query},
                    metadata={"request_id": request_id, "max_steps": max_steps},
                )
                _root_span = _root_trace_cm.__enter__()
                if _root_span:
                    _root_span.update(input={"query": request.query, "max_steps": max_steps})
            except Exception as e:  # broad: observability graceful degradation
                logger.warning("Langfuse root trace init failed: %s", e)
                _root_trace_cm = _root_span = None

        # System prompt с hints из query signals
        system_content = self.system_prompt
        signals = self._ctx.query_signals
        if signals and (signals.strategy_hint or signals.date_from or signals.channels):
            hint_fields = [
                ("strategy_hint", signals.strategy_hint),
                ("date_from", signals.date_from),
                ("date_to", signals.date_to),
                ("channels", signals.channels),
                ("entities", signals.entities),
            ]
            hints = ", ".join(f"{k}={v}" for k, v in hint_fields if v)
            system_content += (
                f"\nSystem detected hints: {hints}. "
                "Use these hints to guide your tool selection. "
                "You may override if incorrect."
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": request.query},
        ]

        try:
            is_valid, violations = security_manager.validate_input(request.query, context="prompt")
            if not is_valid:
                logger.warning("Security violations in %s: %s", request_id, violations)
                yield AgentStepEvent(type="final", data={
                    "answer": "Извините, ваш запрос содержит недопустимые элементы.",
                    "step": 1, "request_id": request_id, "error": "security_violation",
                })
                return

            logger.info(
                "Agent loop: %s (ID: %s)",
                sanitize_for_logging(request.query, max_length=100), request_id,
            )
            _request_deadline = time.monotonic() + (
                getattr(self.settings, "agent_request_timeout", None) or 90
            )
            ctx.deadline = _request_deadline
            agent_state = ctx.agent_state
            _tool_call_counts: dict[str, int] = {}
            _last_analytics_obs: str | None = None

            while step <= max_steps:
                # FIX-08: deadline check
                if time.monotonic() > _request_deadline:
                    logger.warning("Request %s exceeded deadline, aborting", request_id)
                    yield AgentStepEvent(
                        type="final",
                        data={
                            "answer": "Превышено время обработки запроса.",
                            "step": step,
                            "request_id": request_id,
                            "error": "request_timeout",
                        },
                    )
                    return

                self._ctx.step = step
                step_tools = get_step_tools(self._ctx.agent_state, self._ctx)
                visible_tool_names = [
                    t["function"]["name"] for t in step_tools if "function" in t
                ]

                yield AgentStepEvent(
                    type="step_started",
                    data={
                        "step": step,
                        "request_id": request_id,
                        "max_steps": max_steps,
                        "query": request.query,
                        "visible_tools": visible_tool_names,
                    },
                )

                # ── LLM call (extracted to llm_step.py) ──
                llm = self.llm_factory()
                result = call_llm_step(
                    llm, messages, step_tools, visible_tool_names,
                    step, self.settings, agent_state,
                )
                content = result.content
                tool_calls = result.tool_calls
                assistant_message = result.assistant_message

                if content:
                    yield AgentStepEvent(
                        type="thought",
                        data={"content": content, "step": step, "request_id": request_id},
                    )
                    conversation_history.append(f"Thought: {content}")

                # ── Guards (extracted to guards.py) ──
                tool_calls, forced_msg = check_forced_search(
                    tool_calls, agent_state, content, request.query, step,
                )
                if forced_msg is not None:
                    assistant_message = forced_msg

                analytics_payload = check_analytics_shortcircuit(
                    tool_calls, agent_state, content, _last_analytics_obs,
                    request_id, step, self._ctx,
                )
                if analytics_payload is not None:
                    yield AgentStepEvent(type="final", data=analytics_payload)
                    return

                # ── Tool execution loop ──
                if tool_calls:
                    messages.append(assistant_message_for_history(assistant_message))

                    repeat_same_step = False
                    for tool_call in tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["arguments"]

                        # Tool repeat guard
                        if should_block_repeat(tool_name, _tool_call_counts):
                            logger.warning(
                                "Tool repeat blocked: %s called %d times",
                                tool_name, _tool_call_counts[tool_name],
                            )
                            repeat_payload = build_analytics_repeat_payload(
                                _last_analytics_obs, agent_state,
                                request_id, step, self._ctx,
                            )
                            if repeat_payload is not None:
                                yield AgentStepEvent(type="final", data=repeat_payload)
                                return
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", ""),
                                "content": f"Tool {tool_name} already called. Use final_answer to respond.",
                            })
                            continue

                        yield AgentStepEvent(
                            type="tool_invoked",
                            data={
                                "tool": tool_name,
                                "input": tool_args,
                                "step": step,
                                "request_id": request_id,
                            },
                        )

                        action = await execute_action(
                            tool_name=tool_name,
                            params=tool_args,
                            request_id=request_id,
                            step=step,
                            ctx=self._ctx,
                            tool_runner=self.tool_runner,
                            settings=self.settings,
                        )
                        if action is None:
                            action = tool_error_action(
                                tool_name=tool_name,
                                params=tool_args,
                                step=step,
                                error="tool_execution_failed",
                            )

                        apply_action_state(self._ctx, action)
                        observation_text = format_observation(action.output, tool_name)

                        if tool_name in ("entity_tracker", "arxiv_tracker",
                                         "hot_topics", "channel_expertise"):
                            _last_analytics_obs = observation_text

                        yield AgentStepEvent(
                            type="observation",
                            data={
                                "content": observation_text,
                                "success": action.output.ok,
                                "step": step,
                                "request_id": request_id,
                                "took_ms": action.output.meta.took_ms,
                            },
                        )

                        conversation_history.append(
                            f"Action: {tool_name} {json.dumps(action.input, ensure_ascii=False)}"
                        )
                        conversation_history.append(f"Observation: {observation_text}")
                        messages.append(
                            tool_message_for_history(
                                tool_call, tool_name,
                                action.output.data if action.output.ok
                                else {"error": action.output.meta.error},
                            )
                        )

                        # ── compose_context → coverage + refinement ──
                        if tool_name == "compose_context" and action.output.ok:
                            plan = self._ctx.plan_summary or {}
                            cov_result = compute_nugget_coverage(
                                query=request.query,
                                docs=self._ctx.search_hits,
                                nuggets=plan.get("normalized_queries"),
                            )
                            coverage = cov_result.score
                            self._ctx.uncovered_nuggets = cov_result.uncovered
                            agent_state.coverage = coverage
                            self._ctx.coverage_score = coverage

                            if coverage is not None:
                                yield AgentStepEvent(
                                    type="citations",
                                    data={
                                        "citations": action.output.data.get("citations", []),
                                        "coverage": coverage,
                                        "step": step,
                                        "request_id": request_id,
                                    },
                                )

                                # Low similarity abort guard
                                max_sim = max(
                                    (float(hit.get("dense_score") or 0.0)
                                     for hit in self._ctx.search_hits if isinstance(hit, dict)),
                                    default=0.0,
                                )
                                if max_sim < 0.30 and agent_state.refinement_count == 0:
                                    agent_state.coverage = 0.0
                                    self._ctx.coverage_score = 0.0
                                    agent_state.low_coverage_disclaimer = True
                                    abort_thought = (
                                        "Релевантные документы почти не найдены. "
                                        "Дам осторожный ответ только по найденному контексту."
                                    )
                                    yield AgentStepEvent(
                                        type="thought",
                                        data={
                                            "content": abort_thought,
                                            "step": step,
                                            "request_id": request_id,
                                            "system_generated": True,
                                            "abort_guard": True,
                                        },
                                    )
                                    conversation_history.append(f"Thought: {abort_thought}")
                                    continue

                                if (coverage < self.settings.coverage_threshold
                                        and agent_state.refinement_count < self.settings.max_refinements):
                                    events, should_repeat = await run_refinement(
                                        request.query, agent_state, request_id, step,
                                        messages, conversation_history,
                                        ctx=self._ctx, tool_runner=self.tool_runner,
                                        settings=self.settings, label="refinement",
                                    )
                                    for evt in events:
                                        yield evt
                                    if should_repeat:
                                        repeat_same_step = True
                                        break

                                if (coverage < 0.50
                                        and agent_state.refinement_count >= self.settings.max_refinements):
                                    agent_state.low_coverage_disclaimer = True

                        # ── final_answer → verify + finalize ──
                        if tool_name == "final_answer":
                            answer = str(action.output.data.get("answer", "")).strip()
                            verify_res: dict[str, Any] = {}

                            _skip_verify = (
                                agent_state.analytics_done and agent_state.search_count == 0
                            )
                            if self.settings.enable_verify_step and answer and not _skip_verify:
                                verify_res = await verify_answer(
                                    answer, conversation_history,
                                    ctx=self._ctx, tool_runner=self.tool_runner,
                                )
                                if (not verify_res.get("verified", False)
                                        and agent_state.refinement_count < agent_state.max_refinements):
                                    events, should_repeat = await run_refinement(
                                        request.query, agent_state, request_id, step,
                                        messages, conversation_history,
                                        ctx=self._ctx, tool_runner=self.tool_runner,
                                        settings=self.settings,
                                        label="verification_refinement",
                                    )
                                    for evt in events:
                                        yield evt
                                    if should_repeat:
                                        repeat_same_step = True
                                        break

                                if not verify_res.get("verified", False):
                                    answer += " (⚠️ Ответ не подтверждён с высокой уверенностью)"

                            answer = trim_refusal_alternatives(answer)
                            final_payload = build_final_payload(
                                base_payload=action.output.data,
                                answer=answer,
                                verify_res=verify_res,
                                agent_state=agent_state,
                                request_id=request_id,
                                step=step,
                                ctx=self._ctx,
                            )
                            yield AgentStepEvent(type="final", data=final_payload)
                            return

                    if repeat_same_step:
                        continue

                    step += 1
                    continue

                # No tool_calls — direct answer or fallback
                if content and result.finish_reason == "stop":
                    verify_res_direct: dict[str, Any] = {}
                    direct_answer = content
                    if self.settings.enable_verify_step and direct_answer:
                        verify_res_direct = await verify_answer(
                            direct_answer, conversation_history,
                            ctx=self._ctx, tool_runner=self.tool_runner,
                        )
                        if not verify_res_direct.get("verified", False):
                            direct_answer += " (⚠️ Ответ не подтверждён с высокой уверенностью)"
                    final_payload = build_final_payload(
                        base_payload={"answer": content},
                        answer=direct_answer,
                        verify_res=verify_res_direct,
                        agent_state=agent_state,
                        request_id=request_id,
                        step=step,
                        ctx=self._ctx,
                    )
                    yield AgentStepEvent(type="final", data=final_payload)
                    return

                if content:
                    messages.append({"role": "assistant", "content": content})
                else:
                    fallback_thought = (
                        "Модель не вызвала инструмент и не дала осмысленный ответ. Пробую следующий шаг."
                    )
                    yield AgentStepEvent(
                        type="thought",
                        data={
                            "content": fallback_thought,
                            "step": step,
                            "system_generated": True,
                            "request_id": request_id,
                        },
                    )
                    conversation_history.append(f"Thought: {fallback_thought}")
                    messages.append({"role": "assistant", "content": fallback_thought})

                step += 1

            # max_steps exceeded
            logger.warning(
                "Достигнут максимум шагов (%d) без финального ответа для запроса %s",
                max_steps, request_id,
            )
            if self._ctx.final_answer_text:
                yield AgentStepEvent(
                    type="final",
                    data={
                        "answer": self._ctx.final_answer_text,
                        "step": step,
                        "total_steps": max_steps,
                        "request_id": request_id,
                        "fallback": True,
                    },
                )
                return

            yield AgentStepEvent(
                type="final",
                data={
                    "answer": f"Извините, не удалось завершить анализ за {max_steps} шагов. Попробуйте переформулировать вопрос.",
                    "step": step,
                    "total_steps": max_steps,
                    "request_id": request_id,
                    "error": "max_steps_exceeded",
                },
            )

        except Exception as exc:  # broad: agent loop safety
            logger.error("Ошибка в function-calling agent loop: %s", exc, exc_info=True)
            yield AgentStepEvent(
                type="final",
                data={
                    "answer": f"Извините, произошла ошибка при обработке запроса: {exc!s}",
                    "step": step,
                    "request_id": request_id,
                    "error": str(exc),
                },
            )
        finally:
            # Langfuse cleanup — каждый шаг изолирован
            if _root_span is not None:
                try:
                    _root_span.update(output={
                        "steps": step, "coverage": ctx.coverage_score,
                        "search_count": ctx.agent_state.search_count,
                        "refinement_count": ctx.agent_state.refinement_count,
                        "analytics_done": ctx.agent_state.analytics_done,
                        "search_route": ctx.search_route,
                        "strategy": ctx.agent_state.strategy,
                        "plan": ctx.plan_summary,
                        "citations_count": len(ctx.compose_citations),
                        "prompt_tokens": ctx.total_prompt_tokens,
                        "completion_tokens": ctx.total_completion_tokens,
                        "total_tokens": ctx.total_prompt_tokens + ctx.total_completion_tokens,
                        "answer": (ctx.final_answer_text or "")[:500],
                    })
                except Exception:  # broad: observability graceful degradation
                    pass
            if _root_trace_cm is not None:
                try:
                    _root_trace_cm.__exit__(None, None, None)
                except Exception:  # broad: observability graceful degradation
                    pass
            if _langfuse:
                try:
                    _langfuse.flush()
                except Exception:  # broad: observability graceful degradation
                    pass

            try:
                _request_ctx.reset(_ctx_token)
            except ValueError:
                if _request_ctx.get() is ctx:
                    _request_ctx.set(None)
                logger.debug(
                    "Skip ContextVar reset for %s: different async context", request_id,
                )
