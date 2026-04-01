"""
ReAct Agent Service на native function calling через /v1/chat/completions.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from core.observability import get_langfuse, observe_trace, observe_span
from core.security import sanitize_for_logging, security_manager
from core.settings import Settings
from schemas.agent import (
    AgentAction,
    AgentRequest,
    AgentStepEvent,
    ToolMeta,
    ToolRequest,
    ToolResponse,
)
from services.qa_service import QAService
from services.query_signals import extract_query_signals
from services.tools.tool_runner import ToolRunner

logger = logging.getLogger(__name__)

# SPEC-RAG-20c Step 2: routing вынесен в services/agent/routing.py
from services.agent.routing import load_tool_keywords as _load_tool_keywords
from services.agent.routing import load_policy as _load_policy


from services.agent.prompts import SYSTEM_PROMPT, AGENT_TOOLS  # SPEC-RAG-20c Step 1





# SPEC-RAG-20c Step 3: state вынесен в services/agent/state.py
from services.agent.state import (
    AgentState, RequestContext, _request_ctx, apply_action_state,
)
from services.agent.visibility import get_step_tools, get_available_tools  # SPEC-RAG-20c Step 4
from services.agent.executor import execute_action  # SPEC-RAG-20c Step 6
from services.agent.finalization import (  # SPEC-RAG-20c Step 7
    trim_refusal_alternatives, verify_answer, build_final_payload,
)
from services.agent.coverage import compute_nugget_coverage  # LANCER-style coverage
from services.agent.formatting import (  # SPEC-RAG-20c Step 5
    format_observation, extract_tool_calls, assistant_message_for_history,
    tool_message_for_history, trim_messages, tool_error_action,
)


class AgentService:
    """Агент с native function calling и SSE-наблюдаемостью."""

    def __init__(
        self,
        llm_factory: Callable,
        tool_runner: ToolRunner,
        settings: Settings,
        qa_service: Optional[QAService] = None,
    ) -> None:
        self.llm_factory = llm_factory
        self.tool_runner = tool_runner
        self.settings = settings
        self.qa_service = qa_service
        self.system_prompt = SYSTEM_PROMPT

    def get_available_tools(self) -> Dict[str, Any]:
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

        # FIX-01: per-request state через ContextVar, не self
        ctx = RequestContext(
            request_id=request_id,
            query=request.query,
            original_query=request.query,
            query_signals=extract_query_signals(request.query),
        )
        _ctx_token = _request_ctx.set(ctx)

        # Langfuse root trace — explicit enter/exit (SSE async safety)
        # observe_trace создаёт OTel span с as_root=True + trace-level атрибуты
        _langfuse = get_langfuse()
        _root_trace_cm = None
        _root_span = None
        if _langfuse:
            try:
                _root_trace_cm = observe_trace(
                    name=request.trace_name or "agent_request",
                    session_id=request.session_id,
                    tags=request.tags,
                    input_data={"query": request.query},
                    metadata={"request_id": request_id, "max_steps": max_steps},
                )
                _root_span = _root_trace_cm.__enter__()
                # SPEC-RAG-20b: query в trace input (set_attribute не маппится на trace.input)
                if _root_span:
                    _root_span.update(input={"query": request.query, "max_steps": max_steps})
            except Exception as e:
                logger.warning("Langfuse root trace init failed: %s", e)
                _root_trace_cm = None
                _root_span = None

        # Формируем system prompt с hints из query signals (R13-quick §1.3)
        system_content = self.system_prompt
        signals = self._ctx.query_signals
        if signals and (signals.strategy_hint or signals.date_from or signals.channels):
            hints_parts = []
            if signals.strategy_hint:
                hints_parts.append(f"strategy_hint={signals.strategy_hint}")
            if signals.date_from:
                hints_parts.append(f"date_from={signals.date_from}")
            if signals.date_to:
                hints_parts.append(f"date_to={signals.date_to}")
            if signals.channels:
                hints_parts.append(f"channels={signals.channels}")
            if signals.entities:
                hints_parts.append(f"entities={signals.entities}")
            system_content += (
                f"\nSystem detected hints: {', '.join(hints_parts)}. "
                "Use these hints to guide your tool selection. "
                "You may override if incorrect."
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": request.query},
        ]

        try:
            is_valid, violations = security_manager.validate_input(
                request.query, context="prompt"
            )
            if not is_valid:
                logger.warning(
                    "Security violations in request %s: %s", request_id, violations
                )
                yield AgentStepEvent(
                    type="final",
                    data={
                        "answer": "Извините, ваш запрос содержит недопустимые элементы. Пожалуйста, переформулируйте вопрос.",
                        "step": 1,
                        "request_id": request_id,
                        "error": "security_violation",
                    },
                )
                return

            sanitized_query = sanitize_for_logging(request.query, max_length=100)
            logger.info(
                "Начинаем function-calling agent loop для запроса: %s (ID: %s)",
                sanitized_query,
                request_id,
            )

            # FIX-08: cooperative deadline — проверяется между шагами
            _request_deadline = time.monotonic() + (
                getattr(self.settings, "agent_request_timeout", None) or 90
            )
            ctx.deadline = _request_deadline

            # FIX-01: локальный alias для backward compat с кодом внутри loop
            agent_state = ctx.agent_state

            # Tool repeat guard — блокируем зацикливание на одном tool
            _tool_call_counts: Dict[str, int] = {}
            _NO_REPEAT_TOOLS = {"entity_tracker", "arxiv_tracker", "hot_topics",
                                "channel_expertise", "list_channels", "related_posts"}
            _MAX_REPEAT = 2  # search/rerank можно 2 раза (refinement), остальные — 1
            _last_analytics_obs: Optional[str] = None  # последний observation от analytics tool

            while step <= max_steps:
                # FIX-08: deadline check перед каждым шагом
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

                # Динамический набор tools: final_answer доступен
                # только после search. Без этого LLM иногда пропускает
                # поиск и сразу отвечает "не найдено" (особенно Qwen3-30B
                # с 3B active params — ненадёжно следует сложным промптам).
                step_tools = get_step_tools(self._ctx.agent_state, self._ctx)
                visible_tool_names = [
                    t["function"]["name"]
                    for t in step_tools
                    if "function" in t
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

                llm = self.llm_factory()
                trimmed_messages = trim_messages(messages)
                # После compose_context И после analytics/navigation short-circuit
                # LLM может сразу генерировать финальный ответ текстом.
                # Для таких шагов нужен больший budget, иначе получаем
                # finish_reason=length и зацикливание с assistant messages подряд.
                expect_final = (
                    agent_state.coverage > 0
                    or agent_state.analytics_done
                    or agent_state.navigation_answered
                )
                step_max_tokens = (
                    self.settings.agent_final_max_tokens
                    if expect_final
                    else self.settings.agent_tool_max_tokens
                )

                try:
                    _phase = "final" if expect_final else ("post_search" if agent_state.search_count > 0 else "pre_search")
                    llm_span_name = f"llm_step_{step}_{_phase}"
                    with observe_span(llm_span_name, metadata={
                        "step": step, "phase": _phase,
                        "search_count": agent_state.search_count,
                        "refinement_count": agent_state.refinement_count,
                        "coverage": agent_state.coverage,
                        "visible_tools": [t["function"]["name"] for t in step_tools] if step_tools else [],
                    }) as _llm_span:
                        response = llm.chat_completion(
                            messages=trimmed_messages,
                            tools=step_tools,
                            max_tokens=step_max_tokens,
                            temperature=self.settings.agent_tool_temp,
                            top_p=self.settings.agent_tool_top_p,
                            top_k=self.settings.agent_tool_top_k,
                            presence_penalty=self.settings.agent_tool_presence_penalty,
                            seed=42,
                        )
                except Exception as llm_exc:
                    # LLM 400/500 — попробуем с агрессивно обрезанными messages
                    logger.warning(
                        "LLM call failed at step %d: %s — retrying with trimmed history",
                        step, llm_exc,
                    )
                    # Оставляем только system + user + последние 2 messages
                    fallback_messages = messages[:2] + messages[-2:]
                    try:
                        response = llm.chat_completion(
                            messages=fallback_messages,
                            tools=step_tools,
                            max_tokens=step_max_tokens,
                            temperature=self.settings.agent_tool_temp,
                            seed=42,
                        )
                    except Exception:
                        raise llm_exc  # original error если retry тоже failed

                choice = (response.get("choices") or [{}])[0]
                assistant_message = choice.get("message") or {}
                finish_reason = choice.get("finish_reason", "unknown")
                content = (assistant_message.get("content") or "").strip()
                tool_calls = extract_tool_calls(
                    assistant_message, visible_tools=set(visible_tool_names)
                )

                logger.debug(
                    "Agent step %d finish=%s content_len=%d tool_calls=%d",
                    step,
                    finish_reason,
                    len(content),
                    len(tool_calls),
                )

                # SPEC-RAG-20b: llm_step span output
                if _llm_span:
                    _llm_span.update(
                        input={"message_count": len(trimmed_messages)},
                        output={
                            "finish_reason": finish_reason,
                            "content_len": len(content),
                            "tool_calls": [tc["name"] for tc in tool_calls],
                        },
                    )

                if content:
                    yield AgentStepEvent(
                        type="thought",
                        data={
                            "content": content,
                            "step": step,
                            "request_id": request_id,
                        },
                    )
                    conversation_history.append(f"Thought: {content}")

                # Если LLM не вызвала tools и search ещё не был —
                # принудительно вызываем search с оригинальным запросом.
                # Qwen3-30B иногда решает ответить "не найдено" без поиска.
                # НЕ форсим если:
                # - list_channels уже дал ответ (navigation intent)
                # - analytics_done (entity_tracker/arxiv_tracker ответили)
                # - LLM явно отказывается И запрос negative-intent
                #
                # SPEC-RAG-15 fix (q01/q03): refusal bypass применяется ТОЛЬКО
                # для явно negative запросов ("существует ли", "выходила ли", out-of-range).
                # Для обычных factual запросов — всегда форсим search,
                # даже если LLM сгенерировал refusal markers.
                _refusal_markers = _load_policy("refusal_markers")
                _negative_intent_markers = _load_policy("negative_intent_markers")
                is_refusal = content and any(m in content.lower() for m in _refusal_markers)
                is_negative_intent = any(m in request.query.lower() for m in _negative_intent_markers)
                # Bypass forced search только для negative intent refusals
                skip_forced = is_refusal and is_negative_intent
                if not tool_calls and agent_state.search_count == 0 and not agent_state.navigation_answered and not agent_state.analytics_done and not skip_forced:
                    logger.warning(
                        "Agent step %d: LLM не вызвала tools, search_count=0 → forced search",
                        step,
                    )
                    tool_calls = [{
                        "id": f"forced_search_{step}",
                        "name": "search",
                        "arguments": {"queries": [request.query]},
                    }]
                    # Формируем assistant message для истории
                    assistant_message = {
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": [{
                            "id": tool_calls[0]["id"],
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps(tool_calls[0]["arguments"]),
                            },
                        }],
                    }

                # Safety net: analytics done но модель не вызвала final_answer
                # (tool calls отфильтрованы или модель сгенерировала текст/ничего)
                if not tool_calls and agent_state.analytics_done:
                    direct_answer = content or _last_analytics_obs or ""
                    if direct_answer:
                        logger.info(
                            "Analytics forced completion at step %d (content=%d, obs=%s)",
                            step, len(content), bool(_last_analytics_obs),
                        )
                        final_payload = build_final_payload(
                            base_payload={"answer": direct_answer},
                            answer=direct_answer,
                            verify_res={},
                            agent_state=agent_state,
                            request_id=request_id,
                            step=step,
                            ctx=self._ctx,
                        )
                        yield AgentStepEvent(type="final", data=final_payload)
                        return

                if tool_calls:
                    messages.append(assistant_message_for_history(assistant_message))

                    repeat_same_step = False
                    for tool_call in tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["arguments"]

                        # Tool repeat guard — блокируем зацикливание, но
                        # добавляем tool response чтобы не ломать message sequence
                        _tool_call_counts[tool_name] = _tool_call_counts.get(tool_name, 0) + 1
                        max_allowed = 1 if tool_name in _NO_REPEAT_TOOLS else _MAX_REPEAT
                        if _tool_call_counts[tool_name] > max_allowed:
                            logger.warning(
                                "Tool repeat blocked: %s called %d times (max %d)",
                                tool_name, _tool_call_counts[tool_name], max_allowed,
                            )
                            # Если analytics уже done и модель зациклилась — force finish
                            if agent_state.analytics_done and _last_analytics_obs:
                                logger.info(
                                    "Analytics guard forced completion: %s repeat after analytics_done",
                                    tool_name,
                                )
                                final_payload = build_final_payload(
                                    base_payload={"answer": _last_analytics_obs},
                                    answer=_last_analytics_obs,
                                    verify_res={},
                                    agent_state=agent_state,
                                    request_id=request_id,
                                    step=step,
                                    ctx=self._ctx,
                                )
                                yield AgentStepEvent(type="final", data=final_payload)
                                return
                            # Добавляем tool response в history чтобы избежать
                            # "2 assistant messages at end" error от llama-server
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
                        observation_text = format_observation(
                            action.output, tool_name
                        )

                        # Сохраняем analytics observation для safety net
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
                        conversation_history.append(
                            f"Observation: {observation_text}"
                        )
                        messages.append(
                            tool_message_for_history(
                                tool_call,
                                tool_name,
                                action.output.data
                                if action.output.ok
                                else {"error": action.output.meta.error},
                            )
                        )

                        if tool_name == "compose_context" and action.output.ok:
                            # LANCER-style nugget coverage вместо cosine-based
                            plan = self._ctx.plan_summary or {}
                            nuggets = plan.get("normalized_queries")
                            cov_result = compute_nugget_coverage(
                                query=request.query,
                                docs=self._ctx.search_hits,
                                nuggets=nuggets,
                            )
                            coverage = cov_result.score
                            self._ctx.uncovered_nuggets = cov_result.uncovered
                            agent_state.coverage = coverage
                            self._ctx.coverage_score = coverage

                            yield AgentStepEvent(
                                type="citations",
                                data={
                                    "citations": action.output.data.get(
                                        "citations", []
                                    ),
                                    "coverage": coverage,
                                    "step": step,
                                    "request_id": request_id,
                                },
                            )

                            max_sim = max(
                                (
                                    float(hit.get("dense_score") or 0.0)
                                    for hit in self._ctx.search_hits
                                    if isinstance(hit, dict)
                                ),
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
                                conversation_history.append(
                                    f"Thought: {abort_thought}"
                                )
                                continue

                            if self._should_attempt_refinement(
                                coverage, agent_state.refinement_count
                            ):
                                agent_state.refinement_count += 1
                                refine_thought = (
                                    "Покрытие контекста недостаточно. Выполняю дополнительный поиск."
                                )
                                yield AgentStepEvent(
                                    type="thought",
                                    data={
                                        "content": refine_thought,
                                        "step": step,
                                        "request_id": request_id,
                                        "system_generated": True,
                                        "refinement": True,
                                        "refinement_count": agent_state.refinement_count,
                                    },
                                )
                                conversation_history.append(
                                    f"Thought: {refine_thought}"
                                )

                                refinement_actions = await self._perform_refinement(
                                    request.query,
                                    agent_state,
                                    request_id,
                                    step,
                                )

                                for refinement_action in refinement_actions:
                                    yield AgentStepEvent(
                                        type="tool_invoked",
                                        data={
                                            "tool": refinement_action.tool,
                                            "input": refinement_action.input,
                                            "step": step,
                                            "request_id": request_id,
                                            "system_generated": True,
                                            "refinement": True,
                                        },
                                    )

                                    apply_action_state(self._ctx, refinement_action)
                                    refinement_observation = format_observation(
                                        refinement_action.output,
                                        refinement_action.tool,
                                    )
                                    yield AgentStepEvent(
                                        type="observation",
                                        data={
                                            "content": refinement_observation,
                                            "success": refinement_action.output.ok,
                                            "step": step,
                                            "request_id": request_id,
                                            "took_ms": refinement_action.output.meta.took_ms,
                                            "system_generated": True,
                                            "refinement": True,
                                        },
                                    )
                                    conversation_history.append(
                                        f"Action: {refinement_action.tool} {json.dumps(refinement_action.input, ensure_ascii=False)}"
                                    )
                                    conversation_history.append(
                                        f"Observation: {refinement_observation}"
                                    )
                                    messages.append(
                                        tool_message_for_history(
                                            {
                                                "id": f"refinement-{step}-{refinement_action.tool}",
                                                "name": refinement_action.tool,
                                                "arguments": refinement_action.input,
                                            },
                                            refinement_action.tool,
                                            refinement_action.output.data
                                            if refinement_action.output.ok
                                            else {
                                                "error": refinement_action.output.meta.error
                                            },
                                        )
                                    )

                                    if (
                                        refinement_action.tool == "compose_context"
                                        and refinement_action.output.ok
                                    ):
                                        refined_coverage = float(
                                            refinement_action.output.data.get(
                                                "citation_coverage", 0.0
                                            )
                                            or 0.0
                                        )
                                        agent_state.coverage = refined_coverage
                                        self._ctx.coverage_score = refined_coverage
                                        yield AgentStepEvent(
                                            type="citations",
                                            data={
                                                "citations": refinement_action.output.data.get(
                                                    "citations", []
                                                ),
                                                "coverage": refined_coverage,
                                                "step": step,
                                                "request_id": request_id,
                                                "system_generated": True,
                                                "refinement": True,
                                            },
                                        )
                                        if (
                                            refined_coverage < 0.50
                                            and agent_state.refinement_count
                                            >= self.settings.max_refinements
                                        ):
                                            agent_state.low_coverage_disclaimer = True

                                repeat_same_step = True
                                break

                            if (
                                coverage < 0.50
                                and agent_state.refinement_count
                                >= self.settings.max_refinements
                            ):
                                agent_state.low_coverage_disclaimer = True

                        if tool_name == "final_answer":
                            answer = str(action.output.data.get("answer", "")).strip()
                            verify_res: Dict[str, Any] = {}

                            # SPEC-RAG-15: skip verify для analytics-only ответов
                            # (агрегации без документов — verify по документам бессмыслен)
                            _skip_verify = (
                                agent_state.analytics_done
                                and agent_state.search_count == 0
                            )
                            if self.settings.enable_verify_step and answer and not _skip_verify:
                                verify_res = await verify_answer(
                                    answer, conversation_history,
                                    ctx=self._ctx, tool_runner=self.tool_runner,
                                )
                                if (
                                    not verify_res.get("verified", False)
                                    and agent_state.refinement_count
                                    < agent_state.max_refinements
                                ):
                                    agent_state.refinement_count += 1
                                    verify_thought = (
                                        "Ответ требует дополнительной проверки. Выполняю ещё один поиск перед финализацией."
                                    )
                                    yield AgentStepEvent(
                                        type="thought",
                                        data={
                                            "content": verify_thought,
                                            "step": step,
                                            "request_id": request_id,
                                            "system_generated": True,
                                            "verification": True,
                                            "refinement_count": agent_state.refinement_count,
                                        },
                                    )
                                    conversation_history.append(
                                        f"Thought: {verify_thought}"
                                    )

                                    refinement_actions = await self._perform_refinement(
                                        request.query,
                                        agent_state,
                                        request_id,
                                        step,
                                    )
                                    for refinement_action in refinement_actions:
                                        yield AgentStepEvent(
                                            type="tool_invoked",
                                            data={
                                                "tool": refinement_action.tool,
                                                "input": refinement_action.input,
                                                "step": step,
                                                "request_id": request_id,
                                                "system_generated": True,
                                                "verification_refinement": True,
                                            },
                                        )
                                        apply_action_state(self._ctx, refinement_action)
                                        refinement_observation = format_observation(
                                            refinement_action.output,
                                            refinement_action.tool,
                                        )
                                        yield AgentStepEvent(
                                            type="observation",
                                            data={
                                                "content": refinement_observation,
                                                "success": refinement_action.output.ok,
                                                "step": step,
                                                "request_id": request_id,
                                                "took_ms": refinement_action.output.meta.took_ms,
                                                "system_generated": True,
                                                "verification_refinement": True,
                                            },
                                        )
                                        conversation_history.append(
                                            f"Action: {refinement_action.tool} {json.dumps(refinement_action.input, ensure_ascii=False)}"
                                        )
                                        conversation_history.append(
                                            f"Observation: {refinement_observation}"
                                        )
                                        messages.append(
                                            tool_message_for_history(
                                                {
                                                    "id": f"verify-refinement-{step}-{refinement_action.tool}",
                                                    "name": refinement_action.tool,
                                                    "arguments": refinement_action.input,
                                                },
                                                refinement_action.tool,
                                                refinement_action.output.data
                                                if refinement_action.output.ok
                                                else {
                                                    "error": refinement_action.output.meta.error
                                                },
                                            )
                                        )
                                        if (
                                            refinement_action.tool == "compose_context"
                                            and refinement_action.output.ok
                                        ):
                                            refined_coverage = float(
                                                refinement_action.output.data.get(
                                                    "citation_coverage", 0.0
                                                )
                                                or 0.0
                                            )
                                            agent_state.coverage = refined_coverage
                                            self._ctx.coverage_score = refined_coverage
                                            yield AgentStepEvent(
                                                type="citations",
                                                data={
                                                    "citations": refinement_action.output.data.get(
                                                        "citations", []
                                                    ),
                                                    "coverage": refined_coverage,
                                                    "step": step,
                                                    "request_id": request_id,
                                                    "system_generated": True,
                                                    "verification_refinement": True,
                                                },
                                            )
                                    repeat_same_step = True
                                    break

                                if not verify_res.get("verified", False):
                                    answer += (
                                        " (⚠️ Ответ не подтверждён с высокой уверенностью)"
                                    )

                            # SPEC-RAG-15 fix (q19): deterministic refusal trim.
                            # Если ответ содержит refusal ("не найден", "нет в базе")
                            # но потом предлагает альтернативы — обрезать до чистого refusal.
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

                if content and finish_reason == "stop":
                    verify_res: Dict[str, Any] = {}
                    direct_answer = content
                    if self.settings.enable_verify_step and direct_answer:
                        verify_res = await verify_answer(
                            direct_answer, conversation_history,
                            ctx=self._ctx, tool_runner=self.tool_runner,
                        )
                        if not verify_res.get("verified", False):
                            direct_answer += (
                                " (⚠️ Ответ не подтверждён с высокой уверенностью)"
                            )
                    final_payload = build_final_payload(
                        base_payload={"answer": content},
                        answer=direct_answer,
                        verify_res=verify_res,
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

            logger.warning(
                "Достигнут максимум шагов (%d) без финального ответа для запроса %s",
                max_steps,
                request_id,
            )
            # SPEC-RAG-20d: при max_steps используем уже собранный агентный контекст
            # вместо legacy QAService с max_context_length=2000.
            # Если compose_context уже был вызван, его данные есть в ctx.
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

        except Exception as exc:
            logger.error("Ошибка в function-calling agent loop: %s", exc, exc_info=True)
            yield AgentStepEvent(
                type="final",
                data={
                    "answer": f"Извините, произошла ошибка при обработке запроса: {str(exc)}",
                    "step": step,
                    "request_id": request_id,
                    "error": str(exc),
                },
            )
        finally:
            # Langfuse root span cleanup — до ContextVar reset
            if _root_span is not None:
                try:
                    _root_span.update(output={
                        "steps": step,
                        "coverage": ctx.coverage_score,
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
                except Exception:
                    pass
            if _root_trace_cm is not None:
                try:
                    _root_trace_cm.__exit__(None, None, None)
                except Exception:
                    pass  # Не crash'им на observability cleanup
            if _langfuse:
                try:
                    _langfuse.flush()
                except Exception:
                    pass

            # При закрытии SSE async-generator cleanup может выполниться уже
            # в другом Context (например, через GeneratorExit после финального yield).
            # В таком случае reset(original_token) бросает ValueError.
            # Для request-scope это не критично: исходный context уже уходит,
            # а здесь важно не шуметь ложной ошибкой в логах.
            try:
                _request_ctx.reset(_ctx_token)
            except ValueError:
                current_ctx = _request_ctx.get()
                if current_ctx is ctx:
                    _request_ctx.set(None)
                logger.debug(
                    "Skip ContextVar reset for request %s: cleanup runs in different async context",
                    request_id,
                )

    def _should_attempt_refinement(
        self, coverage: float, refinement_count: int
    ) -> bool:
        """Проверяет, нужен ли refinement по coverage и лимиту попыток."""
        return (
            coverage < self.settings.coverage_threshold
            and refinement_count < self.settings.max_refinements
        )


    async def _perform_refinement(
        self, query: str, agent_state: AgentState, request_id: str, step: int
    ) -> List[AgentAction]:
        """Targeted refinement: ищет конкретно то чего не хватает (SEAL-RAG + LANCER).

        Использует uncovered_nuggets из nugget coverage вместо повторного
        поиска по всему запросу. Если nuggets нет — fallback на plan subqueries.
        """
        # Targeted: ищем по непокрытым nuggets
        plan = self._ctx.plan_summary or {}
        uncovered = self._ctx.uncovered_nuggets
        if uncovered:
            queries = list(uncovered)
            logger.info(
                "Targeted refinement: %d uncovered nuggets: %s",
                len(uncovered), [n[:50] for n in uncovered],
            )
        else:
            # Fallback: plan subqueries или original query
            queries = list(plan.get("normalized_queries") or [query])
        if query not in queries:
            queries.insert(0, query)

        # Сохраняем metadata_filters из planner
        filters = {}
        plan_filters = plan.get("metadata_filters")
        if isinstance(plan_filters, dict):
            filters = {k: v for k, v in plan_filters.items() if v is not None}

        actions: List[AgentAction] = []
        search_action = await execute_action(
            tool_name="search",
            params={
                "queries": queries,
                "filters": filters,
                "k": 20,
                "route": "hybrid",
            },
            request_id=request_id,
            step=step,
            ctx=self._ctx,
            tool_runner=self.tool_runner,
            settings=self.settings,
        )
        if search_action is None:
            return actions

        actions.append(search_action)
        if not search_action.output.ok:
            return actions

        apply_action_state(self._ctx, search_action)
        compose_action = await execute_action(
            tool_name="compose_context",
            params={"hit_ids": []},
            request_id=request_id,
            step=step,
            ctx=self._ctx,
            tool_runner=self.tool_runner,
            settings=self.settings,
        )
        if compose_action is not None:
            actions.append(compose_action)
        return actions



