"""
ReAct Agent Service на native function calling через /v1/chat/completions.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

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
from services.tools.tool_runner import ToolRunner

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Ты — RAG-агент для поиска информации в базе новостей из Telegram-каналов.

ПОРЯДОК РАБОТЫ:
1. query_plan — декомпозируй запрос на подзапросы
2. search — найди документы по подзапросам
3. rerank — переранжируй документы по исходному запросу
4. compose_context — собери контекст из лучших документов (вызывай без параметров)
5. final_answer — дай итоговый ответ на основе контекста

ПРАВИЛА ОТВЕТА:
- Отвечай ТОЛЬКО на русском языке
- Основывайся на фактах из предоставленного контекста
- Каждое утверждение подкрепляй ссылкой [1], [2] и т.д.
- Если контекст содержит релевантную информацию — используй её и дай развёрнутый ответ
- Если контекст НЕ содержит информации по теме запроса — честно скажи "В источниках не найдено информации по данному вопросу". НЕ приплетай нерелевантные документы ради объёма ответа
- Не выдавай документы за ответ на вопрос, если они не отвечают на него напрямую
- В final_answer ОБЯЗАТЕЛЬНО заполни поле sources номерами использованных источников
- После compose_context переходи к final_answer, не ищи повторно
"""

AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_plan",
            "description": (
                "Декомпозирует сложный запрос на 3-5 подзапросов с фильтрами. "
                "Вызывай первым для планирования поиска."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Исходный запрос пользователя",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Выполняет гибридный поиск dense+sparse с RRF по коллекции "
                "Telegram-каналов и возвращает документы."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список поисковых запросов (1-5 штук)",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Количество результатов",
                        "default": 10,
                    },
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rerank",
            "description": (
                "Переранжирует найденные документы по семантической близости к запросу. "
                "Вызывай после search. Документы подставляются автоматически."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Исходный пользовательский запрос",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Количество лучших документов",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compose_context",
            "description": (
                "Собирает контекст из ВСЕХ найденных документов с цитатами [1], [2] и считает coverage. "
                "Вызывай после rerank. Документы подставляются автоматически. Не передавай параметры."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Формирует финальный ответ пользователю на русском языке, "
                "опираясь только на собранный контекст."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Текст ответа с цитатами [1], [2]",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Номера использованных источников",
                    },
                },
                "required": ["answer", "sources"],
            },
        },
    },
]


class AgentState:
    """Tracks dynamic state of the agent between steps."""

    def __init__(self) -> None:
        self.coverage: float = 0.0
        self.refinement_count: int = 0
        self.max_refinements: int = 2
        self.low_coverage_disclaimer: bool = False
        self.search_count: int = 0


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
        self._current_request_id: Optional[str] = None
        self._current_step: int = 1
        self._current_query: Optional[str] = None
        self._last_search_hits: List[Dict[str, Any]] = []
        self._last_search_route: Optional[str] = None
        self._last_plan_summary: Optional[Dict[str, Any]] = None
        self._last_compose_citations: List[Dict[str, Any]] = []
        self._last_coverage: float = 0.0
        self.system_prompt = SYSTEM_PROMPT

    async def stream_agent_response(
        self, request: AgentRequest
    ) -> AsyncIterator[AgentStepEvent]:
        """Основной цикл агента на chat/completions + tool_calls."""
        request_id = str(uuid.uuid4())
        requested_steps = request.max_steps or self.settings.agent_default_steps
        max_steps = min(max(requested_steps, 1), self.settings.agent_max_steps)
        step = 1
        agent_state = AgentState()
        self._agent_state = agent_state  # для _get_step_tools и _apply_action_state
        conversation_history = [f"Human: {request.query}"]

        self._current_request_id = request_id
        self._current_query = request.query
        self._last_search_hits = []
        self._last_search_route = None
        self._last_plan_summary = None
        self._last_compose_citations = []
        self._last_coverage = 0.0

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
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

            while step <= max_steps:
                self._current_step = step
                yield AgentStepEvent(
                    type="step_started",
                    data={
                        "step": step,
                        "request_id": request_id,
                        "max_steps": max_steps,
                        "query": request.query,
                    },
                )

                llm = self.llm_factory()
                trimmed_messages = self._trim_messages(messages)
                # После compose_context LLM будет генерировать final_answer
                # с длинным текстом — нужен больший бюджет токенов.
                expect_final = agent_state.coverage > 0
                step_max_tokens = (
                    self.settings.agent_final_max_tokens
                    if expect_final
                    else self.settings.agent_tool_max_tokens
                )
                # Динамический набор tools: final_answer доступен
                # только после search. Без этого LLM иногда пропускает
                # поиск и сразу отвечает "не найдено" (особенно Qwen3-30B
                # с 3B active params — ненадёжно следует сложным промптам).
                step_tools = self._get_step_tools(agent_state)

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

                choice = (response.get("choices") or [{}])[0]
                assistant_message = choice.get("message") or {}
                finish_reason = choice.get("finish_reason", "unknown")
                content = (assistant_message.get("content") or "").strip()
                tool_calls = self._extract_tool_calls(assistant_message)

                logger.debug(
                    "Agent step %d finish=%s content_len=%d tool_calls=%d",
                    step,
                    finish_reason,
                    len(content),
                    len(tool_calls),
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
                if not tool_calls and agent_state.search_count == 0:
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

                if tool_calls:
                    messages.append(self._assistant_message_for_history(assistant_message))

                    repeat_same_step = False
                    for tool_call in tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["arguments"]

                        yield AgentStepEvent(
                            type="tool_invoked",
                            data={
                                "tool": tool_name,
                                "input": tool_args,
                                "step": step,
                                "request_id": request_id,
                            },
                        )

                        action = await self._execute_action(
                            tool_name=tool_name,
                            params=tool_args,
                            request_id=request_id,
                            step=step,
                        )
                        if action is None:
                            action = self._tool_error_action(
                                tool_name=tool_name,
                                params=tool_args,
                                step=step,
                                error="tool_execution_failed",
                            )

                        self._apply_action_state(action)
                        observation_text = self._format_observation(
                            action.output, tool_name
                        )
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
                            self._tool_message_for_history(
                                tool_call,
                                tool_name,
                                action.output.data
                                if action.output.ok
                                else {"error": action.output.meta.error},
                            )
                        )

                        if tool_name == "compose_context" and action.output.ok:
                            coverage = float(
                                action.output.data.get("citation_coverage", 0.0) or 0.0
                            )
                            agent_state.coverage = coverage
                            self._last_coverage = coverage

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
                                    for hit in self._last_search_hits
                                    if isinstance(hit, dict)
                                ),
                                default=0.0,
                            )

                            if max_sim < 0.30 and agent_state.refinement_count == 0:
                                agent_state.coverage = 0.0
                                self._last_coverage = 0.0
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

                                    self._apply_action_state(refinement_action)
                                    refinement_observation = self._format_observation(
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
                                        self._tool_message_for_history(
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
                                        self._last_coverage = refined_coverage
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

                            if self.settings.enable_verify_step and answer:
                                verify_res = await self._verify_answer(
                                    answer, conversation_history
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
                                        self._apply_action_state(refinement_action)
                                        refinement_observation = self._format_observation(
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
                                            self._tool_message_for_history(
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
                                            self._last_coverage = refined_coverage
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

                            final_payload = self._build_final_payload(
                                base_payload=action.output.data,
                                answer=answer,
                                verify_res=verify_res,
                                agent_state=agent_state,
                                request_id=request_id,
                                step=step,
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
                        verify_res = await self._verify_answer(
                            direct_answer, conversation_history
                        )
                        if not verify_res.get("verified", False):
                            direct_answer += (
                                " (⚠️ Ответ не подтверждён с высокой уверенностью)"
                            )
                    final_payload = self._build_final_payload(
                        base_payload={"answer": content},
                        answer=direct_answer,
                        verify_res=verify_res,
                        agent_state=agent_state,
                        request_id=request_id,
                        step=step,
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
            if self.qa_service:
                try:
                    fallback_answer = self.qa_service.answer(request.query)
                    yield AgentStepEvent(
                        type="final",
                        data={
                            "answer": f"Не удалось завершить анализ за {max_steps} шагов. Краткий ответ: {fallback_answer}",
                            "step": step,
                            "total_steps": max_steps,
                            "request_id": request_id,
                            "fallback": True,
                        },
                    )
                    return
                except Exception as exc:
                    logger.error("Fallback через QAService не удался: %s", exc)

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
            self._current_request_id = None
            self._current_query = None

    async def _execute_action(
        self,
        tool_name: str,
        params: Dict[str, Any],
        request_id: str,
        step: int,
    ) -> Optional[AgentAction]:
        """Нормализует параметры и выполняет инструмент через ToolRunner."""
        try:
            safe_params = dict(params or {})
            # Нормализуем ДО security check — rerank/compose_context заполняют
            # данные из внутреннего состояния, а не от пользователя.
            normalized = self._normalize_tool_params(tool_name, safe_params)

            # Security check только для user-facing полей, не для внутренних docs.
            # rerank и compose_context получают docs из _last_search_hits (системно).
            _skip_security = {"rerank", "compose_context"}
            if tool_name not in _skip_security:
                serialized = json.dumps(
                    normalized, ensure_ascii=False, default=str
                )
                is_valid, violations = security_manager.validate_input(
                    serialized
                )
                if not is_valid:
                    logger.warning(
                        "Security violations in tool params for %s: %s",
                        tool_name,
                        violations,
                    )
                    return self._tool_error_action(
                        tool_name=tool_name,
                        params=safe_params,
                        step=step,
                        error="security_violation",
                    )
            tool_request = ToolRequest(tool=tool_name, input=normalized)
            return self.tool_runner.run(request_id, step, tool_request)
        except Exception as exc:
            logger.error("Ошибка выполнения инструмента %s: %s", tool_name, exc)
            return self._tool_error_action(
                tool_name=tool_name,
                params=params,
                step=step,
                error=str(exc),
            )

    def _normalize_tool_params(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Нормализует параметры инструментов для совместимости и системных вызовов."""
        normalized = dict(params or {})

        if tool_name == "query_plan":
            normalized.setdefault("query", self._current_query or "")
            return normalized

        if tool_name == "search":
            if not normalized.get("queries"):
                if normalized.get("query"):
                    normalized["queries"] = [str(normalized.pop("query"))]
                elif self._last_plan_summary and self._last_plan_summary.get(
                    "normalized_queries"
                ):
                    normalized["queries"] = list(
                        self._last_plan_summary["normalized_queries"]
                    )
                elif self._current_query:
                    normalized["queries"] = [self._current_query]

            # Всегда добавляем оригинальный запрос пользователя в subqueries.
            # LLM перефразирует запросы и теряет ключевые сущности —
            # оригинал обеспечивает BM25 keyword match по исходным терминам.
            if self._current_query and normalized.get("queries"):
                orig = self._current_query.strip()
                if orig and orig not in normalized["queries"]:
                    normalized["queries"].insert(0, orig)

            normalized.setdefault(
                "k",
                (
                    self._last_plan_summary or {}
                ).get("k_per_query", self.settings.search_k_per_query_default),
            )
            normalized.setdefault("route", "hybrid")

            # Прокидываем metadata_filters из query_plan (date_from/date_to, channel и пр.)
            if not normalized.get("filters") and self._last_plan_summary:
                plan_filters = self._last_plan_summary.get("metadata_filters")
                if isinstance(plan_filters, dict):
                    # Убираем None значения
                    clean_filters = {
                        k: v for k, v in plan_filters.items() if v is not None
                    }
                    if clean_filters:
                        normalized["filters"] = clean_filters

            return normalized

        if tool_name == "rerank":
            normalized.setdefault("query", self._current_query or "")
            hits = normalized.pop("hits", None) or self._last_search_hits
            if not normalized.get("docs") or not any(normalized["docs"]):
                normalized["docs"] = [
                    str(hit.get("text") or hit.get("snippet") or "")
                    for hit in hits
                    if isinstance(hit, dict)
                    and (hit.get("text") or hit.get("snippet"))
                ]
            if normalized.get("docs") and not normalized.get("top_n"):
                normalized["top_n"] = min(
                    len(normalized["docs"]),
                    max(1, int(self.settings.reranker_top_n)),
                )
            return normalized

        if tool_name == "compose_context":
            normalized.pop("hits", None)
            normalized.pop("raw_input", None)
            normalized.pop("hit_ids", None)  # Игнорируем LLM-выбранные ID
            normalized["query"] = self._current_query or ""

            # Всегда используем все результаты из _last_search_hits.
            # После rerank они уже отсортированы по релевантности.
            # compose_context сам ограничит по max_tokens_ctx.
            last_hits = getattr(self, "_last_search_hits", [])
            selected_hits: List[Dict[str, Any]] = [
                hit for hit in last_hits if isinstance(hit, dict)
            ]

            normalized_docs: List[Dict[str, Any]] = []
            missing_ids: List[str] = []
            for doc in selected_hits:
                doc_id = doc.get("id")
                text_value = (
                    doc.get("text")
                    or doc.get("snippet")
                    or doc.get("meta", {}).get("text")
                    or doc.get("metadata", {}).get("text")
                    or ""
                )
                if not text_value and doc_id:
                    missing_ids.append(str(doc_id))

                normalized_docs.append(
                    {
                        "id": doc_id,
                        "text": text_value,
                        "metadata": doc.get("metadata") or doc.get("meta", {}),
                        "dense_score": doc.get("dense_score"),
                    }
                )

            if missing_ids:
                try:
                    fetch_result = self.tool_runner.run(
                        self._current_request_id or "fetch-docs",
                        self._current_step,
                        ToolRequest(tool="fetch_docs", input={"ids": missing_ids}),
                    )
                    if fetch_result.output.ok:
                        fetched_docs = {
                            item.get("id"): item
                            for item in fetch_result.output.data.get("docs", [])
                            if isinstance(item, dict) and item.get("id")
                        }
                        for doc in normalized_docs:
                            if not doc.get("text") and doc.get("id") in fetched_docs:
                                fetched = fetched_docs[doc["id"]]
                                doc["text"] = fetched.get("text", "")
                                doc["metadata"] = fetched.get(
                                    "metadata", doc.get("metadata", {})
                                )
                except Exception as exc:
                    logger.warning("fetch_docs during compose_context failed: %s", exc)

            normalized["docs"] = normalized_docs
            normalized.setdefault("max_tokens_ctx", 1200)
            return normalized

        if tool_name == "fetch_docs":
            if "doc_ids" in normalized and "ids" not in normalized:
                normalized["ids"] = normalized.pop("doc_ids")
            return normalized

        if tool_name == "final_answer":
            # LLM иногда оборачивает аргументы в raw_input строку
            raw = normalized.pop("raw_input", None)
            if raw and not normalized.get("answer"):
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(parsed, dict):
                        normalized.update(parsed)
                except (json.JSONDecodeError, TypeError):
                    normalized["answer"] = str(raw)
            return normalized

        if tool_name == "verify":
            if "k" in normalized and "top_k" not in normalized:
                normalized["top_k"] = normalized.pop("k")
            normalized.setdefault("query", self._current_query or "")
            return normalized

        return normalized

    def _apply_action_state(self, action: AgentAction) -> None:
        """Сохраняет детерминированное состояние после успешных tool calls."""
        if not action.output.ok:
            return

        if action.tool == "query_plan":
            self._last_plan_summary = action.output.data.get("plan") or {}
            return

        if action.tool == "search":
            self._last_search_hits = list(action.output.data.get("hits", []) or [])
            self._last_search_route = action.output.data.get("route_used")
            self._agent_state.search_count += 1
            return

        if action.tool == "rerank":
            indices = action.output.data.get("indices") or []
            scores = action.output.data.get("scores") or []
            if not isinstance(indices, list) or not self._last_search_hits:
                return

            ranked_hits: List[Dict[str, Any]] = []
            used_indices: set[int] = set()
            for position, raw_idx in enumerate(indices):
                if not isinstance(raw_idx, int):
                    continue
                if raw_idx < 0 or raw_idx >= len(self._last_search_hits):
                    continue
                hit = dict(self._last_search_hits[raw_idx])
                if position < len(scores):
                    hit["rerank_score"] = scores[position]
                ranked_hits.append(hit)
                used_indices.add(raw_idx)

            if ranked_hits:
                tail_hits = [
                    dict(hit)
                    for idx, hit in enumerate(self._last_search_hits)
                    if idx not in used_indices
                ]
                self._last_search_hits = ranked_hits + tail_hits
            return

        if action.tool == "compose_context":
            self._last_compose_citations = list(
                action.output.data.get("citations", []) or []
            )
            self._last_coverage = float(
                action.output.data.get("citation_coverage", 0.0) or 0.0
            )

    def _format_observation(self, tool_response: ToolResponse, tool_name: str = "") -> str:
        """Форматирует результат инструмента для observation SSE."""
        if not tool_response.ok:
            return f"Ошибка: {tool_response.meta.error or 'Неизвестная ошибка'}"

        if not tool_response.data:
            return "Результат получен (пустые данные)"

        try:
            if tool_name == "search" and isinstance(
                tool_response.data.get("hits"), list
            ):
                hits = tool_response.data["hits"]
                hit_ids = [
                    hit.get("id", "unknown")
                    for hit in hits
                    if isinstance(hit, dict)
                ]
                route_used = tool_response.data.get("route_used", "unknown")
                total_found = tool_response.data.get("total_found", len(hits))
                return (
                    f"Found {len(hits)} documents (total: {total_found}). "
                    f"Route: {route_used}. Use these IDs for compose_context: {hit_ids}"
                )

            if tool_name == "rerank":
                indices = tool_response.data.get("indices", [])
                scores = tool_response.data.get("scores", [])
                top_n = tool_response.data.get("top_n", len(indices))
                score_str = ""
                if scores:
                    score_str = f", scores: [{', '.join(f'{s:.3f}' for s in scores[:5])}]"
                return (
                    f"Reranked {top_n} documents by relevance{score_str}. "
                    f"Call compose_context() to build context from reranked results."
                )

            if tool_name == "compose_context":
                coverage = float(
                    tool_response.data.get("citation_coverage", 0.0) or 0.0
                )
                citations_count = len(tool_response.data.get("citations", []))
                contexts_count = len(tool_response.data.get("contexts", []))
                return (
                    f"Composed context with {citations_count} citations, "
                    f"coverage: {coverage:.2f}, contexts: {contexts_count}"
                )

            if tool_name == "verify":
                verified = tool_response.data.get("verified", False)
                confidence = float(tool_response.data.get("confidence", 0.0) or 0.0)
                docs_found = tool_response.data.get("documents_found", 0)
                threshold = tool_response.data.get("threshold", 0.6)
                return (
                    "Verification: "
                    f"{verified} (confidence: {confidence:.3f}, "
                    f"threshold: {threshold}, docs: {docs_found})"
                )

            if tool_name == "query_plan":
                plan = tool_response.data.get("plan", {})
                queries = plan.get("normalized_queries", [])
                k_per_query = plan.get("k_per_query", 0)
                fusion = plan.get("fusion", "unknown")
                return f"Plan: {len(queries)} queries, k={k_per_query}, fusion={fusion}"

            if tool_name == "final_answer":
                answer = str(tool_response.data.get("answer", "")).strip()
                return f"Final answer prepared ({len(answer)} chars)"

            result_parts = []
            for key, value in tool_response.data.items():
                if key in {"error", "result", "answer", "route", "prompt"}:
                    result_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    result_parts.append(f"{key}: {len(value)}")
                elif isinstance(value, dict):
                    result_parts.append(f"{key}: object")
                else:
                    result_parts.append(f"{key}: {str(value)[:100]}")

            return "; ".join(result_parts) if result_parts else str(tool_response.data)
        except Exception as exc:
            logger.warning("Failed to format observation for %s: %s", tool_name, exc)
            return str(tool_response.data)[:500]

    def get_available_tools(self) -> Dict[str, Any]:
        """Возвращает обзор LLM-visible и системных инструментов агента."""
        tools_info = {
            "query_plan": {
                "description": "Декомпозирует пользовательский запрос на подзапросы",
                "parameters": {"query": "string"},
            },
            "search": {
                "description": "Выполняет гибридный поиск по Qdrant",
                "parameters": {"queries": "array<string>", "k": "integer"},
            },
            "rerank": {
                "description": "Переранжирует найденные документы",
                "parameters": {
                    "query": "string",
                    "docs": "array<string>",
                    "top_n": "integer",
                },
            },
            "compose_context": {
                "description": "Собирает prompt с цитатами и coverage",
                "parameters": {"hit_ids": "array<string>?"},
            },
            "final_answer": {
                "description": "Формирует финальный ответ пользователю",
                "parameters": {
                    "answer": "string",
                    "sources": "array<int>",
                },
            },
        }

        system_tools = {
            "fetch_docs": "Системная догрузка полных текстов по id",
            "verify": "Системная верификация финального ответа",
        }

        return {
            "tools": tools_info,
            "system_tools": system_tools,
            "total": len(tools_info),
            "note": "LLM видит только 5 tools schema; verify и fetch_docs вызываются системно.",
        }

    async def _verify_answer(
        self, final_answer: str, conversation_history: List[str]
    ) -> Dict[str, Any]:
        """Проверяет финальный ответ через verify tool."""
        try:
            original_query = conversation_history[0] if conversation_history else ""
            if not original_query.startswith("Human: "):
                original_query = "Human: " + original_query

            result = self.tool_runner.run(
                self._current_request_id or "verify",
                self._current_step,
                ToolRequest(
                    tool="verify",
                    input={
                        "query": original_query,
                        "claim": final_answer,
                        "top_k": 3,
                    },
                ),
            )

            if result.output.ok:
                return result.output.data

            return {
                "verified": False,
                "confidence": 0.0,
                "error": result.output.meta.error or "Tool execution failed",
            }
        except Exception as exc:
            logger.error("Error in _verify_answer: %s", exc, exc_info=True)
            return {"verified": False, "confidence": 0.0, "error": str(exc)}

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
        """Выполняет системный refinement: search → compose_context."""
        del agent_state

        actions: List[AgentAction] = []
        search_action = await self._execute_action(
            tool_name="search",
            params={
                "queries": [query],
                "filters": {},
                "k": 20,
                "route": "hybrid",
            },
            request_id=request_id,
            step=step,
        )
        if search_action is None:
            return actions

        actions.append(search_action)
        if not search_action.output.ok:
            return actions

        self._apply_action_state(search_action)
        compose_action = await self._execute_action(
            tool_name="compose_context",
            params={"hit_ids": []},
            request_id=request_id,
            step=step,
        )
        if compose_action is not None:
            actions.append(compose_action)
        return actions

    def _extract_tool_calls(
        self, assistant_message: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Приводит tool_calls llama-server к единому внутреннему формату."""
        raw_tool_calls = assistant_message.get("tool_calls") or []
        normalized_calls: List[Dict[str, Any]] = []

        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue

            function_block = (
                item.get("function") if isinstance(item.get("function"), dict) else item
            )
            tool_name = function_block.get("name")
            raw_arguments = function_block.get("arguments", {})

            if isinstance(raw_arguments, str):
                try:
                    loaded = json.loads(raw_arguments)
                    parsed_arguments = loaded if isinstance(loaded, dict) else {}
                except json.JSONDecodeError as jde:
                    logger.warning(
                        "JSON parse failed for %s args (len=%d): %s | first 200: %s",
                        tool_name, len(raw_arguments), jde, raw_arguments[:200],
                    )
                    parsed_arguments = {"raw_input": raw_arguments}
            elif isinstance(raw_arguments, dict):
                parsed_arguments = raw_arguments
            else:
                parsed_arguments = {}

            if tool_name:
                normalized_calls.append(
                    {
                        "id": item.get("id"),
                        "name": tool_name,
                        "arguments": parsed_arguments,
                    }
                )

        return normalized_calls

    def _assistant_message_for_history(
        self, assistant_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Сохраняет assistant message в chat history без искажения tool_calls."""
        message = {"role": "assistant"}
        if "content" in assistant_message:
            message["content"] = assistant_message.get("content")
        if "tool_calls" in assistant_message:
            message["tool_calls"] = assistant_message.get("tool_calls")
        return message

    def _tool_message_for_history(
        self,
        tool_call: Dict[str, Any],
        tool_name: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Сериализует результат инструмента в стандартный `role=tool` message."""
        message: Dict[str, Any] = {
            "role": "tool",
            "name": tool_name,
            "content": self._serialize_tool_payload(payload),
        }
        if tool_call.get("id"):
            message["tool_call_id"] = tool_call["id"]
        return message

    def _serialize_tool_payload(
        self, payload: Dict[str, Any], max_chars: int = 8000
    ) -> str:
        """Безопасно сериализует payload инструмента для chat history.

        compose_context.prompt — основной источник знаний для LLM, обрезать минимально.
        prompt уже ограничен max_tokens_ctx=1800 (~7200 chars) в compose_context tool.
        Остальные поля (contexts, docs) убираются — они дублируют prompt.
        """
        try:
            trimmed = dict(payload)
            if "contexts" in trimmed:
                trimmed.pop("contexts", None)
            if "docs" in trimmed:
                trimmed.pop("docs", None)
            # Debug: лог размера prompt для compose_context
            if "prompt" in trimmed:
                logger.info(
                    "tool_payload prompt_len=%d max=%d",
                    len(str(trimmed["prompt"])),
                    max_chars,
                )
            if "prompt" in trimmed and len(str(trimmed["prompt"])) > max_chars:
                trimmed["prompt"] = str(trimmed["prompt"])[:max_chars] + "…"
            serialized = json.dumps(trimmed, ensure_ascii=False, default=str)
            if len(serialized) > max_chars:
                serialized = serialized[:max_chars] + "…}"
            return serialized
        except Exception:
            return json.dumps({"error": "serialization_failed"}, ensure_ascii=False)

    @staticmethod
    def _get_step_tools(agent_state) -> List[Dict[str, Any]]:
        """Динамический набор tools в зависимости от состояния агента.

        final_answer доступен только после search — архитектурная гарантия
        что LLM не пропустит поиск. Промпт-инструкции ненадёжны для маленьких
        моделей (Qwen3-30B, 3B active params).
        """
        search_done = agent_state.search_count > 0
        if search_done:
            return AGENT_TOOLS  # все 5 tools
        # До search: только query_plan, search, rerank (без final_answer и compose_context)
        return [t for t in AGENT_TOOLS if t["function"]["name"] not in ("final_answer", "compose_context")]

    @staticmethod
    def _trim_messages(
        messages: list[dict[str, Any]], max_chars: int = 36000
    ) -> list[dict[str, Any]]:
        """Обрезает messages чтобы уложиться в context window LLM.

        Стратегия: всегда сохраняем system + user (первые 2 сообщения),
        затем оставляем последние N сообщений по бюджету символов.
        Бюджет 36K chars ≈ 12K токенов. Context window = 16384 токена,
        минус ~1500 (tools schema) минус ~500 (system) минус ~1000 (output) = ~13K для messages.
        """
        total = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in messages)
        if total <= max_chars:
            return messages

        # Первые 2 — system + user, всегда сохраняем
        head = messages[:2]
        tail = messages[2:]

        # Берём tail с конца пока не превышаем бюджет
        head_size = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in head)
        budget = max_chars - head_size
        kept: list[dict[str, Any]] = []
        for msg in reversed(tail):
            msg_size = len(json.dumps(msg, ensure_ascii=False, default=str))
            if budget - msg_size < 0 and kept:
                break
            kept.append(msg)
            budget -= msg_size
        kept.reverse()

        logger.debug(
            "Trimmed messages: %d → %d (from %d chars to ~%d)",
            len(messages),
            len(head) + len(kept),
            total,
            max_chars - budget,
        )
        return head + kept

    def _tool_error_action(
        self,
        tool_name: str,
        params: Dict[str, Any],
        step: int,
        error: str,
    ) -> AgentAction:
        """Строит псевдо-результат инструмента при локальной ошибке вызова."""
        return AgentAction(
            step=step,
            tool=tool_name,
            input=dict(params or {}),
            output=ToolResponse(
                ok=False,
                data={},
                meta=ToolMeta(took_ms=0, error=error),
            ),
        )

    def _build_final_payload(
        self,
        base_payload: Dict[str, Any],
        answer: str,
        verify_res: Dict[str, Any],
        agent_state: AgentState,
        request_id: str,
        step: int,
    ) -> Dict[str, Any]:
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
        final_payload.setdefault("citations", self._last_compose_citations)
        final_payload["coverage"] = agent_state.coverage
        final_payload["refinements"] = agent_state.refinement_count
        final_payload["route"] = self._last_search_route
        final_payload["plan"] = self._last_plan_summary
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
        return final_payload
