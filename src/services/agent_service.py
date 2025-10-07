"""
ReAct Agent Service с пошаговым мышлением и наблюдаемостью
"""

import json
import logging
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Callable

from core.settings import Settings
from core.security import security_manager, sanitize_for_logging
from schemas.agent import AgentRequest, AgentStepEvent, ToolRequest, AgentAction
from services.tools.tool_runner import ToolRunner
from services.qa_service import QAService

logger = logging.getLogger(__name__)


class AgentState:
    """Tracks dynamic state of the agent between steps."""

    def __init__(self):
        self.coverage: float = 0.0
        self.refinement_count: int = 0
        self.max_refinements: int = 1  # allow at most 1 extra round


class AgentService:
    """ReAct агент с пошаговым мышлением и наблюдаемостью"""

    def __init__(
        self,
        llm_factory: Callable,
        tool_runner: ToolRunner,
        settings: Settings,
        qa_service: Optional[QAService] = None,
    ):
        self.llm_factory = llm_factory
        self.tool_runner = tool_runner
        self.settings = settings
        self.qa_service = qa_service
        self._current_request_id: Optional[str] = None
        self._current_step: int = 1  # Текущий шаг для системных вызовов
        self._current_query: Optional[str] = None  # Текущий запрос пользователя

        # Системный промпт для ReAct (на английском для лучшей работы Qwen)
        self.system_prompt = """You are a ReAct agent that helps answer user questions using available tools.

CRITICAL: Always respond in the SAME LANGUAGE as the user's query (Russian, English, etc.). Your Thought and FinalAnswer MUST match the user's language.

WORKFLOW:
Think step-by-step using this format:

Thought: [your reasoning about what to do next]
Action: [tool_name] {"param": "value"}
Observation: [tool execution result]

Repeat this cycle until you have enough information to answer.

When you have sufficient information for a complete answer:
FinalAnswer: [your final answer to the user]

AVAILABLE TOOLS AND CONTRACTS:

1. router_select: selects optimal search route
   Parameters: {"query": "string"}

2. query_plan: creates search plan
   Parameters: {"query": "string"}

3. search: performs hybrid search on collection
   Parameters: {"queries": ["string"], "route": "hybrid|bm25|dense", "k": int}
   Returns: {"hits": [{"id": "...", "text": "...", "metadata": {...}}], "route_used": "..."}

4. rerank: re-ranks documents by relevance
   Parameters: {"query": "string", "docs": ["text1", "text2", ...], "top_n": int}

5. fetch_docs: retrieves documents by ID list
   Parameters: {"ids": ["id1", "id2", ...]}

6. compose_context: assembles context from documents with citations
   Parameters: {"hit_ids": ["id1", "id2", ...]}
   IMPORTANT: Use hit_ids from latest search results. System automatically converts ids to docs.
   Returns: {"prompt": "...", "citations": [...], "citation_coverage": float}

7. verify: verifies claims through knowledge base search
   Parameters: {"query": "string", "claim": "string", "top_k": int}

RULES:
1. Always start with Thought
2. Use tools to gather information
3. MANDATORY: After EVERY successful search, you MUST call compose_context with hit_ids from search results
4. NEVER generate FinalAnswer without calling compose_context first
5. Pass ONLY specified parameters to tools
6. End with FinalAnswer ONLY after compose_context
7. CRITICAL: Match the user's query language in ALL your outputs (Thought, FinalAnswer, etc.)

WORKFLOW EXAMPLE:
Step 1: Thought → search → Observation (got 5 hits with ids)
Step 2: Thought → compose_context {hit_ids: [ids from step 1]} → Observation (got context with citations)
Step 3: Thought → FinalAnswer (based on context from step 2, with citation numbers [1], [2], etc.)

SYSTEM DETERMINISTIC LOGIC:
- After compose_context, system auto-checks citation coverage (>=80%)
- If coverage insufficient, system performs additional search round
- Before final answer, system may verify claims
- Maximum 1 additional search round to avoid infinite loops

Be accurate, logical, and helpful."""

    async def stream_agent_response(
        self, request: AgentRequest
    ) -> AsyncIterator[AgentStepEvent]:
        """Основная ReAct петля с SSE стримингом"""
        request_id = str(uuid.uuid4())
        step = 1
        conversation_history = []
        requested_steps = request.max_steps or self.settings.agent_default_steps
        max_steps = min(max(requested_steps, 1), self.settings.agent_max_steps)
        agent_state = AgentState()  # Track coverage and refinements

        self._current_request_id = request_id
        self._current_query = request.query  # Сохраняем текущий запрос для нормализации
        try:
            # Валидация и санитизация входных данных
            is_valid, violations = security_manager.validate_input(
                request.query, context="prompt"
            )
            if not is_valid:
                logger.warning(
                    f"Security violations in request {request_id}: {violations}"
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

            # Санитизированный запрос для логирования
            sanitized_query = sanitize_for_logging(request.query, max_length=100)
            logger.info(
                f"Начинаем ReAct петлю для запроса: {sanitized_query} (ID: {request_id})"
            )

            # Уведомляем о начале
            yield AgentStepEvent(
                type="step_started",
                data={
                    "step": 1,
                    "request_id": request_id,
                    "max_steps": max_steps,
                    "query": request.query,
                },
            )

            # Добавляем исходный запрос в историю
            conversation_history.append(f"Human: {request.query}")

            while step <= max_steps:
                logger.info(f"Шаг {step}/{max_steps} для запроса {request_id}")

                # Генерируем следующий шаг
                llm_response = await self._generate_step(
                    conversation_history, request_id, step
                )

                # Парсим ответ
                thought, action_text, final_answer = self._parse_llm_response(
                    llm_response
                )

                # Отправляем мысль если есть
                if thought:
                    yield AgentStepEvent(
                        type="thought", data={"content": thought, "step": step}
                    )
                    conversation_history.append(f"Thought: {thought}")

                # Проверяем на финальный ответ
                if final_answer:
                    # Before finalizing, optionally verify the answer
                    verified_answer = final_answer

                    if self.settings.enable_verify_step:
                        logger.debug(
                            "Starting answer verification for request %s", request_id
                        )
                        verify_res = await self._verify_answer(
                            final_answer, conversation_history
                        )
                        logger.debug(
                            "Verification result: verified=%s, confidence=%.3f",
                            verify_res.get("verified", False),
                            verify_res.get("confidence", 0.0),
                        )

                        if (
                            not verify_res.get("verified", False)
                            and agent_state.refinement_count
                            < agent_state.max_refinements
                        ):
                            # Low confidence in answer, trigger one refinement round
                            agent_state.refinement_count += 1

                            # Send system-generated thought about verification failure
                            verify_thought = "The answer may be incomplete or not fully verified; conducting additional verification search."
                            yield AgentStepEvent(
                                type="thought",
                                data={
                                    "content": verify_thought,
                                    "step": step,
                                    "system_generated": True,
                                    "verification": True,
                                },
                            )
                            conversation_history.append(f"Thought: {verify_thought}")

                            # Execute verification refinement search with error handling
                            try:
                                verify_refinement_result = (
                                    await self._perform_refinement(
                                        request.query, agent_state, request_id, step
                                    )
                                )

                                if verify_refinement_result:
                                    # Send tool invocation for verification refinement
                                    yield AgentStepEvent(
                                        type="tool_invoked",
                                        data={
                                            "tool": verify_refinement_result.tool,
                                            "input": verify_refinement_result.input,
                                            "step": step,
                                            "verification_refinement": True,
                                        },
                                    )

                                    # Send observation for verification refinement
                                    verify_ref_observation = self._format_observation(
                                        verify_refinement_result.output,
                                        verify_refinement_result.tool,
                                    )
                                    yield AgentStepEvent(
                                        type="observation",
                                        data={
                                            "content": verify_ref_observation,
                                            "success": verify_refinement_result.output.ok,
                                            "step": step,
                                            "took_ms": verify_refinement_result.output.meta.took_ms,
                                            "verification_refinement": True,
                                        },
                                    )

                                    conversation_history.append(
                                        f"Action: search (verification refinement)"
                                    )
                                    conversation_history.append(
                                        f"Observation: {verify_ref_observation}"
                                    )

                                    # Continue loop for additional verification
                                    continue
                                else:
                                    logger.warning(
                                        f"Verification refinement failed for request {request_id}"
                                    )
                                    # Continue without verification refinement if it fails
                            except Exception as e:
                                logger.error(
                                    f"Error during verification refinement for request {request_id}: {e}"
                                )
                                # Continue without verification refinement if it fails
                        elif not verify_res.get("verified", False):
                            # Verification failed but no more refinements available
                            verified_answer += (
                                " (⚠️ Answer not verified with high confidence)"
                            )

                    # Добавляем информацию о верификации в финальный ответ
                    final_data = {
                        "answer": verified_answer,
                        "step": step,
                        "total_steps": step,
                        "request_id": request_id,
                        "coverage": agent_state.coverage,
                        "refinements": agent_state.refinement_count,
                    }

                    if self.settings.enable_verify_step and verify_res:
                        final_data.update(
                            {
                                "verification": {
                                    "verified": verify_res.get("verified", False),
                                    "confidence": verify_res.get("confidence", 0.0),
                                    "documents_found": verify_res.get(
                                        "documents_found", 0
                                    ),
                                }
                            }
                        )

                    yield AgentStepEvent(
                        type="final",
                        data=final_data,
                    )

                    logger.info(
                        "ReAct loop completed at step %d for request %s "
                        "(coverage: %.2f, refinements: %d, verified: %s, confidence: %.3f)",
                        step,
                        request_id,
                        agent_state.coverage,
                        agent_state.refinement_count,
                        verify_res.get("verified", False) if verify_res else "N/A",
                        verify_res.get("confidence", 0.0) if verify_res else 0.0,
                    )
                    return

                # Выполняем действие если есть
                if action_text:
                    # Нормализация для всех инструментов будет выполнена в _execute_action через _normalize_tool_params
                    normalized_action_text = action_text

                    logger.debug(
                        "Agent executing tool | tool=%s | payload=%s",
                        normalized_action_text.split(" ", 1)[0],
                        normalized_action_text,
                    )
                    action_result = await self._execute_action(
                        normalized_action_text, request_id, step
                    )

                    if action_result:
                        # Отправляем информацию о вызове инструмента
                        yield AgentStepEvent(
                            type="tool_invoked",
                            data={
                                "tool": action_result.tool,
                                "input": action_result.input,
                                "step": step,
                            },
                        )

                        # Отправляем результат наблюдения
                        observation = self._format_observation(
                            action_result.output, action_result.tool
                        )
                        yield AgentStepEvent(
                            type="observation",
                            data={
                                "content": observation,
                                "success": action_result.output.ok,
                                "step": step,
                                "took_ms": action_result.output.meta.took_ms,
                            },
                        )

                        # Добавляем в историю
                        conversation_history.append(f"Action: {action_text}")
                        conversation_history.append(f"Observation: {observation}")

                        if action_result.tool == "search" and action_result.output.ok:
                            action_result.output.data.setdefault(
                                "route",
                                action_result.output.data.get("route_used"),
                            )
                            search_hits = action_result.output.data.get("hits", [])
                            self._last_search_hits = search_hits
                            self._last_search_route = action_result.output.data.get(
                                "route_used"
                            )

                            # Логируем информацию о найденных документах
                            if search_hits:
                                hit_ids = [
                                    hit.get("id")
                                    for hit in search_hits[:5]
                                    if hit.get("id")
                                ]
                                logger.debug(
                                    "Search found %d hits, first IDs: %s",
                                    len(search_hits),
                                    hit_ids,
                                )
                            else:
                                logger.warning("Search returned no hits")

                        if (
                            action_result.tool == "rerank"
                            and action_result.output.meta.error
                        ):
                            logger.debug(
                                "Rerank tool returned error: %s",
                                action_result.output.meta.error,
                            )

                        # Если compose_context дал покрытие, проверить необходимость refinement
                        if action_result.tool == "compose_context":
                            coverage = action_result.output.data.get(
                                "citation_coverage", 0.0
                            )
                            citations = action_result.output.data.get("citations", [])
                            contexts = action_result.output.data.get("contexts", [])

                            agent_state.coverage = coverage

                            logger.debug(
                                "Compose context: coverage=%.2f, citations=%d, contexts=%d",
                                coverage,
                                len(citations),
                                len(contexts),
                            )

                            if self._should_attempt_refinement(
                                coverage, agent_state.refinement_count
                            ):
                                # Not enough coverage, trigger refinement search
                                agent_state.refinement_count += 1

                                # Send system-generated thought about refinement
                                refinement_thought = "Not enough information coverage, conducting additional search."
                                yield AgentStepEvent(
                                    type="thought",
                                    data={
                                        "content": refinement_thought,
                                        "step": step,
                                        "system_generated": True,
                                    },
                                )
                                conversation_history.append(
                                    f"Thought: {refinement_thought}"
                                )

                                # Execute refinement search with error handling
                                try:
                                    refinement_result = await self._perform_refinement(
                                        request.query, agent_state, request_id, step
                                    )

                                    if refinement_result:
                                        # Send tool invocation for refinement
                                        yield AgentStepEvent(
                                            type="tool_invoked",
                                            data={
                                                "tool": refinement_result.tool,
                                                "input": refinement_result.input,
                                                "step": step,
                                                "refinement": True,
                                            },
                                        )

                                        # Send observation for refinement
                                        ref_observation = self._format_observation(
                                            refinement_result.output,
                                            refinement_result.tool,
                                        )
                                        yield AgentStepEvent(
                                            type="observation",
                                            data={
                                                "content": ref_observation,
                                                "success": refinement_result.output.ok,
                                                "step": step,
                                                "took_ms": refinement_result.output.meta.took_ms,
                                                "refinement": True,
                                            },
                                        )

                                        conversation_history.append(
                                            f"Action: search (refinement)"
                                        )
                                        conversation_history.append(
                                            f"Observation: {ref_observation}"
                                        )

                                        # Continue loop without incrementing step for refinement
                                        continue
                                    else:
                                        logger.warning(
                                            f"Refinement failed for request {request_id}"
                                        )
                                        # Continue without refinement if it fails
                                except Exception as e:
                                    logger.error(
                                        f"Error during refinement for request {request_id}: {e}"
                                    )
                                    # Continue without refinement if it fails

                step += 1
                self._current_step = step  # Обновляем текущий шаг

                # Уведомляем о следующем шаге
                if step <= max_steps:
                    yield AgentStepEvent(
                        type="step_started",
                        data={"step": step, "request_id": request_id},
                    )

            # Если достигли максимума шагов без финального ответа
            logger.warning(
                f"Достигнут максимум шагов ({max_steps}) без финального ответа для запроса {request_id}"
            )

            # Попробуем использовать fallback через QAService
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
                except Exception as e:
                    logger.error(f"Fallback через QAService не удался: {e}")

            # Последний fallback
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

        except Exception as e:
            logger.error(f"Ошибка в ReAct петле: {e}", exc_info=True)
            yield AgentStepEvent(
                type="final",
                data={
                    "answer": f"Извините, произошла ошибка при обработке запроса: {str(e)}",
                    "step": step,
                    "request_id": request_id,
                    "error": str(e),
                },
            )
        finally:
            self._current_request_id = None
            self._current_query = None

    async def _generate_step(
        self, conversation_history: List[str], request_id: str, step: int
    ) -> str:
        """Генерирует следующий шаг через LLM"""
        try:
            # Ограничение истории скользящим окном для предотвращения overflow
            MAX_HISTORY_ITEMS = 10  # Последние 5 Thought-Action-Observation триплетов
            if len(conversation_history) > MAX_HISTORY_ITEMS:
                # Сохраняем первый элемент (исходный запрос) и последние MAX_HISTORY_ITEMS
                conversation_history = [conversation_history[0]] + conversation_history[
                    -MAX_HISTORY_ITEMS:
                ]

            # Собираем промпт
            history_text = "\n".join(conversation_history)

            # Определяем язык запроса пользователя (первая строка истории)
            user_query_lang = (
                "Russian"
                if any(
                    ord(c) >= 0x0400 and ord(c) <= 0x04FF
                    for c in conversation_history[0]
                )
                else "English"
            )

            prompt = f"""System Instruction: {self.system_prompt}

REMINDER: The user's query is in {user_query_lang}. You MUST respond in {user_query_lang} for both Thought and FinalAnswer.

Conversation Context:
{history_text}

Continue reasoning following ReAct format (Thought/Action/Observation or FinalAnswer):"""

            # Получаем LLM
            llm = self.llm_factory()

            # Генерируем ответ
            response = llm(
                prompt,
                max_tokens=self.settings.agent_token_budget,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.2,
                stop=["Human:", "Observation:"],
                seed=42,
            )

            return response["choices"][0]["text"].strip()

        except Exception as e:
            logger.error(f"Ошибка генерации шага {step} для запроса {request_id}: {e}")
            return f"Ошибка генерации: {str(e)}"

    def _parse_llm_response(
        self, response: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Парсит ответ LLM на компоненты: thought, action, final_answer"""
        thought = None
        action = None
        final_answer = None

        # Ищем Thought
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|$)", response, re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Ищем Action
        action_match = re.search(
            r"Action:\s*(.*?)(?=\nObservation:|$)", response, re.DOTALL | re.IGNORECASE
        )
        if action_match:
            action = action_match.group(1).strip()

        # Ищем FinalAnswer
        final_match = re.search(
            r"FinalAnswer:\s*(.*)", response, re.DOTALL | re.IGNORECASE
        )
        if final_match:
            final_answer = final_match.group(1).strip()

        return thought, action, final_answer

    async def _execute_action(
        self, action_text: str, request_id: str, step: int
    ) -> Optional[AgentAction]:
        """Выполняет действие (вызов инструмента)"""
        try:
            # Парсим строку действия: "tool_name {json_params}"
            parts = action_text.split(" ", 1)
            if len(parts) < 1:
                logger.warning(f"Невалидное действие: {action_text}")
                return None

            tool_name = parts[0].strip()

            # Парсим параметры JSON с валидацией
            params = {}
            if len(parts) > 1:
                params_text = parts[1].strip()
                if params_text:
                    # Проверка на SQL injection и другие атаки
                    is_valid, violations = security_manager.validate_input(params_text)
                    if not is_valid:
                        logger.warning(
                            f"Security violations in tool params: {violations}"
                        )
                        return None

                    try:
                        # Ограничение размера JSON для предотвращения DoS
                        if len(params_text) > security_manager.max_input_length:
                            logger.warning(
                                f"Tool parameters too large: {len(params_text)}"
                            )
                            return None

                        params = json.loads(params_text)

                        # Дополнительная проверка вложенности
                        def check_depth(obj, depth=0, max_depth=5):
                            if depth > max_depth:
                                return False
                            if isinstance(obj, dict):
                                return all(
                                    check_depth(v, depth + 1, max_depth)
                                    for v in obj.values()
                                )
                            elif isinstance(obj, list):
                                return all(
                                    check_depth(v, depth + 1, max_depth) for v in obj
                                )
                            return True

                        if not check_depth(params):
                            logger.warning("Tool parameters too deeply nested")
                            return None

                    except json.JSONDecodeError:
                        sanitized_text = sanitize_for_logging(params_text)
                        logger.warning(
                            f"Невалидный JSON в параметрах: {sanitized_text}"
                        )
                        params = {"raw_input": params_text}

            # Нормализуем параметры через единый метод
            params = self._normalize_tool_params(tool_name, params)

            # Создаем запрос к инструменту
            tool_request = ToolRequest(tool=tool_name, input=params)

            # Выполняем
            result = self.tool_runner.run(request_id, step, tool_request)
            return result

        except Exception as e:
            logger.error(f"Ошибка выполнения действия '{action_text}': {e}")
            return None

    def _normalize_tool_params(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Нормализует параметры инструментов для совместимости."""

        if tool_name == "compose_context":
            # Убираем лишние параметры, которые LLM может генерировать
            params.pop("query", None)
            params.pop("hits", None)

            # Извлекаем hit_ids
            hit_ids: List[str] = params.pop("hit_ids", []) or []

            # Получаем последние hits из поиска
            last_hits = getattr(self, "_last_search_hits", [])
            hits_by_id = {
                hit.get("id"): hit
                for hit in last_hits
                if isinstance(hit, dict) and hit.get("id")
            }

            # Логируем для диагностики
            logger.debug(
                "compose_context: hit_ids=%s, available_hits=%s, last_hits_count=%s",
                hit_ids,
                list(hits_by_id.keys()),
                len(last_hits),
            )

            # Выбираем документы по hit_ids или берём все последние hits
            selected_hits: List[Dict[str, Any]] = []
            if hit_ids:
                for hid in hit_ids:
                    match = hits_by_id.get(hid)
                    if match:
                        selected_hits.append(match)
                    else:
                        logger.warning(
                            "compose_context: hit_id %s not found in last_hits", hid
                        )

            if not selected_hits:
                selected_hits = [hit for hit in last_hits if isinstance(hit, dict)]
                logger.debug(
                    "compose_context: using all last_hits, count=%s", len(selected_hits)
                )

            # Нормализуем в формат docs
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
                    missing_ids.append(doc_id)

                normalized_docs.append(
                    {
                        "id": doc_id,
                        "text": text_value,
                        "metadata": doc.get("metadata") or doc.get("meta", {}),
                    }
                )

            if missing_ids:
                try:
                    fetch_request = ToolRequest(
                        tool="fetch_docs",
                        input={"ids": missing_ids},
                    )
                    fetch_result = self.tool_runner.run(
                        self._current_request_id or "compose_context",
                        self._current_step,
                        fetch_request,
                    )
                    if fetch_result and fetch_result.output.ok:
                        fetched_docs = fetch_result.output.data.get("docs", [])
                        fetched_by_id = {
                            item.get("id"): item
                            for item in fetched_docs
                            if isinstance(item, dict)
                        }
                        for doc in normalized_docs:
                            doc_id = doc.get("id")
                            if doc_id in fetched_by_id and not doc.get("text"):
                                fetched = fetched_by_id[doc_id]
                                doc["text"] = fetched.get("text", "")
                                doc["metadata"] = fetched.get("metadata", {})
                    elif fetch_result and fetch_result.output.meta.error:
                        logger.warning(
                            "compose_context: fetch_docs returned error for ids %s: %s",
                            missing_ids,
                            fetch_result.output.meta.error,
                        )
                except Exception as fetch_error:
                    logger.warning(
                        "compose_context: failed to fetch docs for ids %s: %s",
                        missing_ids,
                        fetch_error,
                    )

            params["docs"] = normalized_docs

            # Ограничиваем max_tokens_ctx для предотвращения overflow
            params.setdefault("max_tokens_ctx", 1200)

        elif tool_name == "fetch_docs":
            # Нормализация hit_ids → ids
            hit_ids = params.pop("hit_ids", None)
            doc_ids = params.pop("doc_ids", None)
            if hit_ids is not None and "ids" not in params:
                params["ids"] = hit_ids
            elif doc_ids is not None and "ids" not in params:
                params["ids"] = doc_ids

        elif tool_name == "rerank":
            # Автоматически добавляем docs из последних hits если не переданы
            if "docs" not in params:
                hits_payload = params.pop("hits", None)
                if not hits_payload:
                    hits_payload = getattr(self, "_last_search_hits", [])
                if hits_payload:
                    params["docs"] = [
                        item.get("snippet") or item.get("text") or ""
                        for item in hits_payload
                        if item
                    ]
            params.setdefault("query", getattr(self, "_current_query", None) or "")

        elif tool_name == "verify":
            # Нормализация k → top_k
            k_value = params.pop("k", None)
            if k_value is not None and "top_k" not in params:
                params["top_k"] = k_value

        return params

    def _format_observation(self, tool_response, tool_name: str = "") -> str:
        """Форматирует результат инструмента для наблюдения"""
        if not tool_response.ok:
            return f"Ошибка: {tool_response.meta.error or 'Неизвестная ошибка'}"

        # Форматируем данные
        if not tool_response.data:
            return "Результат получен (пустые данные)"

        # Преобразуем в читаемый вид
        try:
            if isinstance(tool_response.data, dict):
                # Специальная обработка для search - показываем hit IDs
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
                    return f"Found {len(hits)} documents (total: {total_found}). Route: {route_used}. Use these IDs for compose_context: {hit_ids}"

                # Специальная обработка для compose_context
                if tool_name == "compose_context":
                    coverage = tool_response.data.get("citation_coverage", 0.0)
                    citations_count = len(tool_response.data.get("citations", []))
                    contexts_count = len(tool_response.data.get("contexts", []))
                    return f"Composed context with {citations_count} citations, coverage: {coverage:.2f}, contexts: {contexts_count}"

                # Специальная обработка для verify
                if tool_name == "verify":
                    verified = tool_response.data.get("verified", False)
                    confidence = tool_response.data.get("confidence", 0.0)
                    docs_found = tool_response.data.get("documents_found", 0)
                    threshold = tool_response.data.get("threshold", 0.6)
                    evidence_count = len(tool_response.data.get("evidence", []))
                    return f"Verification: {verified} (confidence: {confidence:.3f}, threshold: {threshold}, docs: {docs_found}, evidence: {evidence_count})"

                # Специальная обработка для query_plan
                if tool_name == "query_plan":
                    plan = tool_response.data.get("plan", {})
                    queries = plan.get("normalized_queries", [])
                    k_per_query = plan.get("k_per_query", 0)
                    fusion = plan.get("fusion", "unknown")
                    return f"Plan: {len(queries)} queries, k={k_per_query}, fusion={fusion}"

                # Красиво форматируем ключевые поля
                result_parts = []
                for key, value in tool_response.data.items():
                    if key in ["error", "result", "answer", "route", "prompt"]:
                        result_parts.append(f"{key}: {value}")
                    elif isinstance(value, (list, dict)):
                        if key == "citations" and tool_name == "compose_context":
                            # Для compose_context показываем детали цитат
                            citations = value if isinstance(value, list) else []
                            if citations:
                                citation_ids = [
                                    c.get("id", "unknown") for c in citations[:3]
                                ]
                                result_parts.append(
                                    f"citations: {citation_ids}{'...' if len(citations) > 3 else ''}"
                                )
                            else:
                                result_parts.append("citations: none")
                        else:
                            result_parts.append(
                                f"{key}: {len(value) if isinstance(value, list) else 'object'}"
                            )
                    else:
                        result_parts.append(f"{key}: {str(value)[:100]}")

                return (
                    "; ".join(result_parts) if result_parts else str(tool_response.data)
                )
            else:
                return str(tool_response.data)[:500]  # Ограничиваем длину

        except Exception as e:
            logger.error(f"Error formatting observation for {tool_name}: {e}")
            return "Результат получен (ошибка форматирования)"

    def get_available_tools(self) -> Dict[str, Any]:
        """Возвращает схемы всех доступных инструментов"""
        tools = {}

        # Базовая информация об инструментах
        tools_info = {
            "router_select": {
                "description": "Выбирает оптимальный маршрут поиска (bm25/dense/hybrid)",
                "parameters": {"query": "string"},
            },
            "query_plan": {
                "description": "Создает план поиска для заданного запроса",
                "parameters": {"query": "string"},
            },
            "search": {
                "description": "Выполняет гибридный поиск по коллекции с RRF слиянием",
                "parameters": {
                    "queries": "array of strings",
                    "filters": {
                        "date_from": "YYYY-MM-DD",
                        "date_to": "YYYY-MM-DD",
                        "channel": "string",
                    },
                    "k": "integer (default 10)",
                    "route": "string (bm25|dense|hybrid, default hybrid)",
                },
            },
            "rerank": {
                "description": "Переранжирует документы по релевантности к запросу",
                "parameters": {
                    "query": "string",
                    "docs": "array of strings",
                    "top_n": "integer?",
                },
            },
            "fetch_docs": {
                "description": "Получает документы по списку ID",
                "parameters": {"ids": "array of strings"},
            },
            "compose_context": {
                "description": "Собирает контекст из документов с цитированием",
                "parameters": {"docs": "array", "max_tokens_ctx": "integer"},
            },
            "verify": {
                "description": "Проверяет утверждения через поиск в базе знаний",
                "parameters": {"query": "string", "claim": "string"},
            },
        }

        return {
            "tools": tools_info,
            "total": len(tools_info),
            "note": "Все инструменты принимают JSON параметры",
        }

    async def _verify_answer(
        self, final_answer: str, conversation_history: List[str]
    ) -> Dict[str, Any]:
        """Verify the final answer's claims using the knowledge base."""
        try:
            # Use the original user query and answer as input to verify tool
            original_query = conversation_history[0] if conversation_history else ""
            if not original_query.startswith("Human: "):
                original_query = "Human: " + original_query

            logger.debug(
                "Verifying answer: query='%s', claim_length=%d",
                original_query[:100],
                len(final_answer),
            )

            request_id_for_verify = self._current_request_id or "verify"
            verify_step = self._current_step if hasattr(self, "_current_step") else 1

            result = self.tool_runner.run(
                request_id_for_verify,
                verify_step,
                ToolRequest(
                    tool="verify",
                    input={
                        "query": original_query,
                        "claim": final_answer,
                        "top_k": 3,
                    },
                ),
            )

            logger.debug(
                "Verify tool result: ok=%s, data_keys=%s",
                result.output.ok if result else False,
                list(result.output.data.keys()) if result and result.output.ok else [],
            )

            if result and result.output.ok:
                verify_data = result.output.data
                logger.debug(
                    "Verification successful: verified=%s, confidence=%.3f, docs=%d",
                    verify_data.get("verified", False),
                    verify_data.get("confidence", 0.0),
                    verify_data.get("documents_found", 0),
                )
                return verify_data
            else:
                error_msg = "Tool execution failed"
                if result and result.output.meta.error:
                    error_msg = result.output.meta.error
                logger.warning("Verification failed: %s", error_msg)
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "error": error_msg,
                }

        except Exception as e:
            logger.error(f"Error in _verify_answer: {e}", exc_info=True)
            return {"verified": False, "confidence": 0.0, "error": str(e)}

    def _should_attempt_refinement(
        self, coverage: float, refinement_count: int
    ) -> bool:
        """Check if a refinement is warranted based on coverage and refinement count."""
        return (
            coverage < self.settings.coverage_threshold
            and refinement_count < self.settings.max_refinements
        )

    async def _perform_refinement(
        self, query: str, agent_state: AgentState, request_id: str, step: int
    ) -> Optional[AgentAction]:
        """Execute the refinement process with expanded search parameters."""
        try:
            # Increase retrieval scope for refinement - use hybrid_top_bm25 as base
            new_k = min(
                self.settings.hybrid_top_bm25 * 2, 200
            )  # double BM25 results, cap at 200

            # Prepare a refined search action
            refine_input = {
                "queries": [query],
                "filters": {},
                "k": new_k,
                "route": "hybrid",  # use hybrid for broader coverage
            }

            # Execute the search tool directly
            action_text = f"search {json.dumps(refine_input)}"
            result = await self._execute_action(action_text, request_id, step)

            return result

        except Exception as e:
            logger.error(f"Error in _perform_refinement: {e}")
            return None
