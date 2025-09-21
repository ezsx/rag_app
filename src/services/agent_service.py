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

        # Системный промпт для ReAct
        self.system_prompt = """Ты — ReAct агент, который помогает отвечать на вопросы пользователей, используя доступные инструменты.

ФОРМАТ РАБОТЫ:
Мысли пошагово, используя следующий формат:

Thought: [твоё размышление о том, что нужно сделать]
Action: [название_инструмента] {"param": "value"}
Observation: [результат выполнения инструмента]

Повторяй этот цикл до получения достаточной информации для ответа.

Когда у тебя есть достаточно информации для полного ответа:
FinalAnswer: [твой итоговый ответ пользователю]

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
- router_select: выбирает оптимальный маршрут поиска (bm25/dense/hybrid)
- fetch_docs: получает документы по списку ID
- compose_context: собирает контекст из документов с цитированием
- dedup_diversify: удаляет дубликаты и диверсифицирует результаты
- verify: проверяет утверждения через поиск в базе знаний
- math_eval: безопасно вычисляет математические выражения
- time_now: получает текущее время в различных форматах
- multi_query_rewrite: генерирует дополнительные перефразы запроса
- web_search: ищет актуальную информацию в интернете
- temporal_normalize: нормализует временные выражения (даты, периоды) в ISO формат
- summarize: резюмирует длинные тексты, выделяя ключевые моменты
 - translate: переводит текст между языками (LLM/наивный)
 - fact_check_advanced: проверяет утверждение по базе знаний с confidence
 - semantic_similarity: считает косинусную близость между текстами
 - content_filter: базовая модерация контента (PII/URLs/токсичность)
 - export_to_formats: экспорт контента в md/txt/json

ПРАВИЛА:
1. Всегда начинай с Thought
2. Используй инструменты для получения информации
3. Анализируй результаты в Observation
4. Продолжай до получения полной информации
5. Завершай с FinalAnswer

Будь точным, логичным и полезным."""

    async def stream_agent_response(
        self, request: AgentRequest
    ) -> AsyncIterator[AgentStepEvent]:
        """Основная ReAct петля с SSE стримингом"""
        request_id = str(uuid.uuid4())
        step = 1
        conversation_history = []
        max_steps = min(request.max_steps, self.settings.agent_max_steps)

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
                    yield AgentStepEvent(
                        type="final",
                        data={
                            "answer": final_answer,
                            "step": step,
                            "total_steps": step,
                            "request_id": request_id,
                        },
                    )
                    logger.info(
                        f"ReAct петля завершена на шаге {step} для запроса {request_id}"
                    )
                    return

                # Выполняем действие если есть
                if action_text:
                    action_result = await self._execute_action(
                        action_text, request_id, step
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
                        observation = self._format_observation(action_result.output)
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

                step += 1

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

    async def _generate_step(
        self, conversation_history: List[str], request_id: str, step: int
    ) -> str:
        """Генерирует следующий шаг через LLM"""
        try:
            # Собираем промпт
            history_text = "\n".join(conversation_history)

            prompt = f"""Системная инструкция: {self.system_prompt}

Контекст разговора:
{history_text}

Продолжи рассуждение, следуя формату ReAct (Thought/Action/Observation или FinalAnswer):"""

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
                        if len(params_text) > 5000:
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

            # Создаем запрос к инструменту
            tool_request = ToolRequest(tool=tool_name, input=params)

            # Выполняем
            result = self.tool_runner.run(request_id, step, tool_request)
            return result

        except Exception as e:
            logger.error(f"Ошибка выполнения действия '{action_text}': {e}")
            return None

    def _format_observation(self, tool_response) -> str:
        """Форматирует результат инструмента для наблюдения"""
        if not tool_response.ok:
            return f"Ошибка: {tool_response.meta.error or 'Неизвестная ошибка'}"

        # Форматируем данные
        if not tool_response.data:
            return "Результат получен (пустые данные)"

        # Преобразуем в читаемый вид
        try:
            if isinstance(tool_response.data, dict):
                # Красиво форматируем ключевые поля
                result_parts = []
                for key, value in tool_response.data.items():
                    if key in ["error", "result", "answer", "route", "prompt"]:
                        result_parts.append(f"{key}: {value}")
                    elif isinstance(value, (list, dict)):
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

        except Exception:
            return "Результат получен (ошибка форматирования)"

    def get_available_tools(self) -> Dict[str, Any]:
        """Возвращает схемы всех доступных инструментов"""
        tools = {}

        # Базовая информация об инструментах
        tools_info = {
            "router_select": {
                "description": "Выбирает оптимальный маршрут поиска",
                "parameters": {"query": "string"},
            },
            "fetch_docs": {
                "description": "Получает документы по списку ID",
                "parameters": {"ids": "array of strings"},
            },
            "compose_context": {
                "description": "Собирает контекст из документов",
                "parameters": {"docs": "array", "max_tokens_ctx": "integer"},
            },
            "dedup_diversify": {
                "description": "Удаляет дубликаты и диверсифицирует результаты",
                "parameters": {"hits": "array", "lambda_": "float", "k": "integer"},
            },
            "verify": {
                "description": "Проверяет утверждения через поиск",
                "parameters": {"query": "string", "claim": "string"},
            },
            "math_eval": {
                "description": "Вычисляет математические выражения",
                "parameters": {"expression": "string"},
            },
            "time_now": {
                "description": "Получает текущее время",
                "parameters": {"format": "string", "timezone_name": "string"},
            },
            "multi_query_rewrite": {
                "description": "Генерирует дополнительные перефразы запроса",
                "parameters": {
                    "query": "string",
                    "existing_queries": "array of strings",
                    "target_count": "integer",
                },
            },
            "web_search": {
                "description": "Ищет актуальную информацию в интернете",
                "parameters": {
                    "query": "string",
                    "num_results": "integer (1-10)",
                    "region": "string",
                    "time_range": "string (d/w/m/y)",
                },
            },
            "temporal_normalize": {
                "description": "Нормализует временные выражения (даты, периоды) в ISO формат",
                "parameters": {"text": "string"},
            },
            "summarize": {
                "description": "Резюмирует длинные тексты, выделяя ключевые моменты",
                "parameters": {
                    "text": "string",
                    "max_sentences": "integer (1-10)",
                    "min_length": "integer (50-500)",
                    "mode": "string (extractive/bullets)",
                },
            },
            "translate": {
                "description": "Переводит текст между языками",
                "parameters": {
                    "text": "string",
                    "target_lang": "string (ru|en|...)",
                    "source_lang": "string?",
                    "max_length": "integer (<=2000)",
                },
            },
            "fact_check_advanced": {
                "description": "Проверяет утверждение с confidence по базе знаний",
                "parameters": {
                    "claim": "string",
                    "query": "string?",
                    "k": "integer (1-10)",
                },
            },
            "semantic_similarity": {
                "description": "Считает косинусную близость для пары/набора текстов",
                "parameters": {
                    "texts": "array of strings (>=2)",
                    "pairs": "array of [i,j]?",
                },
            },
            "content_filter": {
                "description": "Базовая модерация контента",
                "parameters": {
                    "text": "string",
                    "categories": "array of strings? (pii|toxicity|hate|sexual)",
                },
            },
            "export_to_formats": {
                "description": "Экспорт контента в поддерживаемые форматы",
                "parameters": {
                    "content": "string",
                    "fmt": "string (md|txt|json|pdf|docx)",
                    "filename_base": "string?",
                },
            },
        }

        return {
            "tools": tools_info,
            "total": len(tools_info),
            "note": "Все инструменты принимают JSON параметры",
        }
