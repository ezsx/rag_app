"""
ReAct Agent API endpoints
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from core.deps import get_agent_service
from core.settings import get_settings, Settings
from core.auth import get_current_user, require_read, TokenData
from schemas.agent import AgentRequest
from services.agent_service import AgentService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/agent/stream", tags=["agent"])
async def agent_stream(
    request: AgentRequest,
    fastapi_request: Request,
    agent_service: AgentService = Depends(get_agent_service),
    settings: Settings = Depends(get_settings),
    current_user: TokenData = Depends(require_read()),
) -> EventSourceResponse:
    """
    Пошаговый ReAct агент с SSE стримингом

    Агент выполняет цикл мышления-действия-наблюдения (ReAct) и транслирует
    каждый шаг через Server-Sent Events в реальном времени.

    **Параметры:**
    - **query**: Вопрос пользователя (обязательно)
    - **collection**: Название коллекции (опционально, использует текущую если не указано)
    - **model_profile**: Профиль модели (опционально)
    - **tools_allowlist**: Разрешенные инструменты (опционально, по умолчанию все)
    - **planner**: Использовать ли планировщик запросов (по умолчанию true)
    - **max_steps**: Максимальное количество шагов (1-10, по умолчанию 4)

    **События SSE:**
    - `step_started`: Начало нового шага
    - `thought`: Мысль агента
    - `tool_invoked`: Вызов инструмента
    - `observation`: Результат выполнения инструмента
    - `final`: Финальный ответ агента
    """

    async def event_generator():
        try:
            logger.info(f"Начинаем ReAct агент для запроса: {request.query[:100]}...")

            # Если указана коллекция, временно переключаемся
            original_collection = None
            if request.collection and request.collection != settings.current_collection:
                original_collection = settings.current_collection
                settings.update_collection(request.collection)
                logger.info(
                    f"Временно переключились на коллекцию: {request.collection}"
                )

                # Получаем новый agent_service с обновленной коллекцией
                from core.deps import get_agent_service

                agent_service_temp = get_agent_service()
                agent_service_to_use = agent_service_temp
            else:
                agent_service_to_use = agent_service

            # Стримим события от агента
            async for event in agent_service_to_use.stream_agent_response(request):
                # Проверяем отключение клиента
                if await fastapi_request.is_disconnected():
                    logger.info("Клиент отключился, останавливаем ReAct агент")
                    break

                # Отправляем событие
                yield {"event": event.type, "data": event.data, "retry": 3000}

                # Если это финальное событие, завершаем
                if event.type == "final":
                    break

            # Восстанавливаем оригинальную коллекцию если меняли
            if original_collection:
                settings.update_collection(original_collection)
                logger.info(
                    f"Восстановили оригинальную коллекцию: {original_collection}"
                )

            logger.info("ReAct агент завершен")

        except Exception as e:
            logger.error(f"Ошибка в ReAct агенте: {e}", exc_info=True)
            # Отправляем сообщение об ошибке
            yield {
                "event": "error",
                "data": {"error": f"Ошибка сервера: {str(e)}"},
                "retry": 3000,
            }
            yield {
                "event": "final",
                "data": {
                    "answer": f"Извините, произошла ошибка: {str(e)}",
                    "error": True,
                },
                "retry": 3000,
            }

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Отключаем буферизацию nginx
        },
    )


@router.get("/agent/tools", tags=["agent"])
async def list_tools(
    agent_service: AgentService = Depends(get_agent_service),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Список доступных инструментов агента

    Возвращает информацию о всех доступных инструментах с их описаниями
    и JSON-схемами параметров.

    **Возвращает:**
    - Список инструментов с описаниями
    - Схемы параметров для каждого инструмента
    - Общее количество доступных инструментов
    """
    try:
        tools_info = agent_service.get_available_tools()

        # Добавляем дополнительную метаинформацию
        tools_info.update(
            {
                "usage": 'Используйте формат: Action: tool_name {"param": "value"}',
                "supported_formats": {
                    "json": "Параметры передаются в формате JSON",
                    "example": 'Action: math_eval {"expression": "2 + 2"}',
                },
            }
        )

        return tools_info

    except Exception as e:
        logger.error(f"Ошибка получения списка инструментов: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить список инструментов: {str(e)}",
        )


@router.get("/agent/status", tags=["agent"])
async def agent_status(
    settings: Settings = Depends(get_settings),
    current_user: TokenData = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Статус и конфигурация агента

    Возвращает текущие настройки агента и информацию о его состоянии.
    """
    try:
        return {
            "status": "active",
            "configuration": {
                "max_steps": settings.agent_max_steps,
                "token_budget": settings.agent_token_budget,
                "tool_timeout": settings.agent_tool_timeout,
                "current_collection": settings.current_collection,
                "current_llm": settings.current_llm_key,
                "enable_query_planner": settings.enable_query_planner,
            },
            "features": {
                "react_reasoning": True,
                "sse_streaming": True,
                "tool_execution": True,
                "fallback_qa": True,
            },
        }

    except Exception as e:
        logger.error(f"Ошибка получения статуса агента: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить статус агента: {str(e)}",
        )
