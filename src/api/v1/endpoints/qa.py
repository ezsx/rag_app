"""
QA (Question Answering) endpoints
"""

import logging
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from core.deps import (
    get_qa_service,
    get_redis_client,
)
from core.settings import Settings, get_settings
from schemas.qa import ContextItem, QARequest, QAResponse, QAResponseWithContext
from services.qa_service import QAService

logger = logging.getLogger(__name__)
router = APIRouter()


from core.cache import cache_get, cache_set


@router.post(
    "/qa", response_model=Union[QAResponse, QAResponseWithContext], tags=["qa"]
)
async def answer_question(
    request: QARequest,

    qa_service: QAService = Depends(get_qa_service),
    redis_client=Depends(get_redis_client),
    settings: Settings = Depends(get_settings),
) -> Union[QAResponse, QAResponseWithContext]:
    """
    Отвечает на вопрос пользователя используя RAG подход

    - **query**: Вопрос пользователя (обязательно)
    - **include_context**: Включить ли контекст документов в ответ (опционально)
    - **collection**: Название коллекции (опционально, использует текущую если не указано)

    Возвращает ответ модели, основанный на найденных в базе документах.
    """
    try:
        logger.info("Получен QA запрос: %s...", request.query[:100])

        # FIX-01B: НЕ мутируем global settings для collection override
        if request.collection and request.collection != settings.current_collection:
            logger.info(
                "QA запрос с collection=%s (default=%s) — "
                "collection override пока не поддержан, используем default",
                request.collection, settings.current_collection,
            )

        # Подготовка ключа кеша
        # FIX-01B: cache key привязан к фактической (default) коллекции
        cache_key = f"qa:{hash(request.query + str(request.include_context) + settings.current_collection)}"

        # Проверяем кеш
        if settings.redis_enabled:
            cached_result = cache_get(redis_client, cache_key)
            if cached_result:
                if request.include_context:
                    context_items = [
                        ContextItem(**item) for item in cached_result.get("context", [])
                    ]
                    return QAResponseWithContext(
                        answer=cached_result["answer"],
                        query=cached_result["query"],
                        context=context_items,
                        context_count=cached_result.get("context_count", 0),
                    )
                else:
                    return QAResponse(
                        answer=cached_result["answer"], query=cached_result["query"]
                    )

        # Генерируем ответ
        response: Union[QAResponse, QAResponseWithContext]
        if request.include_context:
            result = qa_service.answer_with_context(request.query)
            context_items = [
                ContextItem(
                    document=item["document"],
                    metadata=item["metadata"],
                    distance=item["distance"],
                )
                for item in result["context"]
            ]

            response = QAResponseWithContext(
                answer=result["answer"],
                query=result["query"],
                context=context_items,
                context_count=result["context_count"],
            )

            # Сохраняем в кеш
            if settings.redis_enabled:
                cache_data = {
                    "answer": result["answer"],
                    "query": result["query"],
                    "context": [
                        {
                            "document": item["document"],
                            "metadata": item["metadata"],
                            "distance": item["distance"],
                        }
                        for item in result["context"]
                    ],
                    "context_count": result["context_count"],
                }
                cache_set(
                    redis_client, cache_key, cache_data, settings.cache_ttl
                )

        else:
            answer = qa_service.answer(request.query)
            response = QAResponse(answer=answer, query=request.query)

            # Сохраняем в кеш
            if settings.redis_enabled:
                cache_data = {"answer": answer, "query": request.query}
                cache_set(
                    redis_client, cache_key, cache_data, settings.cache_ttl
                )

        return response

    except ValidationError as e:
        logger.error("Ошибка валидации: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Ошибка валидации запроса: {e!s}",
        )
    except FileNotFoundError as e:
        logger.error("Файл модели не найден: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM модель недоступна. Проверьте конфигурацию.",
        )
    except Exception as e:  # broad: endpoint safety net
        logger.error("Неожиданная ошибка: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {e!s}",
        )


@router.post("/qa/stream", tags=["qa"])
async def qa_stream(
    request: QARequest,
    fastapi_request: Request,

    qa_service: QAService = Depends(get_qa_service),
    settings: Settings = Depends(get_settings),
) -> EventSourceResponse:
    """
    Стримящий ответ на вопрос пользователя используя RAG подход через Server-Sent Events

    - **query**: Вопрос пользователя (обязательно)
    - **include_context**: Включить ли контекст документов (опционально)
    - **collection**: Название коллекции (опционально, использует текущую если не указано)

    Возвращает поток токенов от LLM модели в реальном времени через SSE.
    """

    async def event_generator():
        try:
            logger.info("Начинаем SSE стрим для запроса: %s...", request.query[:100])

            # FIX-01B: НЕ мутируем global settings
            if request.collection and request.collection != settings.current_collection:
                logger.info(
                    "QA stream с collection=%s — collection override не поддержан",
                    request.collection,
                )

            token_count = 0

            async for token in qa_service.stream_answer(
                request.query, request.include_context
            ):
                if await fastapi_request.is_disconnected():
                    logger.info("Клиент отключился, останавливаем стрим")
                    break

                token_count += 1
                yield {"event": "token", "data": token, "retry": 3000}

            # Отправляем завершающий токен
            yield {"event": "end", "data": "[DONE]", "retry": 3000}

            logger.info("SSE стрим завершен. Отправлено токенов: %s", token_count)

        except Exception as e:  # broad: endpoint safety net
            logger.error("Ошибка в SSE стриме: %s", e)
            # Отправляем сообщение об ошибке
            yield {"event": "error", "data": f"Ошибка: {e!s}", "retry": 3000}
            yield {"event": "end", "data": "[DONE]", "retry": 3000}

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Отключаем буферизацию nginx
        },
    )
