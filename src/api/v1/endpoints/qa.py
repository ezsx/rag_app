"""
QA (Question Answering) endpoints
"""

import logging
from typing import Union, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from core.deps import (
    get_qa_service,
    get_redis_client,
    get_retriever,
    get_query_planner,
)
from core.settings import get_settings, Settings
from services.qa_service import QAService
from schemas.qa import QARequest, QAResponse, QAResponseWithContext, ContextItem

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_from_cache(redis_client, cache_key: str) -> Optional[dict]:
    """Получить результат из кеша"""
    if not redis_client:
        return None
    try:
        import json

        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Найден кеш для: {cache_key[:50]}...")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Ошибка чтения кеша: {e}")
    return None


async def save_to_cache(redis_client, cache_key: str, data: dict, ttl: int):
    """Сохранить результат в кеш"""
    if not redis_client:
        return
    try:
        import json

        redis_client.setex(
            cache_key, ttl, json.dumps(data, ensure_ascii=False, default=str)
        )
        logger.info(f"Результат сохранен в кеш: {cache_key[:50]}...")
    except Exception as e:
        logger.warning(f"Ошибка записи в кеш: {e}")


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
        logger.info(f"Получен QA запрос: {request.query[:100]}...")

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
            cached_result = await get_from_cache(redis_client, cache_key)
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
                await save_to_cache(
                    redis_client, cache_key, cache_data, settings.cache_ttl
                )

        else:
            answer = qa_service.answer(request.query)
            response = QAResponse(answer=answer, query=request.query)

            # Сохраняем в кеш
            if settings.redis_enabled:
                cache_data = {"answer": answer, "query": request.query}
                await save_to_cache(
                    redis_client, cache_key, cache_data, settings.cache_ttl
                )

        return response

    except ValidationError as e:
        logger.error(f"Ошибка валидации: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Ошибка валидации запроса: {str(e)}",
        )
    except FileNotFoundError as e:
        logger.error(f"Файл модели не найден: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM модель недоступна. Проверьте конфигурацию.",
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}",
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
            logger.info(f"Начинаем SSE стрим для запроса: {request.query[:100]}...")

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

            logger.info(f"SSE стрим завершен. Отправлено токенов: {token_count}")

        except Exception as e:
            logger.error(f"Ошибка в SSE стриме: {e}")
            # Отправляем сообщение об ошибке
            yield {"event": "error", "data": f"Ошибка: {str(e)}", "retry": 3000}
            yield {"event": "end", "data": "[DONE]", "retry": 3000}

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Отключаем буферизацию nginx
        },
    )
