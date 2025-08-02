"""
FastAPI приложение для RAG системы
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from core.deps import get_qa_service
from services.qa_service import QAService
from schemas.qa import QARequest, QAResponse, QAResponseWithContext, ContextItem

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events для приложения"""
    logger.info("🚀 RAG API запускается...")

    # Проверяем доступность зависимостей при старте
    try:
        # Здесь можно добавить проверки подключения к ChromaDB и LLM
        logger.info("✅ Инициализация завершена успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации: {e}")
        raise

    yield

    logger.info("🛑 RAG API завершает работу...")


# Создаем FastAPI приложение
app = FastAPI(
    title="RAG QA API",
    description="API для ответов на вопросы с использованием Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware для разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене ограничить конкретными доменами
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def root():
    """Главная страница API"""
    return {
        "message": "RAG QA API v1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Проверка состояния сервиса"""
    return {"status": "healthy", "service": "rag-qa-api", "version": "1.0.0"}


@app.post(
    "/v1/qa", response_model=Union[QAResponse, QAResponseWithContext], tags=["qa"]
)
async def answer_question(
    request: QARequest, qa_service: QAService = Depends(get_qa_service)
) -> Union[QAResponse, QAResponseWithContext]:
    """
    Отвечает на вопрос пользователя используя RAG подход

    - **query**: Вопрос пользователя (обязательно)
    - **include_context**: Включить ли контекст документов в ответ (опционально)

    Возвращает ответ модели, основанный на найденных в базе документах.
    """
    try:
        logger.info(f"Получен запрос: {request.query[:100]}...")

        if request.include_context:
            # Возвращаем ответ с контекстом
            result = qa_service.answer_with_context(request.query)

            context_items = [
                ContextItem(
                    document=item["document"],
                    metadata=item["metadata"],
                    distance=item["distance"],
                )
                for item in result["context"]
            ]

            return QAResponseWithContext(
                answer=result["answer"],
                query=result["query"],
                context=context_items,
                context_count=result["context_count"],
            )
        else:
            # Возвращаем только ответ
            answer = qa_service.answer(request.query)
            return QAResponse(answer=answer, query=request.query)

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


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик исключений"""
    logger.error(f"Необработанная ошибка: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Внутренняя ошибка сервера",
            "message": str(exc) if os.getenv("DEBUG") else "Что-то пошло не так",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Запуск сервера на {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )
