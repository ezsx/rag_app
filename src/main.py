"""
FastAPI приложение для RAG системы
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.v1.router import router as v1_router

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
        # Опциональный прогрев LLM. По умолчанию выключен для быстрых рестартов.
        if os.getenv("LLM_WARMUP", "false").lower() == "true":
            try:
                from core.deps import get_llm

                _ = get_llm()
            except Exception as e:
                logger.error(f"LLM warmup failed: {e}")
                # не падаем: пусть API поднимется, но лог останется
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

# Подключаем API v1 роутер
app.include_router(v1_router)


@app.get("/", tags=["root"])
async def root():
    """Главная страница API"""
    return {
        "message": "RAG QA API v1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/v1/health",
        "api_v1": "/v1",
        "available_endpoints": {
            "qa": "/v1/qa",
            "search": "/v1/search",
            "collections": "/v1/collections",
            "models": "/v1/models",
            "ingest": "/v1/ingest",
        },
    }


@app.get("/health", tags=["health"])
async def health_check_legacy():
    """Легаси проверка состояния сервиса (редирект на v1)"""
    return {
        "status": "healthy",
        "service": "rag-qa-api",
        "version": "1.0.0",
        "note": "Используйте /v1/health",
    }


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
