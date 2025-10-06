"""
FastAPI приложение для RAG системы
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from api.v1.router import router as v1_router
from core.rate_limit import RateLimitMiddleware
from core.security import sanitize_for_logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Углубленное логирование для инструментов агента
logging.getLogger("services.tools.search").setLevel(logging.DEBUG)
logging.getLogger("services.tools.rerank").setLevel(logging.DEBUG)
logging.getLogger("services.tools.fetch_docs").setLevel(logging.DEBUG)
logging.getLogger("services.agent_service").setLevel(logging.DEBUG)


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

# Security middleware - порядок важен!

# 1. GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 2. Trusted Host для защиты от Host header injection
allowed_hosts = os.getenv("ALLOWED_HOSTS", "*").split(",")
if allowed_hosts != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

# 3. Rate limiting
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
    requests_per_hour=int(os.getenv("RATE_LIMIT_PER_HOUR", "1000")),
    burst_size=int(os.getenv("RATE_LIMIT_BURST", "10")),
    enable_exponential_backoff=True,
)

# 4. CORS middleware для разработки
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
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
    # Безопасное логирование без чувствительных данных
    sanitized_path = sanitize_for_logging(str(request.url.path))
    sanitized_error = sanitize_for_logging(str(exc))
    logger.error(f"Необработанная ошибка на {sanitized_path}: {sanitized_error}")

    # В продакшене не показываем детали ошибок
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Внутренняя ошибка сервера",
            "message": str(exc) if debug_mode else "Что-то пошло не так",
            "request_id": getattr(request.state, "request_id", None),
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
