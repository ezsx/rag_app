"""
Системные endpoints для проверки состояния API
"""

from fastapi import APIRouter, Depends
from core.settings import get_settings, Settings

router = APIRouter()


@router.get("/health", tags=["system"])
async def health_check():
    """Проверка состояния сервиса"""
    return {"status": "healthy", "service": "rag-qa-api", "version": "1.0.0"}


@router.get("/info", tags=["system"])
async def system_info(settings: Settings = Depends(get_settings)):
    """Информация о текущих настройках системы"""
    return {
        "current_llm_model": settings.current_llm_key,
        "current_embedding_model": settings.current_embedding_key,
        "current_collection": settings.current_collection,
        "redis_enabled": settings.redis_enabled,
        "chroma_host": settings.chroma_host,
        "chroma_port": settings.chroma_port,
    }
