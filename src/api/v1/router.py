"""
Главный роутер для API v1
"""

from fastapi import APIRouter

from api.v1.endpoints import system, qa, search, collections, models, ingest

# Создаем главный роутер для API v1
router = APIRouter(prefix="/v1")

# Подключаем все endpoints
router.include_router(system.router, tags=["system"])
router.include_router(qa.router, tags=["qa"])
router.include_router(search.router, tags=["search"])
router.include_router(collections.router, tags=["collections"])
router.include_router(models.router, tags=["models"])
router.include_router(ingest.router, tags=["ingest"])
