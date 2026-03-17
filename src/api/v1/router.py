"""
Главный роутер для API v1
"""

from fastapi import APIRouter

from api.v1.endpoints import (
    system,
    qa,
    search,
    # collections — отключён до переписывания на Qdrant (SPEC-RAG-06).
    # Phase 0 код импортирует get_chroma_client из deps.py, который удалён в Phase 1.
    models,
    ingest,
    agent,
    auth,
)

# Создаем главный роутер для API v1
router = APIRouter(prefix="/v1")

# Подключаем все endpoints
router.include_router(system.router, tags=["system"])
router.include_router(qa.router, tags=["qa"])
router.include_router(search.router, tags=["search"])
# router.include_router(collections.router, tags=["collections"])  # Phase 0 legacy, disabled
router.include_router(models.router, tags=["models"])
router.include_router(ingest.router, tags=["ingest"])
router.include_router(agent.router, tags=["agent"])
router.include_router(auth.router, tags=["auth"])
