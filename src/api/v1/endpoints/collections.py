"""
Collections endpoints для управления коллекциями ChromaDB
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from core.deps import get_chroma_client
from core.settings import get_settings, Settings
from schemas.qa import (
    CollectionInfo,
    CollectionsResponse,
    SelectCollectionRequest,
    SelectCollectionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/collections", response_model=CollectionsResponse, tags=["collections"])
async def list_collections(
    chroma_client=Depends(get_chroma_client), settings: Settings = Depends(get_settings)
) -> CollectionsResponse:
    """
    Получает список всех доступных коллекций ChromaDB

    Возвращает информацию о каждой коллекции включая количество документов
    и метаданные, а также указывает текущую активную коллекцию.
    """
    try:
        logger.info("Получение списка коллекций ChromaDB...")

        # Получаем список коллекций
        collections = chroma_client.list_collections()

        collection_infos = []
        for collection in collections:
            try:
                # Получаем информацию о коллекции
                collection_obj = chroma_client.get_collection(collection.name)
                count = collection_obj.count()
                metadata = collection.metadata or {}

                collection_infos.append(
                    CollectionInfo(name=collection.name, count=count, metadata=metadata)
                )

            except Exception as e:
                logger.warning(
                    f"Не удалось получить информацию о коллекции {collection.name}: {e}"
                )
                # Добавляем коллекцию с базовой информацией
                collection_infos.append(
                    CollectionInfo(
                        name=collection.name,
                        count=0,
                        metadata={"error": f"Недоступна: {str(e)}"},
                    )
                )

        logger.info(f"Найдено {len(collection_infos)} коллекций")

        return CollectionsResponse(
            collections=collection_infos, current_collection=settings.current_collection
        )

    except Exception as e:
        logger.error(f"Ошибка при получении списка коллекций: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить список коллекций: {str(e)}",
        )


@router.post(
    "/collections/select", response_model=SelectCollectionResponse, tags=["collections"]
)
async def select_collection(
    request: SelectCollectionRequest,
    chroma_client=Depends(get_chroma_client),
    settings: Settings = Depends(get_settings),
) -> SelectCollectionResponse:
    """
    Выбирает активную коллекцию для работы с системой

    - **collection_name**: Название коллекции для активации

    После выбора все последующие запросы QA и Search будут использовать
    эту коллекцию по умолчанию, пока не будет выбрана другая.
    """
    try:
        collection_name = request.collection_name.strip()
        logger.info(f"Попытка выбрать коллекцию: {collection_name}")

        # Проверяем существование коллекции
        try:
            collection = chroma_client.get_collection(collection_name)
            document_count = collection.count()
        except Exception as e:
            logger.error(f"Коллекция {collection_name} не найдена: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Коллекция '{collection_name}' не найдена",
            )

        # Обновляем текущую коллекцию в настройках
        old_collection = settings.current_collection
        settings.update_collection(collection_name)

        logger.info(f"Коллекция изменена: {old_collection} → {collection_name}")

        return SelectCollectionResponse(
            collection_name=collection_name,
            document_count=document_count,
            message=f"Коллекция '{collection_name}' успешно выбрана. Найдено {document_count} документов.",
        )

    except HTTPException:
        raise  # Пробрасываем HTTP ошибки как есть
    except Exception as e:
        logger.error(f"Ошибка при выборе коллекции: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось выбрать коллекцию: {str(e)}",
        )


@router.get(
    "/collections/{collection_name}/info",
    response_model=CollectionInfo,
    tags=["collections"],
)
async def get_collection_info(
    collection_name: str, chroma_client=Depends(get_chroma_client)
) -> CollectionInfo:
    """
    Получает детальную информацию о конкретной коллекции

    - **collection_name**: Название коллекции

    Возвращает количество документов и метаданные коллекции.
    """
    try:
        logger.info(f"Получение информации о коллекции: {collection_name}")

        # Проверяем существование и получаем информацию
        try:
            collection = chroma_client.get_collection(collection_name)
            count = collection.count()
            metadata = getattr(collection, "metadata", {}) or {}

        except Exception as e:
            logger.error(f"Коллекция {collection_name} не найдена: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Коллекция '{collection_name}' не найдена",
            )

        return CollectionInfo(name=collection_name, count=count, metadata=metadata)

    except HTTPException:
        raise  # Пробрасываем HTTP ошибки как есть
    except Exception as e:
        logger.error(f"Ошибка при получении информации о коллекции: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить информацию о коллекции: {str(e)}",
        )
