"""
Models endpoints для управления LLM и embedding моделями
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from core.settings import get_settings, Settings
from utils.model_downloader import RECOMMENDED_MODELS
from schemas.qa import (
    ModelInfo,
    ModelType,
    AvailableModelsResponse,
    SelectModelRequest,
    SelectModelResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models", response_model=AvailableModelsResponse, tags=["models"])
async def list_available_models(
    settings: Settings = Depends(get_settings),
) -> AvailableModelsResponse:
    """
    Получает список всех доступных моделей для LLM и embedding

    Возвращает информацию о поддерживаемых моделях и указывает
    какие модели используются в данный момент.
    """
    try:
        logger.info("Получение списка доступных моделей...")

        # Формируем список LLM моделей
        llm_models = []
        for key, config in RECOMMENDED_MODELS["llm"].items():
            llm_models.append(
                ModelInfo(
                    key=key,
                    name=config.get("repo", "Unknown"),
                    description=config.get("description", "Нет описания"),
                    type=ModelType.LLM,
                )
            )

        # Формируем список embedding моделей
        embedding_models = []
        for key, config in RECOMMENDED_MODELS["embedding"].items():
            embedding_models.append(
                ModelInfo(
                    key=key,
                    name=config.get("name", "Unknown"),
                    description=config.get("description", "Нет описания"),
                    type=ModelType.EMBEDDING,
                )
            )

        logger.info(
            f"Доступно LLM моделей: {len(llm_models)}, embedding моделей: {len(embedding_models)}"
        )

        return AvailableModelsResponse(
            llm_models=llm_models,
            embedding_models=embedding_models,
            current_llm=settings.current_llm_key,
            current_embedding=settings.current_embedding_key,
        )

    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить список моделей: {str(e)}",
        )


@router.post("/models/select", response_model=SelectModelResponse, tags=["models"])
async def select_model(
    request: SelectModelRequest, settings: Settings = Depends(get_settings)
) -> SelectModelResponse:
    """
    Выбирает активную модель для LLM или embedding

    - **model_key**: Ключ модели из списка доступных
    - **model_type**: Тип модели (llm или embedding)

    После выбора система будет использовать новую модель для всех
    последующих запросов. Происходит горячая перезагрузка без
    перезапуска сервиса.
    """
    try:
        model_key = request.model_key.strip()
        model_type = request.model_type

        logger.info(f"Попытка выбрать {model_type.value} модель: {model_key}")

        # Проверяем существование модели в конфигурации
        if model_type == ModelType.LLM:
            if model_key not in RECOMMENDED_MODELS["llm"]:
                available_keys = list(RECOMMENDED_MODELS["llm"].keys())
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"LLM модель '{model_key}' не найдена. Доступные: {available_keys}",
                )

            # Обновляем LLM модель
            old_model = settings.current_llm_key
            settings.update_llm_model(model_key)

            # Получаем описание модели
            model_config = RECOMMENDED_MODELS["llm"][model_key]
            message = f"LLM модель изменена с '{old_model}' на '{model_key}'. {model_config.get('description', '')}"

        elif model_type == ModelType.EMBEDDING:
            if model_key not in RECOMMENDED_MODELS["embedding"]:
                available_keys = list(RECOMMENDED_MODELS["embedding"].keys())
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Embedding модель '{model_key}' не найдена. Доступные: {available_keys}",
                )

            # Обновляем embedding модель
            old_model = settings.current_embedding_key
            settings.update_embedding_model(model_key)

            # Получаем описание модели
            model_config = RECOMMENDED_MODELS["embedding"][model_key]
            message = f"Embedding модель изменена с '{old_model}' на '{model_key}'. {model_config.get('description', '')}"

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неизвестный тип модели: {model_type}",
            )

        logger.info(f"Модель успешно изменена: {model_key}")

        return SelectModelResponse(
            model_key=model_key, model_type=model_type, message=message
        )

    except HTTPException:
        raise  # Пробрасываем HTTP ошибки как есть
    except Exception as e:
        logger.error(f"Ошибка при выборе модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось выбрать модель: {str(e)}",
        )


@router.get("/models/{model_type}/current", tags=["models"])
async def get_current_model(
    model_type: ModelType, settings: Settings = Depends(get_settings)
) -> dict:
    """
    Получает информацию о текущей активной модели указанного типа

    - **model_type**: Тип модели (llm или embedding)

    Возвращает ключ, название и описание текущей модели.
    """
    try:
        if model_type == ModelType.LLM:
            current_key = settings.current_llm_key
            if current_key in RECOMMENDED_MODELS["llm"]:
                config = RECOMMENDED_MODELS["llm"][current_key]
                return {
                    "model_key": current_key,
                    "model_type": model_type,
                    "name": config.get("repo", "Unknown"),
                    "description": config.get("description", "Нет описания"),
                    "filename": config.get("filename", "Unknown"),
                }
            else:
                return {
                    "model_key": current_key,
                    "model_type": model_type,
                    "name": "Custom model",
                    "description": "Пользовательская модель",
                    "filename": "Unknown",
                }

        elif model_type == ModelType.EMBEDDING:
            current_key = settings.current_embedding_key
            if current_key in RECOMMENDED_MODELS["embedding"]:
                config = RECOMMENDED_MODELS["embedding"][current_key]
                return {
                    "model_key": current_key,
                    "model_type": model_type,
                    "name": config.get("name", "Unknown"),
                    "description": config.get("description", "Нет описания"),
                }
            else:
                return {
                    "model_key": current_key,
                    "model_type": model_type,
                    "name": "Custom model",
                    "description": "Пользовательская модель",
                }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неизвестный тип модели: {model_type}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении информации о модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить информацию о модели: {str(e)}",
        )
