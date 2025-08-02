import os
import logging
from functools import lru_cache
from pathlib import Path
from fastapi import Depends, HTTPException
import chromadb
from llama_cpp import Llama

from adapters.chroma import Retriever
from services.qa_service import QAService
from utils.model_downloader import auto_download_models, RECOMMENDED_MODELS

logger = logging.getLogger(__name__)


@lru_cache
def get_chroma_client():
    """Создает и возвращает ChromaDB клиент"""
    chroma_path = os.getenv("CHROMA_PATH", "/data/chroma")
    return chromadb.PersistentClient(path=chroma_path)


def get_retriever(client=Depends(get_chroma_client)):
    """Создает и возвращает Retriever для поиска в ChromaDB"""
    collection_name = os.getenv("CHROMA_COLLECTION", "news_demo4")
    embedding_model_key = os.getenv("EMBEDDING_MODEL_KEY", "multilingual-e5-large")

    # Получаем полное название модели из конфигурации
    if embedding_model_key in RECOMMENDED_MODELS["embedding"]:
        embedding_model = RECOMMENDED_MODELS["embedding"][embedding_model_key]["name"]
    else:
        # Fallback на прямое указание модели
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    logger.info(f"Используем embedding модель: {embedding_model}")

    # Автоскачивание embedding модели если необходимо
    auto_download_embedding = (
        os.getenv("AUTO_DOWNLOAD_EMBEDDING", "true").lower() == "true"
    )
    if auto_download_embedding:
        try:
            from utils.model_downloader import download_embedding_model

            cache_dir = os.getenv("TRANSFORMERS_CACHE", "/models/.cache")
            download_embedding_model(embedding_model, cache_dir)
        except Exception as e:
            logger.warning(f"Не удалось скачать embedding модель: {e}")

    return Retriever(client, collection_name, embedding_model)


@lru_cache
def get_llm():
    """Создает и возвращает LLM модель с автоскачиванием"""
    # Конфигурация модели
    llm_model_key = os.getenv("LLM_MODEL_KEY", "vikhr-7b-instruct")
    models_dir = os.getenv("MODELS_DIR", "/models")
    cache_dir = os.getenv("TRANSFORMERS_CACHE", "/models/.cache")
    auto_download = os.getenv("AUTO_DOWNLOAD_LLM", "true").lower() == "true"

    # Параметры модели
    n_gpu_layers = int(os.getenv("LLM_GPU_LAYERS", "0"))  # CPU по умолчанию
    n_ctx = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
    n_threads = int(os.getenv("LLM_THREADS", "8"))

    # Определяем путь к модели
    if llm_model_key in RECOMMENDED_MODELS["llm"]:
        model_config = RECOMMENDED_MODELS["llm"][llm_model_key]
        model_filename = model_config["filename"]
        model_path = os.path.join(models_dir, model_filename)
        logger.info(f"Используем LLM модель: {model_config['description']}")
    else:
        # Fallback на прямое указание пути
        model_path = os.getenv(
            "LLM_MODEL_PATH", f"{models_dir}/Vikhr-7B-instruct-Q4_K_M.gguf"
        )
        logger.info(f"Используем пользовательский путь к модели: {model_path}")

    # Проверяем существование модели
    if not os.path.exists(model_path):
        if auto_download and llm_model_key in RECOMMENDED_MODELS["llm"]:
            logger.info(f"🔄 Модель не найдена, запускаем автоскачивание...")

            # Автоскачивание
            downloaded_path, _ = auto_download_models(
                llm_model_key=llm_model_key,
                embedding_model_key="",  # Не скачиваем embedding здесь
                models_dir=models_dir,
                cache_dir=cache_dir,
            )

            if downloaded_path and os.path.exists(downloaded_path):
                model_path = downloaded_path
                logger.info(f"✅ Модель успешно скачана: {model_path}")
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Не удалось скачать LLM модель. Проверьте подключение к интернету.",
                )
        else:
            raise FileNotFoundError(f"LLM model not found at {model_path}")

    # Загружаем модель
    try:
        logger.info(f"📚 Загружаем LLM модель: {os.path.basename(model_path)}")
        logger.info(
            f"   GPU слои: {n_gpu_layers}, Контекст: {n_ctx}, Потоки: {n_threads}"
        )

        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

        logger.info("✅ LLM модель успешно загружена")
        return llm

    except Exception as e:
        logger.error(f"❌ Ошибка загрузки LLM модели: {e}")
        raise HTTPException(
            status_code=503, detail=f"Не удалось загрузить LLM модель: {str(e)}"
        )


def get_qa_service(retriever=Depends(get_retriever), llm=Depends(get_llm)):
    """Создает и возвращает QA сервис"""
    top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    return QAService(retriever, llm, top_k)
