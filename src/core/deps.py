import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional
from fastapi import Depends, HTTPException
import chromadb
from llama_cpp import Llama

from adapters.chroma import Retriever
from services.qa_service import QAService
from services.query_planner_service import QueryPlannerService
from utils.model_downloader import auto_download_models, RECOMMENDED_MODELS
from core.settings import get_settings, Settings

logger = logging.getLogger(__name__)


@lru_cache
def get_chroma_client():
    """Создает и возвращает ChromaDB клиент"""
    settings = get_settings()
    try:
        # Попробуем HTTP клиент (для Docker compose)
        client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )
        # Проверим подключение
        client.heartbeat()
        logger.info(
            f"Подключение к ChromaDB HTTP: {settings.chroma_host}:{settings.chroma_port}"
        )
        return client
    except Exception as e:
        logger.warning(f"HTTP подключение не удалось ({e}), пробуем локальный клиент")
        # Fallback на локальный клиент
        return chromadb.PersistentClient(path=settings.chroma_path)


@lru_cache
def get_retriever() -> Retriever:
    settings = get_settings()
    client = get_chroma_client()
    """Создает и возвращает Retriever для поиска в ChromaDB"""
    embedding_model_key = settings.current_embedding_key
    collection_name = settings.current_collection

    # Получаем полное название модели из конфигурации
    if embedding_model_key in RECOMMENDED_MODELS["embedding"]:
        embedding_model = RECOMMENDED_MODELS["embedding"][embedding_model_key]["name"]
    else:
        # Fallback на прямое указание модели
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    logger.info(
        f"Используем embedding модель: {embedding_model} (коллекция: {collection_name})"
    )

    # Автоскачивание embedding модели если необходимо
    auto_download_embedding = (
        os.getenv("AUTO_DOWNLOAD_EMBEDDING", "true").lower() == "true"
    )
    if auto_download_embedding:
        try:
            from utils.model_downloader import download_embedding_model

            download_embedding_model(embedding_model, settings.cache_dir)
        except Exception as e:
            logger.warning(f"Не удалось скачать embedding модель: {e}")

    return Retriever(client, collection_name, embedding_model)


@lru_cache
def get_llm():
    """Создает и возвращает LLM модель с автоскачиванием"""
    settings = get_settings()
    # Конфигурация модели из настроек
    llm_model_key = settings.current_llm_key
    models_dir = settings.models_dir
    cache_dir = settings.cache_dir
    auto_download = os.getenv("AUTO_DOWNLOAD_LLM", "true").lower() == "true"

    # Параметры модели
    n_gpu_layers = int(os.getenv("LLM_GPU_LAYERS", "-1"))
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
        model_path = os.getenv("LLM_MODEL_PATH", f"{models_dir}/gpt-oss-20b-Q6_K.gguf")
        logger.info(f"Используем пользовательский путь к модели: {model_path}")

    # Диагностическое логирование окружения и пути
    try:
        import llama_cpp as _ll

        logger.info(f"llama_cpp version: {getattr(_ll, '__version__', 'unknown')}")
    except Exception as _:
        logger.info("llama_cpp version: <unavailable>")
    logger.info(
        f"ENV: CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}, "
        f"LLM_GPU_LAYERS={os.getenv('LLM_GPU_LAYERS', str(n_gpu_layers))}, "
        f"LLM_CONTEXT_SIZE={os.getenv('LLM_CONTEXT_SIZE', str(n_ctx))}, "
        f"LLM_THREADS={os.getenv('LLM_THREADS', str(n_threads))}"
    )
    logger.info(f"Путь к модели: {model_path}")
    if os.path.exists(model_path):
        try:
            sz = Path(model_path).stat().st_size
            logger.info(f"Размер файла модели: {sz / (1024**3):.2f} GB")
        except Exception:
            pass

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
        # Дополнительные подсказки по ошибке
        if "Failed to load model from file" in str(e):
            logger.error(
                "Возможные причины: некорректный путь/имя GGUF, поврежденный файл, несовместимая квантовка."
            )
            logger.error(
                f"Проверьте наличие файла и доступ: ls -lh {models_dir}; и переменную LLM_MODEL_PATH/KEY."
            )
        raise HTTPException(
            status_code=503, detail=f"Не удалось загрузить LLM модель: {str(e)}"
        )


@lru_cache
def get_query_planner() -> QueryPlannerService:
    settings = get_settings()
    llm = get_llm()
    return QueryPlannerService(llm, settings)


@lru_cache
def get_qa_service() -> QAService:
    settings = get_settings()
    retriever = get_retriever()
    llm = get_llm()
    top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    planner = get_query_planner() if settings.enable_query_planner else None
    return QAService(retriever, llm, top_k, settings=settings, planner=planner)


# === Новые зависимости для кеширования ===


@lru_cache
def get_redis_client(settings: Settings = Depends(get_settings)):
    """Создает Redis клиент если включено кеширование"""
    if not settings.redis_enabled:
        return None

    try:
        import redis

        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True,
        )
        # Проверяем подключение
        client.ping()
        logger.info(f"Redis подключен: {settings.redis_host}:{settings.redis_port}")
        return client
    except Exception as e:
        logger.warning(f"Redis недоступен: {e}")
        return None
