"""
Утилиты для автоматического скачивания моделей
"""

import os
import logging
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


def download_file_with_progress(url: str, local_path: str) -> bool:
    """
    Скачивает файл с индикатором прогресса

    Args:
        url: URL файла
        local_path: Локальный путь для сохранения

    Returns:
        True если скачивание успешно
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "wb") as file, tqdm(
            desc=f"Скачивание {os.path.basename(local_path)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

        logger.info(f"✅ Модель скачана: {local_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка скачивания {url}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def download_llm_model_from_hf(
    model_repo: str, filename: str, local_dir: str, cache_dir: Optional[str] = None
) -> Optional[str]:
    """
    Скачивает LLM модель с Hugging Face Hub

    Args:
        model_repo: Репозиторий модели (например, "microsoft/DialoGPT-medium")
        filename: Имя файла модели (например, "model.gguf")
        local_dir: Локальная директория для сохранения
        cache_dir: Директория кэша HF

    Returns:
        Путь к скачанной модели или None
    """
    try:
        logger.info(f"🔄 Скачивание LLM модели {model_repo}/{filename}...")

        # Скачиваем модель через HuggingFace Hub
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=filename,
            local_dir=local_dir,
            cache_dir=cache_dir,
        )

        logger.info(f"✅ LLM модель скачана: {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"❌ Ошибка скачивания LLM модели: {e}")
        return None


def download_embedding_model(model_name: str, cache_dir: Optional[str] = None) -> bool:
    """
    Скачивает embedding модель с Hugging Face

    Args:
        model_name: Название модели
        cache_dir: Директория кэша

    Returns:
        True если скачивание успешно
    """
    try:
        logger.info(f"🔄 Скачивание embedding модели {model_name}...")

        # Устанавливаем переменные окружения для кэша
        if cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["HF_HOME"] = cache_dir

        # Скачиваем модель полностью
        snapshot_download(
            repo_id=model_name, cache_dir=cache_dir, local_files_only=False
        )

        logger.info(f"✅ Embedding модель скачана: {model_name}")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка скачивания embedding модели: {e}")
        return False


# Предопределенные модели для русского языка
RECOMMENDED_MODELS = {
    "llm": {
        "vikhr-7b-instruct": {
            "repo": "oblivious/Vikhr-7B-instruct-GGUF",
            "filename": "Vikhr-7B-instruct-Q4_K_M.gguf",
            "description": "Vikhr 7B - русскоязычная модель от Vikhrmodels",
        },
        "qwen2.5-7b-instruct": {
            "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
            "description": "Qwen2.5 7B - отличная модель для русского языка",
        },
        "saiga-mistral-7b": {
            "repo": "IlyaGusev/saiga_mistral_7b_gguf",
            "filename": "model-q4_K.gguf",
            "description": "Saiga Mistral 7B - специально дообученная для русского",
        },
        "openchat-3.6-8b": {
            "repo": "openchat/openchat-3.6-8b-20240522-GGUF",
            "filename": "openchat-3.6-8b-20240522-q4_k_m.gguf",
            "description": "OpenChat 3.6 8B - универсальная модель",
        },
    },
    "embedding": {
        "multilingual-e5-large": {
            "name": "intfloat/multilingual-e5-large",
            "description": "Лучшая многоязычная embedding модель",
        },
        "multilingual-mpnet": {
            "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "description": "Быстрая многоязычная модель",
        },
        "bge-m3": {
            "name": "BAAI/bge-m3",
            "description": "BGE M3 - отличная многоязычная модель",
        },
    },
}


def auto_download_models(
    llm_model_key: str = "vikhr-7b-instruct",
    embedding_model_key: str = "multilingual-e5-large",
    models_dir: str = "/models",
    cache_dir: Optional[str] = None,
) -> tuple[Optional[str], bool]:
    """
    Автоматически скачивает рекомендованные модели

    Returns:
        (путь_к_llm_модели, успех_embedding_модели)
    """
    llm_path = None
    embedding_success = False

    # Скачиваем LLM модель
    if llm_model_key in RECOMMENDED_MODELS["llm"]:
        llm_config = RECOMMENDED_MODELS["llm"][llm_model_key]
        logger.info(f"📥 {llm_config['description']}")

        llm_path = download_llm_model_from_hf(
            model_repo=llm_config["repo"],
            filename=llm_config["filename"],
            local_dir=models_dir,
            cache_dir=cache_dir,
        )

    # Скачиваем embedding модель
    if embedding_model_key in RECOMMENDED_MODELS["embedding"]:
        embedding_config = RECOMMENDED_MODELS["embedding"][embedding_model_key]
        logger.info(f"📥 {embedding_config['description']}")

        embedding_success = download_embedding_model(
            model_name=embedding_config["name"], cache_dir=cache_dir
        )

    return llm_path, embedding_success
