"""
Утилиты для автоматического скачивания моделей
"""

import os
import logging
import requests
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from huggingface_hub.utils import logging as hf_logging

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


# ---- LLM: точечное скачивание, без snapshot ----

_PREFERRED_ORDER = [
    "q8_0",  # явно хотим 8-bit
    "q6_k",
    "q5_k_m",
    "q5_k_s",
    "q4_k_m",
    "q4_k_s",
    "q4_0",
]


def _list_repo_gguf(repo_id: str) -> List[str]:
    api = HfApi()
    files = [
        f.rfilename
        for f in api.list_files_info(repo_id, expand=True)
        if f.rfilename.lower().endswith(".gguf")
    ]
    # возвращаем только имена файлов (без путей)
    return [Path(f).name for f in files]


def _pick_best_filename(available: List[str], hint_filename: str) -> Optional[str]:
    if not available:
        return None
    avail_lower = [s.lower() for s in available]
    hint_lower = hint_filename.lower()

    # 1) точное совпадение по имени
    for i, s in enumerate(avail_lower):
        if s == hint_lower:
            return available[i]

    # 2) приоритет по PREFERRED_ORDER с учётом хинта
    def rank(fname: str) -> tuple[int, int, int]:
        s = fname.lower()
        # содержит ли ключевую подсказку (q8/q6/q5/q4) из исходного имени
        hint_score = (
            1
            if any(k in hint_lower and k in s for k in ["q8_0", "q6_k", "q5", "q4"])
            else 0
        )
        # индекс предпочтения (ниже — лучше)
        pref_idx = next((i for i, k in enumerate(_PREFERRED_ORDER) if k in s), 999)
        # детерминизация
        return (hint_score, -int(1e6 - pref_idx), -len(s))

    best = sorted(available, key=rank, reverse=True)[0]
    return best


def download_llm_model_from_hf(
    model_repo: str, filename: str, local_dir: str, cache_dir: Optional[str] = None
) -> Optional[str]:
    """
    Скачивает один GGUF из Hugging Face Hub без snapshot.
    Сначала пытается точное имя, затем — подбирает лучший доступный вариант.
    """
    # хотим детальный лог и прогресс от HF
    hf_logging.set_verbosity_info()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    os.makedirs(local_dir, exist_ok=True)

    def _download(fname: str) -> Optional[str]:
        logger.info(f"🔄 Скачивание LLM модели {model_repo}/{fname}...")
        return hf_hub_download(
            repo_id=model_repo,
            filename=fname,
            local_dir=local_dir,
            cache_dir=cache_dir,
            resume_download=True,
        )

    # 1) пробуем точное имя
    try:
        path = _download(filename)
        logger.info(f"✅ LLM модель скачана: {path}")
        return path
    except Exception as e:
        logger.warning(f"⚠️ Не удалось скачать '{filename}' из {model_repo}: {e}")

    # 2) без snapshot — список файлов и подбор
    try:
        files = _list_repo_gguf(model_repo)
        if not files:
            logger.error("❌ В репозитории нет *.gguf")
            return None

        picked = _pick_best_filename(files, filename)
        if not picked:
            logger.error("❌ Не удалось подобрать GGUF")
            return None

        logger.info(f"➡️ Выбрали доступный GGUF: {picked}")
        path = _download(picked)
        logger.info(f"✅ LLM модель скачана: {path}")
        return path

    except Exception as e2:
        logger.error(f"❌ Ошибка подбора/скачивания GGUF: {e2}")
        return None


# ---- Embedding / Reranker (оставляем snapshot, это норм) ----


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
        if cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["HF_HOME"] = cache_dir

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
        "gpt-oss-20b": {
            "repo": "unsloth/gpt-oss-20b-GGUF",
            "filename": "gpt-oss-20b-Q6_K.gguf",
            "description": "OpenAI gpt-oss-20b (GGUF, Q6_K) от Unsloth",
        },
        "vikhr-7b-instruct": {
            "repo": "oblivious/Vikhr-7B-instruct-GGUF",
            "filename": "Vikhr-7B-instruct-Q4_K_M.gguf",
            "description": "Vikhr 7B - русскоязычная модель от Vikhrmodels",
        },
        # Переходим на репозиторий bartowski и фиксируем Q8_0
        "qwen2.5-7b-instruct": {
            "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "filename": "Qwen2.5-7B-Instruct-Q8_0.gguf",
            "description": "Qwen2.5 7B Instruct (GGUF, Q8_0) — быстро заменить без snapshot",
        },
        "qwen2.5-3b-instruct": {
            "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "description": "Qwen2.5 3B Instruct (GGUF, Q4_K_M)",
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
    "reranker": {
        # Ключи можно расширить при необходимости; по умолчанию используем полный repo id
        "bge-reranker-v2-m3": {
            "name": "BAAI/bge-reranker-v2-m3",
            "description": "BAAI bge-reranker-v2-m3 (CrossEncoder, CPU)",
        }
    },
}


def auto_download_models(
    llm_model_key: str = "gpt-oss-20b",
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


def download_reranker_model(model_name: str, cache_dir: Optional[str] = None) -> bool:
    """
    Скачивает CrossEncoder ререйкер из Hugging Face, чтобы избежать загрузки на первом запросе.

    Args:
        model_name: repo id, например "BAAI/bge-reranker-v2-m3"
        cache_dir: Директория кэша HF

    Returns:
        True если скачивание успешно
    """
    try:
        logger.info(f"🔄 Скачивание reranker модели {model_name}...")
        # Кладем в кэш HF (TRANSFORMERS_CACHE/HF_HOME читаются из окружения на уровне контейнера)
        snapshot_download(
            repo_id=model_name, cache_dir=cache_dir, local_files_only=False
        )
        logger.info(f"✅ Reranker модель скачана: {model_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка скачивания reranker модели: {e}")
        return False
