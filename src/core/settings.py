"""
Конфигурация приложения с поддержкой горячего переключения моделей
"""

import os
from typing import Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class Settings:
    """Настройки приложения с поддержкой горячего переключения"""

    def __init__(self):
        self.current_llm_key: str = os.getenv("LLM_MODEL_KEY", "gpt-oss-20b")
        self.current_embedding_key: str = os.getenv(
            "EMBEDDING_MODEL_KEY", "multilingual-e5-large"
        )
        self.current_collection: str = os.getenv("CHROMA_COLLECTION", "news_demo4")

        # Redis кеширование
        self.redis_enabled: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
        self.redis_host: str = os.getenv("REDIS_HOST", "localhost")
        self.redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
        self.cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 час

        # ChromaDB настройки
        self.chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
        self.chroma_path: str = os.getenv("CHROMA_PATH", "/data/chroma")

        # Модели директории
        self.models_dir: str = os.getenv("MODELS_DIR", "/models")
        self.cache_dir: str = os.getenv("TRANSFORMERS_CACHE", "/models/.cache")

        # === Query Planner / Fusion настройки ===
        self.enable_query_planner: bool = (
            os.getenv("ENABLE_QUERY_PLANNER", "true").lower() == "true"
        )
        self.fusion_strategy: str = os.getenv("FUSION_STRATEGY", "rrf").lower()
        self.k_fusion: int = int(os.getenv("K_FUSION", "60"))
        self.enable_reranker: bool = (
            os.getenv("ENABLE_RERANKER", "false").lower() == "true"
        )
        self.search_k_per_query_default: int = int(
            os.getenv("SEARCH_K_PER_QUERY_DEFAULT", "10")
        )
        self.max_plan_subqueries: int = int(os.getenv("MAX_PLAN_SUBQUERIES", "5"))

        # Встроенный in-memory кеш (TTL), отдельный от Redis
        self.enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"

        logger.info(
            f"Настройки загружены: LLM={self.current_llm_key}, "
            f"Embedding={self.current_embedding_key}, Collection={self.current_collection}"
        )

    def update_llm_model(self, model_key: str) -> None:
        """Обновляет текущую LLM модель и сбрасывает кеш"""
        old_key = self.current_llm_key
        self.current_llm_key = model_key

        # Сбрасываем кеш зависимостей для горячей перезагрузки
        from core.deps import get_llm, get_qa_service

        get_llm.cache_clear()
        get_qa_service.cache_clear()

        logger.info(f"LLM модель изменена: {old_key} → {model_key}")

    def update_embedding_model(self, model_key: str) -> None:
        """Обновляет текущую embedding модель и сбрасывает кеш"""
        old_key = self.current_embedding_key
        self.current_embedding_key = model_key

        # Сбрасываем кеш зависимостей
        from core.deps import get_retriever, get_qa_service

        get_retriever.cache_clear()
        get_qa_service.cache_clear()

        logger.info(f"Embedding модель изменена: {old_key} → {model_key}")

    def update_collection(self, collection_name: str) -> None:
        """Обновляет текущую коллекцию ChromaDB"""
        old_collection = self.current_collection
        self.current_collection = collection_name

        # Сбрасываем кеш retriever
        from core.deps import get_retriever, get_qa_service

        get_retriever.cache_clear()
        get_qa_service.cache_clear()

        logger.info(f"Коллекция изменена: {old_collection} → {collection_name}")


@lru_cache()
def get_settings() -> Settings:
    """Singleton для настроек приложения"""
    return Settings()
