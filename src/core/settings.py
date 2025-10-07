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
        self.current_llm_key: str = os.getenv("LLM_MODEL_KEY", "qwen2.5-7b-instruct")
        # Отдельная модель для Query Planner (CPU)
        self.planner_llm_key: str = os.getenv(
            "PLANNER_LLM_MODEL_KEY", "qwen2.5-3b-instruct"
        )
        self.planner_llm_device: str = os.getenv("PLANNER_LLM_DEVICE", "cpu")
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
        # MMR параметры
        self.enable_mmr: bool = os.getenv("ENABLE_MMR", "true").lower() == "true"
        try:
            self.mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))
        except Exception:
            self.mmr_lambda = 0.7
        self.mmr_top_n: int = int(os.getenv("MMR_TOP_N", "120"))
        self.mmr_output_k: int = int(os.getenv("MMR_OUTPUT_K", "60"))
        # Ререйкер (CPU)
        self.enable_reranker: bool = (
            os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        )
        self.reranker_model_key: str = os.getenv(
            "RERANKER_MODEL_KEY", "BAAI/bge-reranker-v2-m3"
        )
        self.reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "80"))
        self.reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
        self.search_k_per_query_default: int = int(
            os.getenv("SEARCH_K_PER_QUERY_DEFAULT", "10")
        )
        self.max_plan_subqueries: int = int(os.getenv("MAX_PLAN_SUBQUERIES", "5"))

        # Планировщик через GBNF (llama.cpp grammar) — можно быстро отключить
        self.use_gbnf_planner: bool = (
            os.getenv("USE_GBNF_PLANNER", "true").lower() == "true"
        )
        self.planner_timeout: float = float(os.getenv("PLANNER_TIMEOUT", "15.0"))
        self.planner_token_budget: int = int(os.getenv("PLANNER_TOKEN_BUDGET", "4096"))
        self.planner_temp: float = float(os.getenv("PLANNER_TEMP", "0.2"))
        self.planner_top_p: float = float(os.getenv("PLANNER_TOP_P", "0.9"))
        self.planner_top_k: int = int(os.getenv("PLANNER_TOP_K", "40"))
        self.planner_repeat_penalty: float = float(
            os.getenv("PLANNER_REPEAT_PENALTY", "1.1")
        )
        self.planner_stop: List[str] = os.getenv("PLANNER_STOP", "Observation:").split(
            "||"
        )

        # Встроенный in-memory кеш (TTL), отдельный от Redis
        self.enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"

        # === ReAct Agent настройки ===
        self.enable_agent: bool = os.getenv("ENABLE_AGENT", "true").lower() == "true"
        self.agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "15"))
        self.agent_default_steps: int = int(os.getenv("AGENT_DEFAULT_STEPS", "8"))
        self.agent_tool_timeout: float = float(os.getenv("AGENT_TOOL_TIMEOUT", "15.0"))
        self.agent_token_budget: int = int(os.getenv("AGENT_TOKEN_BUDGET", "2000"))

        # === Оптимизированные параметры декодинга для ReAct ===
        # Шаги инструментов (короткие, детерминированные)
        self.agent_tool_temp: float = float(os.getenv("AGENT_TOOL_TEMP", "0.2"))
        self.agent_tool_top_p: float = float(os.getenv("AGENT_TOOL_TOP_P", "0.9"))
        self.agent_tool_top_k: int = int(os.getenv("AGENT_TOOL_TOP_K", "40"))
        self.agent_tool_repeat_penalty: float = float(
            os.getenv("AGENT_TOOL_REPEAT_PENALTY", "1.15")
        )
        self.agent_tool_max_tokens: int = int(os.getenv("AGENT_TOOL_MAX_TOKENS", "64"))

        # Финальные ответы (более креативные)
        self.agent_final_temp: float = float(os.getenv("AGENT_FINAL_TEMP", "0.3"))
        self.agent_final_top_p: float = float(os.getenv("AGENT_FINAL_TOP_P", "0.9"))
        self.agent_final_max_tokens: int = int(
            os.getenv("AGENT_FINAL_MAX_TOKENS", "512")
        )

        # === Enhanced ReAct Agent настройки ===
        self.coverage_threshold: float = float(os.getenv("COVERAGE_THRESHOLD", "0.8"))
        self.max_refinements: int = int(os.getenv("MAX_REFINEMENTS", "1"))
        self.enable_verify_step: bool = (
            os.getenv("ENABLE_VERIFY_STEP", "true").lower() == "true"
        )

        # === BM25 / Hybrid настройки ===
        self.bm25_index_root: str = os.getenv("BM25_INDEX_ROOT", "./bm25-index")
        self.hybrid_enabled: bool = (
            os.getenv("HYBRID_ENABLED", "true").lower() == "true"
        )
        self.hybrid_top_bm25: int = int(os.getenv("HYBRID_TOP_BM25", "100"))
        self.hybrid_top_dense: int = int(os.getenv("HYBRID_TOP_DENSE", "100"))
        self.bm25_default_top_k: int = int(os.getenv("BM25_DEFAULT_TOP_K", "100"))
        self.bm25_reload_min_interval_sec: int = int(
            os.getenv("BM25_RELOAD_MIN_INTERVAL_SEC", "5")
        )
        # Алиас для совместимости со старыми настройками
        self.enable_hybrid_retriever: bool = self.hybrid_enabled
        self.enforce_router_route: bool = (
            os.getenv("ENFORCE_ROUTER_ROUTE", "false").lower() == "true"
        )

        logger.info(
            f"Настройки загружены: LLM={self.current_llm_key}, "
            f"Embedding={self.current_embedding_key}, Collection={self.current_collection}, "
            f"Agent={self.enable_agent}, Hybrid={self.hybrid_enabled}, "
            f"Coverage={self.coverage_threshold}, MaxRefinements={self.max_refinements}, Verify={self.enable_verify_step}"
        )

    def update_llm_model(self, model_key: str) -> None:
        """Обновляет текущую LLM модель и сбрасывает кеш"""
        old_key = self.current_llm_key
        self.current_llm_key = model_key

        # Сбрасываем кеш зависимостей для горячей перезагрузки
        from core.deps import get_llm, get_qa_service

        try:
            from core.deps import get_agent_service

            get_agent_service.cache_clear()
        except ImportError:
            pass  # Агент сервис может быть еще не зарегистрирован

        get_llm.cache_clear()
        get_qa_service.cache_clear()

        logger.info(f"LLM модель изменена: {old_key} → {model_key}")

    def update_embedding_model(self, model_key: str) -> None:
        """Обновляет текущую embedding модель и сбрасывает кеш"""
        old_key = self.current_embedding_key
        self.current_embedding_key = model_key

        # Сбрасываем кеш зависимостей
        from core.deps import get_retriever, get_qa_service

        try:
            from core.deps import get_agent_service

            get_agent_service.cache_clear()
        except ImportError:
            pass  # Агент сервис может быть еще не зарегистрирован

        get_retriever.cache_clear()
        get_qa_service.cache_clear()

        logger.info(f"Embedding модель изменена: {old_key} → {model_key}")

    def update_collection(self, collection_name: str) -> None:
        """Обновляет текущую коллекцию ChromaDB"""
        old_collection = self.current_collection
        self.current_collection = collection_name

        # Сбрасываем кеш retriever и agent service
        from core.deps import get_retriever, get_qa_service

        try:
            from core.deps import get_agent_service

            get_agent_service.cache_clear()
        except ImportError:
            pass  # Агент сервис может быть еще не зарегистрирован

        get_retriever.cache_clear()
        get_qa_service.cache_clear()

        logger.info(f"Коллекция изменена: {old_collection} → {collection_name}")


@lru_cache()
def get_settings() -> Settings:
    """Singleton для настроек приложения"""
    return Settings()
