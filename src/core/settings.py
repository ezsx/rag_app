"""
Конфигурация приложения — Pydantic BaseSettings.

Все параметры читаются из переменных окружения с поддержкой .env файлов.
Singleton через get_settings().
"""

import logging
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Настройки приложения. Singleton через get_settings()."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ── LLM — llama-server (V100 на Windows Host) ─────────────
    current_llm_key: str = Field("qwen3-30b-a3b", alias="LLM_MODEL_KEY")
    llm_base_url: str = Field(
        "http://host.docker.internal:8080", alias="LLM_BASE_URL"
    )
    llm_model_name: str = Field("qwen3-30b-a3b", alias="LLM_MODEL_NAME")
    llm_request_timeout: int = Field(120, alias="LLM_REQUEST_TIMEOUT")

    # Query Planner — отдельный endpoint (если задан)
    planner_llm_base_url: str = Field("", alias="PLANNER_LLM_BASE_URL")
    planner_llm_key: str = Field("qwen3-30b-a3b", alias="PLANNER_LLM_MODEL_KEY")

    # ── Embedding — gpu_server.py (WSL2, RTX 5060 Ti, порт 8082) ─
    current_embedding_key: str = Field(
        "qwen3-embedding-0.6b", alias="EMBEDDING_MODEL_KEY"
    )
    embedding_tei_url: str = Field(
        "http://host.docker.internal:8082", alias="EMBEDDING_TEI_URL"
    )
    embedding_query_instruction: str = Field(
        (
            "Instruct: Given a user question about ML, AI, LLM or tech news, "
            "retrieve relevant Telegram channel posts\n"
            "Query: "
        ),
        alias="EMBEDDING_QUERY_INSTRUCTION",
    )

    # ── Reranker — gpu_server.py (тот же порт 8082) ──────────
    reranker_tei_url: str = Field(
        "http://host.docker.internal:8082", alias="RERANKER_TEI_URL"
    )
    enable_reranker: bool = Field(True, alias="ENABLE_RERANKER")
    reranker_top_n: int = Field(80, alias="RERANKER_TOP_N")
    reranker_batch_size: int = Field(16, alias="RERANKER_BATCH_SIZE")

    # ── Qdrant (Docker CPU, порт 6333) ───────────────────────
    qdrant_url: str = Field("http://qdrant:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field("news_colbert", alias="QDRANT_COLLECTION")

    # ── PCA Whitening ────────────────────────────────────────
    whitening_params_path: str = Field("", alias="WHITENING_PARAMS_PATH")

    # ── Redis кеширование ────────────────────────────────────
    redis_enabled: bool = Field(False, alias="REDIS_ENABLED")
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_password: str | None = Field(None, alias="REDIS_PASSWORD")
    cache_ttl: int = Field(3600, alias="CACHE_TTL")

    # ── Query Planner / Fusion ───────────────────────────────
    enable_query_planner: bool = Field(True, alias="ENABLE_QUERY_PLANNER")
    fusion_strategy: str = Field("rrf", alias="FUSION_STRATEGY")
    k_fusion: int = Field(60, alias="K_FUSION")

    # MMR
    enable_mmr: bool = Field(True, alias="ENABLE_MMR")
    mmr_lambda: float = Field(0.7, alias="MMR_LAMBDA")
    mmr_top_n: int = Field(120, alias="MMR_TOP_N")
    mmr_output_k: int = Field(60, alias="MMR_OUTPUT_K")

    search_k_per_query_default: int = Field(10, alias="SEARCH_K_PER_QUERY_DEFAULT")
    max_plan_subqueries: int = Field(5, alias="MAX_PLAN_SUBQUERIES")

    # ── Hybrid Retriever ─────────────────────────────────────
    hybrid_enabled: bool = Field(True, alias="HYBRID_ENABLED")
    hybrid_top_dense: int = Field(100, alias="HYBRID_TOP_DENSE")
    hybrid_top_sparse: int = Field(100, alias="HYBRID_TOP_SPARSE")
    enforce_router_route: bool = Field(False, alias="ENFORCE_ROUTER_ROUTE")

    # ── Planner декодинг ─────────────────────────────────────
    use_gbnf_planner: bool = Field(True, alias="USE_GBNF_PLANNER")
    planner_timeout: float = Field(30.0, alias="PLANNER_TIMEOUT")
    planner_token_budget: int = Field(4096, alias="PLANNER_TOKEN_BUDGET")
    planner_temp: float = Field(0.2, alias="PLANNER_TEMP")
    planner_top_p: float = Field(0.9, alias="PLANNER_TOP_P")
    planner_top_k: int = Field(40, alias="PLANNER_TOP_K")
    planner_repeat_penalty: float = Field(1.1, alias="PLANNER_REPEAT_PENALTY")
    planner_stop: str = Field("Observation:", alias="PLANNER_STOP")

    # ── In-memory кеш ────────────────────────────────────────
    enable_cache: bool = Field(True, alias="ENABLE_CACHE")

    # ── ReAct Agent ──────────────────────────────────────────
    enable_agent: bool = Field(True, alias="ENABLE_AGENT")
    agent_max_steps: int = Field(15, alias="AGENT_MAX_STEPS")
    agent_default_steps: int = Field(8, alias="AGENT_DEFAULT_STEPS")
    agent_tool_timeout: float = Field(15.0, alias="AGENT_TOOL_TIMEOUT")
    agent_token_budget: int = Field(2000, alias="AGENT_TOKEN_BUDGET")

    # Agent tool-step декодинг
    agent_tool_temp: float = Field(0.7, alias="AGENT_TOOL_TEMP")
    agent_tool_top_p: float = Field(0.8, alias="AGENT_TOOL_TOP_P")
    agent_tool_top_k: int = Field(20, alias="AGENT_TOOL_TOP_K")
    agent_tool_presence_penalty: float = Field(1.5, alias="AGENT_TOOL_PRESENCE_PENALTY")
    agent_tool_repeat_penalty: float = Field(1.15, alias="AGENT_TOOL_REPEAT_PENALTY")
    agent_tool_max_tokens: int = Field(384, alias="AGENT_TOOL_MAX_TOKENS")

    # Agent final-answer декодинг
    agent_final_temp: float = Field(0.3, alias="AGENT_FINAL_TEMP")
    agent_final_top_p: float = Field(0.9, alias="AGENT_FINAL_TOP_P")
    agent_final_max_tokens: int = Field(1024, alias="AGENT_FINAL_MAX_TOKENS")

    # ── Coverage / Refinement (LANCER-style) ─────────────────
    coverage_threshold: float = Field(0.75, alias="COVERAGE_THRESHOLD")
    max_refinements: int = Field(1, alias="MAX_REFINEMENTS")
    enable_verify_step: bool = Field(True, alias="ENABLE_VERIFY_STEP")

    # ── Chunking для ingest ──────────────────────────────────
    chunk_char_threshold: int = Field(1500, alias="CHUNK_CHAR_THRESHOLD")
    chunk_target_size: int = Field(1200, alias="CHUNK_TARGET_SIZE")

    @property
    def current_collection(self) -> str:
        """Алиас для обратной совместимости."""
        return self.qdrant_collection

    @property
    def enable_hybrid_retriever(self) -> bool:
        """Алиас для обратной совместимости."""
        return self.hybrid_enabled

    @property
    def planner_stop_list(self) -> list[str]:
        """planner_stop как список (разделитель ||)."""
        return self.planner_stop.split("||")

    def model_post_init(self, __context) -> None:
        """Логирование после инициализации."""
        logger.info(
            "Настройки загружены: LLM=%s, Embedding=%s, Qdrant=%s/%s, "
            "EmbTEI=%s, RerankTEI=%s, Coverage=%.2f, MaxRefinements=%d",
            self.current_llm_key,
            self.current_embedding_key,
            self.qdrant_url,
            self.qdrant_collection,
            self.embedding_tei_url,
            self.reranker_tei_url,
            self.coverage_threshold,
            self.max_refinements,
        )

    def update_llm_model(self, model_key: str) -> None:
        """Горячая смена LLM модели. Сбрасывает lru_cache фабрик."""
        old_key = self.current_llm_key
        self.current_llm_key = model_key
        from core.deps import get_llm, get_qa_service

        try:
            from core.deps import get_agent_service

            get_agent_service.cache_clear()
        except ImportError:
            pass
        get_llm.cache_clear()
        get_qa_service.cache_clear()
        logger.info("LLM модель изменена: %s → %s", old_key, model_key)

    def update_embedding_model(self, model_key: str) -> None:
        """Горячая смена embedding модели. Сбрасывает lru_cache фабрик."""
        old_key = self.current_embedding_key
        self.current_embedding_key = model_key
        from core.deps import get_hybrid_retriever, get_qa_service, get_retriever

        try:
            from core.deps import get_agent_service

            get_agent_service.cache_clear()
        except ImportError:
            pass
        get_hybrid_retriever.cache_clear()
        get_retriever.cache_clear()
        get_qa_service.cache_clear()
        logger.info("Embedding модель изменена: %s → %s", old_key, model_key)

    def update_collection(self, collection_name: str) -> None:
        """Горячая смена Qdrant-коллекции. Сбрасывает lru_cache фабрик."""
        old = self.qdrant_collection
        self.qdrant_collection = collection_name
        from core.deps import (
            get_hybrid_retriever,
            get_qa_service,
            get_qdrant_store,
            get_retriever,
        )

        try:
            from core.deps import get_agent_service

            get_agent_service.cache_clear()
        except ImportError:
            pass
        get_hybrid_retriever.cache_clear()
        get_qdrant_store.cache_clear()
        get_retriever.cache_clear()
        get_qa_service.cache_clear()
        logger.info("Qdrant коллекция изменена: %s → %s", old, collection_name)


@lru_cache
def get_settings() -> Settings:
    """Singleton настроек приложения."""
    return Settings()
