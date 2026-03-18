"""
Конфигурация приложения — Phase 1 (Qdrant + TEI HTTP).

Изменения по сравнению с Phase 0:
- Удалены: ChromaDB поля, BM25 поля, local-model пути
- Добавлены: qdrant_url, qdrant_collection, embedding_tei_url, reranker_tei_url
- Исправлены: coverage_threshold=0.65, max_refinements=2, LLM=qwen3-30b-a3b
- Текущий embedding: Qwen3-Embedding-0.6B + instruction prefix
- Добавлены настройки chunking для ingest
"""

import os
from typing import List, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class Settings:
    """Настройки приложения Phase 1. Singleton через get_settings()."""

    def __init__(self):
        # === LLM — llama-server (V100 на Windows Host) ===
        # llama-server.exe запускается на хосте, Docker обращается через host.docker.internal.
        # V100 TCC недоступен в WSL2/Docker — только через HTTP на хосте.
        self.current_llm_key: str = os.getenv("LLM_MODEL_KEY", "qwen3-30b-a3b")
        self.llm_base_url: str = os.getenv(
            "LLM_BASE_URL", "http://host.docker.internal:8080"
        )
        self.llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen3-30b-a3b")
        self.llm_request_timeout: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

        # Query Planner может использовать отдельный endpoint.
        # Если PLANNER_LLM_BASE_URL не задан — используется тот же llama-server.
        self.planner_llm_base_url: str = os.getenv("PLANNER_LLM_BASE_URL", "")
        self.planner_llm_key: str = os.getenv(
            "PLANNER_LLM_MODEL_KEY", "qwen3-30b-a3b"
        )

        # === Embedding — TEI HTTP (WSL2 native, RTX 5060 Ti, порт 8082) ===
        # Модель: Qwen/Qwen3-Embedding-0.6B (1024-dim, cosine, long context).
        # TEI запускается отдельно в WSL2, не в Docker (DEC-0024).
        self.current_embedding_key: str = os.getenv(
            "EMBEDDING_MODEL_KEY", "qwen3-embedding-0.6b"
        )
        self.embedding_tei_url: str = os.getenv(
            "EMBEDDING_TEI_URL", "http://host.docker.internal:8082"
        )
        self.embedding_query_instruction: str = os.getenv(
            "EMBEDDING_QUERY_INSTRUCTION",
            (
                "Instruct: Given a user question about ML, AI, LLM or tech news, "
                "retrieve relevant Telegram channel posts\n"
                "Query: "
            ),
        )

        # === Reranker — gpu_server.py (WSL2 native, RTX 5060 Ti, порт 8082) ===
        # Embedding и Reranker обслуживаются одним процессом на порту 8082.
        self.reranker_tei_url: str = os.getenv(
            "RERANKER_TEI_URL", "http://host.docker.internal:8082"
        )
        self.enable_reranker: bool = (
            os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        )
        self.reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "80"))
        self.reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))

        # === Qdrant (Docker CPU, порт 6333) ===
        self.qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "news")
        # Алиас для обратной совместимости с кодом, обращающимся к current_collection
        self.current_collection: str = self.qdrant_collection

        # === Redis кеширование (отключён по умолчанию) ===
        self.redis_enabled: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
        self.redis_host: str = os.getenv("REDIS_HOST", "localhost")
        self.redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
        self.cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))

        # === Query Planner / Fusion ===
        self.enable_query_planner: bool = (
            os.getenv("ENABLE_QUERY_PLANNER", "true").lower() == "true"
        )
        self.fusion_strategy: str = os.getenv("FUSION_STRATEGY", "rrf").lower()
        self.k_fusion: int = int(os.getenv("K_FUSION", "60"))

        # MMR — нативно через Qdrant MmrQuery
        self.enable_mmr: bool = os.getenv("ENABLE_MMR", "true").lower() == "true"
        try:
            self.mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))
        except Exception:
            self.mmr_lambda = 0.7
        self.mmr_top_n: int = int(os.getenv("MMR_TOP_N", "120"))
        self.mmr_output_k: int = int(os.getenv("MMR_OUTPUT_K", "60"))

        self.search_k_per_query_default: int = int(
            os.getenv("SEARCH_K_PER_QUERY_DEFAULT", "10")
        )
        self.max_plan_subqueries: int = int(os.getenv("MAX_PLAN_SUBQUERIES", "5"))

        # === Hybrid Retriever ===
        self.hybrid_enabled: bool = (
            os.getenv("HYBRID_ENABLED", "true").lower() == "true"
        )
        # Лимиты prefetch для dense и sparse в Qdrant prefetch запросе
        self.hybrid_top_dense: int = int(os.getenv("HYBRID_TOP_DENSE", "100"))
        self.hybrid_top_sparse: int = int(os.getenv("HYBRID_TOP_SPARSE", "100"))
        self.enforce_router_route: bool = (
            os.getenv("ENFORCE_ROUTER_ROUTE", "false").lower() == "true"
        )
        # Алиас для совместимости
        self.enable_hybrid_retriever: bool = self.hybrid_enabled

        # === Planner параметры декодинга ===
        self.use_gbnf_planner: bool = (
            os.getenv("USE_GBNF_PLANNER", "true").lower() == "true"
        )
        self.planner_timeout: float = float(os.getenv("PLANNER_TIMEOUT", "30.0"))
        self.planner_token_budget: int = int(os.getenv("PLANNER_TOKEN_BUDGET", "4096"))
        self.planner_temp: float = float(os.getenv("PLANNER_TEMP", "0.2"))
        self.planner_top_p: float = float(os.getenv("PLANNER_TOP_P", "0.9"))
        self.planner_top_k: int = int(os.getenv("PLANNER_TOP_K", "40"))
        self.planner_repeat_penalty: float = float(
            os.getenv("PLANNER_REPEAT_PENALTY", "1.1")
        )
        self.planner_stop: List[str] = os.getenv(
            "PLANNER_STOP", "Observation:"
        ).split("||")

        # === In-memory кеш (TTL) ===
        self.enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"

        # === ReAct Agent ===
        self.enable_agent: bool = os.getenv("ENABLE_AGENT", "true").lower() == "true"
        self.agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "15"))
        self.agent_default_steps: int = int(os.getenv("AGENT_DEFAULT_STEPS", "8"))
        self.agent_tool_timeout: float = float(os.getenv("AGENT_TOOL_TIMEOUT", "15.0"))
        self.agent_token_budget: int = int(os.getenv("AGENT_TOKEN_BUDGET", "2000"))

        # Параметры декодинга для tool-шагов (короткие, детерминированные)
        self.agent_tool_temp: float = float(os.getenv("AGENT_TOOL_TEMP", "0.7"))
        self.agent_tool_top_p: float = float(os.getenv("AGENT_TOOL_TOP_P", "0.8"))
        self.agent_tool_top_k: int = int(os.getenv("AGENT_TOOL_TOP_K", "20"))
        self.agent_tool_presence_penalty: float = float(
            os.getenv("AGENT_TOOL_PRESENCE_PENALTY", "1.5")
        )
        self.agent_tool_repeat_penalty: float = float(
            os.getenv("AGENT_TOOL_REPEAT_PENALTY", "1.15")
        )
        self.agent_tool_max_tokens: int = int(os.getenv("AGENT_TOOL_MAX_TOKENS", "384"))

        # Параметры декодинга для финального ответа
        self.agent_final_temp: float = float(os.getenv("AGENT_FINAL_TEMP", "0.3"))
        self.agent_final_top_p: float = float(os.getenv("AGENT_FINAL_TOP_P", "0.9"))
        self.agent_final_max_tokens: int = int(
            os.getenv("AGENT_FINAL_MAX_TOKENS", "1024")
        )

        # === Coverage / Refinement (DEC-0018, DEC-0019) ===
        # 0.65 — откалиброван под composite 5-signal metric (R04).
        # 0.8 был слишком агрессивен: вызывал false-negative refinements.
        self.coverage_threshold: float = float(
            os.getenv("COVERAGE_THRESHOLD", "0.65")
        )
        # 2 refinements дают +12% recall без существенного роста latency (R04).
        self.max_refinements: int = int(os.getenv("MAX_REFINEMENTS", "2"))
        self.enable_verify_step: bool = (
            os.getenv("ENABLE_VERIFY_STEP", "true").lower() == "true"
        )

        # === Chunking для ingest ===
        self.chunk_char_threshold: int = int(
            os.getenv("CHUNK_CHAR_THRESHOLD", "1500")
        )
        self.chunk_target_size: int = int(os.getenv("CHUNK_TARGET_SIZE", "1200"))

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
        self.current_collection = collection_name  # синхронизируем алиас
        from core.deps import (
            get_hybrid_retriever,
            get_qdrant_store,
            get_qa_service,
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


@lru_cache()
def get_settings() -> Settings:
    """Singleton настроек приложения."""
    return Settings()
