"""
Dependency Injection — Phase 1.

Фабрики всех сервисов и клиентов.
Все фабрики используют @lru_cache — синглтоны на процесс.
Смена настроек требует явного cache_clear() через settings.update_*().
"""

import logging
import os
from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding

from adapters.llm.llama_server_client import LlamaServerClient
from adapters.qdrant.store import QdrantStore
from adapters.search.hybrid_retriever import HybridRetriever
from adapters.tei.embedding_client import TEIEmbeddingClient
from adapters.tei.reranker_client import TEIRerankerClient
from core.settings import get_settings
from services.agent_service import AgentService
from services.qa_service import QAService
from services.query_planner_service import QueryPlannerService
from services.reranker_service import RerankerService

logger = logging.getLogger(__name__)


@lru_cache
def get_llm() -> LlamaServerClient:
    """HTTP-клиент к llama-server."""
    settings = get_settings()
    logger.info(
        "LLM: llama-server → %s (model=%s)",
        settings.llm_base_url,
        settings.llm_model_name,
    )
    return LlamaServerClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model_name,
        timeout=settings.llm_request_timeout,
    )


@lru_cache
def get_planner_llm() -> LlamaServerClient:
    """HTTP-клиент для QueryPlanner."""
    settings = get_settings()
    base_url = settings.planner_llm_base_url or settings.llm_base_url
    logger.info("Planner LLM: llama-server → %s", base_url)
    return LlamaServerClient(
        base_url=base_url,
        model=settings.llm_model_name,
        timeout=settings.llm_request_timeout,
    )


@lru_cache
def get_query_planner() -> QueryPlannerService:
    settings = get_settings()
    try:
        planner_llm = get_planner_llm()
    except Exception:  # broad: lazy init safety
        planner_llm = get_llm()
    return QueryPlannerService(planner_llm, settings)


@lru_cache
def get_tei_embedding_client() -> TEIEmbeddingClient:
    """Singleton TEIEmbeddingClient."""
    settings = get_settings()
    logger.info("TEI embedding client: %s", settings.embedding_tei_url)
    return TEIEmbeddingClient(
        base_url=settings.embedding_tei_url,
        query_instruction=settings.embedding_query_instruction,
        whitening_params_path=settings.whitening_params_path,
    )


@lru_cache
def get_tei_reranker_client() -> TEIRerankerClient:
    """Singleton TEIRerankerClient."""
    settings = get_settings()
    logger.info("TEI reranker client: %s", settings.reranker_tei_url)
    return TEIRerankerClient(base_url=settings.reranker_tei_url)


@lru_cache
def get_qdrant_store() -> QdrantStore:
    """Singleton QdrantStore."""
    settings = get_settings()
    return QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)


@lru_cache
def get_sparse_encoder() -> SparseTextEmbedding:
    """Singleton BM25 sparse encoder (fastembed, CPU)."""
    logger.info("Инициализация sparse encoder: Qdrant/bm25 (language=russian)")
    return SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")


@lru_cache
def get_hybrid_retriever() -> HybridRetriever | None:
    """Singleton HybridRetriever."""
    settings = get_settings()
    if not settings.hybrid_enabled:
        logger.info("HybridRetriever отключён (hybrid_enabled=False)")
        return None
    try:
        return HybridRetriever(
            store=get_qdrant_store(),
            embedding_client=get_tei_embedding_client(),
            sparse_encoder=get_sparse_encoder(),
            settings=settings,
        )
    except Exception as exc:  # broad: lazy init safety
        logger.error("HybridRetriever init failed: %s", exc)
        return None


@lru_cache
def get_retriever() -> HybridRetriever | None:
    """Backward-compatible алиас для кода, ожидающего get_retriever()."""
    return get_hybrid_retriever()


@lru_cache
def get_reranker() -> RerankerService | None:
    """Singleton sync-обёртки над async TEI reranker client."""
    settings = get_settings()
    if not settings.enable_reranker:
        return None
    return RerankerService(get_tei_reranker_client())


@lru_cache
def get_qa_service() -> QAService:
    settings = get_settings()
    hybrid_retriever = get_hybrid_retriever()

    def _llm_factory():
        return get_llm()

    top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    planner = get_query_planner() if settings.enable_query_planner else None
    reranker = get_reranker() if settings.enable_reranker else None

    return QAService(
        hybrid_retriever,
        _llm_factory,
        top_k,
        settings=settings,
        planner=planner,
        reranker=reranker,
        hybrid=hybrid_retriever,
    )


@lru_cache
def get_redis_client() -> Any | None:
    """Redis-клиент если кеширование включено."""
    settings = get_settings()
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
        client.ping()
        logger.info("Redis подключён: %s:%s", settings.redis_host, settings.redis_port)
        return client
    except Exception as exc:  # broad: lazy init safety
        logger.warning("Redis недоступен: %s", exc)
        return None


@lru_cache
def get_agent_service() -> AgentService:
    """Singleton AgentService с полным набором инструментов."""
    from services.tools.registry import build_tool_runner

    settings = get_settings()
    tool_runner = build_tool_runner(
        settings=settings,
        hybrid_retriever=get_hybrid_retriever(),
        qdrant_store=get_qdrant_store(),
        reranker=get_reranker(),
        query_planner=get_query_planner() if settings.enable_query_planner else None,
    )
    return AgentService(
        llm_factory=get_llm,
        tool_runner=tool_runner,
        settings=settings,
    )
