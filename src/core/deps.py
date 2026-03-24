"""
Dependency Injection — Phase 1.

Фабрики всех сервисов и клиентов.
Все фабрики используют @lru_cache — синглтоны на процесс.
Смена настроек требует явного cache_clear() через settings.update_*().
"""

import logging
import os
from functools import lru_cache
from typing import Optional

from adapters.llm.llama_server_client import LlamaServerClient
from adapters.qdrant.store import QdrantStore
from adapters.search.hybrid_retriever import HybridRetriever
from adapters.tei.embedding_client import TEIEmbeddingClient
from adapters.tei.reranker_client import TEIRerankerClient
from core.settings import Settings, get_settings
from fastembed import SparseTextEmbedding
from services.agent_service import AgentService
from services.qa_service import QAService
from services.query_planner_service import QueryPlannerService
from services.reranker_service import RerankerService
from services.tools.tool_runner import ToolRunner

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
    except Exception:
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
def get_hybrid_retriever() -> Optional[HybridRetriever]:
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
    except Exception as exc:
        logger.error("HybridRetriever init failed: %s", exc)
        return None


@lru_cache
def get_retriever() -> Optional[HybridRetriever]:
    """Backward-compatible алиас для кода, ожидающего get_retriever()."""
    return get_hybrid_retriever()


@lru_cache
def get_reranker() -> Optional[RerankerService]:
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
def get_redis_client():
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
    except Exception as exc:
        logger.warning("Redis недоступен: %s", exc)
        return None


@lru_cache
def get_agent_service() -> AgentService:
    """Singleton AgentService с полным набором инструментов Phase 1."""
    settings = get_settings()

    tool_runner = ToolRunner(default_timeout_sec=settings.agent_tool_timeout)

    from services.tools.compose_context import compose_context
    from services.tools.fetch_docs import fetch_docs
    from services.tools.final_answer import final_answer
    from services.tools.query_plan import query_plan
    from services.tools.rerank import rerank
    from services.tools.router_select import router_select
    from services.tools.search import search
    from services.tools.verify import verify
    from services.tools.list_channels import list_channels
    from services.tools.related_posts import related_posts
    from services.tools.cross_channel_compare import cross_channel_compare
    from services.tools.summarize_channel import summarize_channel

    qdrant_store = get_qdrant_store()
    hybrid_retriever = get_hybrid_retriever()
    reranker = get_reranker()
    query_planner = get_query_planner() if settings.enable_query_planner else None

    def search_wrapper(**kwargs):
        return search(hybrid_retriever=hybrid_retriever, **kwargs)

    def verify_wrapper(**kwargs):
        return verify(hybrid_retriever=hybrid_retriever, **kwargs)

    def fetch_docs_wrapper(**kwargs):
        return fetch_docs(qdrant_store=qdrant_store, **kwargs)

    def query_plan_wrapper(**kwargs):
        return query_plan(query_planner=query_planner, **kwargs)

    def rerank_wrapper(**kwargs):
        return rerank(reranker=reranker, **kwargs)

    tool_runner.register("router_select", router_select)
    tool_runner.register(
        "query_plan", query_plan_wrapper, timeout_sec=settings.planner_timeout
    )
    tool_runner.register(
        "search",
        search_wrapper,
        timeout_sec=settings.agent_tool_timeout,
    )
    tool_runner.register(
        "rerank",
        rerank_wrapper,
        timeout_sec=settings.agent_tool_timeout,
    )
    tool_runner.register("fetch_docs", fetch_docs_wrapper)
    tool_runner.register("compose_context", compose_context)
    tool_runner.register("verify", verify_wrapper)
    tool_runner.register("final_answer", final_answer)

    # SPEC-RAG-13: новые tools — все через hybrid_retriever sync bridge
    def list_channels_wrapper(**kwargs):
        return list_channels(hybrid_retriever=hybrid_retriever, **kwargs)

    def related_posts_wrapper(**kwargs):
        return related_posts(hybrid_retriever=hybrid_retriever, **kwargs)

    def cross_channel_compare_wrapper(**kwargs):
        return cross_channel_compare(hybrid_retriever=hybrid_retriever, **kwargs)

    def summarize_channel_wrapper(**kwargs):
        return summarize_channel(hybrid_retriever=hybrid_retriever, **kwargs)

    tool_runner.register("list_channels", list_channels_wrapper)
    tool_runner.register("related_posts", related_posts_wrapper)
    tool_runner.register(
        "cross_channel_compare", cross_channel_compare_wrapper,
        timeout_sec=settings.agent_tool_timeout,
    )
    tool_runner.register(
        "summarize_channel", summarize_channel_wrapper,
        timeout_sec=settings.agent_tool_timeout,
    )

    def _llm_factory():
        return get_llm()

    qa_service = get_qa_service()

    return AgentService(
        llm_factory=_llm_factory,
        tool_runner=tool_runner,
        settings=settings,
        qa_service=qa_service,
    )
