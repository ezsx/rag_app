"""
Tool registry — регистрация всех LLM tools в ToolRunner.

Извлечено из deps.py (SPEC-RAG-27 Phase 3).
Каждый tool получает свои зависимости через partial-подобные wrappers.

NOTE: temporal_search и channel_search — virtual tools.
LLM видит их в schema (prompts.py), executor маппит на search (executor.py:66).
Не регистрируются отдельно в ToolRunner.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from core.settings import Settings
from services.tools.tool_runner import ToolRunner


def build_tool_runner(
    settings: Settings,
    hybrid_retriever: Any,
    qdrant_store: Any,
    reranker: Any,
    query_planner: Any,
) -> ToolRunner:
    """Создаёт ToolRunner со всеми зарегистрированными tools.

    Args:
        settings: application settings
        hybrid_retriever: HybridRetriever instance (или None)
        qdrant_store: QdrantStore instance
        reranker: RerankerService instance (или None)
        query_planner: QueryPlannerService instance (или None)
    """
    from services.tools.arxiv_tracker import arxiv_tracker
    from services.tools.channel_expertise import channel_expertise
    from services.tools.compose_context import compose_context
    from services.tools.cross_channel_compare import cross_channel_compare
    from services.tools.entity_tracker import entity_tracker
    from services.tools.fetch_docs import fetch_docs
    from services.tools.final_answer import final_answer
    from services.tools.hot_topics import hot_topics
    from services.tools.list_channels import list_channels
    from services.tools.query_plan import query_plan
    from services.tools.related_posts import related_posts
    from services.tools.rerank import rerank
    from services.tools.search import search
    from services.tools.summarize_channel import summarize_channel
    from services.tools.verify import evidence_support_check, verify

    runner = ToolRunner(default_timeout_sec=settings.agent_tool_timeout)

    # ── Core tools ──
    runner.register(
        "query_plan",
        partial(query_plan, query_planner=query_planner),
        timeout_sec=settings.planner_timeout,
    )
    runner.register(
        "search",
        partial(search, hybrid_retriever=hybrid_retriever),
        timeout_sec=settings.agent_tool_timeout,
    )
    runner.register(
        "rerank",
        partial(rerank, reranker=reranker),
        timeout_sec=settings.agent_tool_timeout,
    )
    runner.register("fetch_docs", partial(fetch_docs, qdrant_store=qdrant_store))
    runner.register("compose_context", compose_context)
    runner.register(
        "evidence_support_check",
        partial(evidence_support_check, hybrid_retriever=hybrid_retriever),
    )
    runner.register("verify", partial(verify, hybrid_retriever=hybrid_retriever))
    runner.register("final_answer", final_answer)

    # ── SPEC-RAG-13: navigation + comparison tools ──
    runner.register("list_channels", partial(list_channels, hybrid_retriever=hybrid_retriever))
    runner.register("related_posts", partial(related_posts, hybrid_retriever=hybrid_retriever))
    runner.register(
        "cross_channel_compare",
        partial(cross_channel_compare, hybrid_retriever=hybrid_retriever),
        timeout_sec=settings.agent_tool_timeout,
    )
    runner.register(
        "summarize_channel",
        partial(summarize_channel, hybrid_retriever=hybrid_retriever),
        timeout_sec=settings.agent_tool_timeout,
    )

    # ── SPEC-RAG-15: analytics tools (facet API) ──
    runner.register("entity_tracker", partial(entity_tracker, hybrid_retriever=hybrid_retriever))
    runner.register("arxiv_tracker", partial(arxiv_tracker, hybrid_retriever=hybrid_retriever))

    # ── SPEC-RAG-16: pre-computed analytics (собственные коллекции) ──
    runner.register("hot_topics", hot_topics)
    runner.register("channel_expertise", channel_expertise)

    return runner
