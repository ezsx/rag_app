"""LlamaIndex agent — two configs: stock and maxed.

LI-stock: FunctionAgent + search tool (default hybrid retrieval), no tuning.
LI-maxed: FunctionAgent + search tool (weighted RRF + reranker) + initial_tool_choice.

Использует OpenAILike для подключения к llama-server (Qwen3.5-35B-A3B).
enable_thinking=False критично для Qwen3 tool calling compatibility.
"""

from __future__ import annotations

import asyncio
import time

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike

from benchmarks.config import (
    LLM_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_URL,
)
from benchmarks.llamaindex_pipeline.retriever import (
    LlamaIndexRetrieverMaxed,
    LlamaIndexRetrieverStock,
)
from benchmarks.protocols import AgentResult

SYSTEM_PROMPT = (
    "Ты помощник по новостям AI/ML из русскоязычных Telegram-каналов. "
    "Отвечай на русском языке. Используй инструмент search для поиска информации. "
    "Ссылайся на источники в ответе."
)


def _build_llm() -> OpenAILike:
    """OpenAILike → llama-server с Qwen3.5-35B-A3B."""
    return OpenAILike(
        model="qwen3.5-35b",
        api_base=f"{LLM_URL}/v1",
        api_key="not-needed",
        is_function_calling_model=True,
        is_chat_model=True,
        context_window=LLM_CONTEXT_WINDOW,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        # Qwen3 thinking mode конфликтует с tool_calls.
        # OpenAI SDK не принимает enable_thinking как top-level kwarg.
        # llama-server принимает его через extra_body в request.
        additional_kwargs={
            "extra_body": {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    )


def _format_nodes(nodes, max_docs: int = 5) -> str:
    """Форматирует retrieval results для LLM."""
    parts = []
    for i, r in enumerate(nodes[:max_docs], 1):
        text = r.text or "(текст недоступен)"
        if len(text) > 1500:
            text = text[:1500] + "..."
        parts.append(f"[{i}] ({r.doc_id})\n{text}")
    return "\n\n".join(parts)


def _build_search_tool(retriever, label: str, docs_capture: list) -> FunctionTool:
    """Создаёт search tool обёртку вокруг retriever.

    docs_capture — mutable list, сохраняет retrieved docs для agent result.
    """

    def search_documents(query: str) -> str:
        """Поиск релевантных постов из Telegram-каналов по запросу."""
        results = retriever.retrieve(query, top_k=10)
        docs_capture.clear()
        docs_capture.extend(results)
        return _format_nodes(results, max_docs=5)

    return FunctionTool.from_defaults(
        fn=search_documents,
        name="search",
        description="Поиск релевантных постов из Telegram-каналов AI/ML по запросу",
    )


def _run_agent_sync(agent, query: str) -> str:
    """Запускает async agent в sync контексте через asyncio.run()."""
    async def _inner():
        result = await agent.run(user_msg=query)
        return str(result) if result else ""
    return asyncio.run(_inner())


class LlamaIndexAgentStock:
    """LI-stock: FunctionAgent, default hybrid retrieval, no tuning."""

    def __init__(self):
        self._retriever = LlamaIndexRetrieverStock()
        self._llm = _build_llm()
        self._docs_capture: list = []
        search_tool = _build_search_tool(self._retriever, "stock", self._docs_capture)
        self._agent = FunctionAgent(
            name="rag_agent_stock",
            llm=self._llm,
            tools=[search_tool],
            system_prompt=SYSTEM_PROMPT,
        )

    def run(self, query: str) -> AgentResult:
        self._docs_capture.clear()
        t0 = time.time()
        answer = _run_agent_sync(self._agent, query)
        latency = time.time() - t0
        return AgentResult(
            answer=answer,
            docs=list(self._docs_capture),
            tool_calls=["search"],
            latency=latency,
        )


class LlamaIndexAgentMaxed:
    """LI-maxed: FunctionAgent, weighted RRF + reranker, initial_tool_choice."""

    def __init__(self):
        self._retriever = LlamaIndexRetrieverMaxed()
        self._llm = _build_llm()
        self._docs_capture: list = []
        search_tool = _build_search_tool(self._retriever, "maxed", self._docs_capture)
        self._agent = FunctionAgent(
            name="rag_agent_maxed",
            llm=self._llm,
            tools=[search_tool],
            system_prompt=SYSTEM_PROMPT,
            initial_tool_choice="search",
        )

    def run(self, query: str) -> AgentResult:
        self._docs_capture.clear()
        t0 = time.time()
        answer = _run_agent_sync(self._agent, query)
        latency = time.time() - t0
        return AgentResult(
            answer=answer,
            docs=list(self._docs_capture),
            tool_calls=["search"],
            latency=latency,
        )
