"""Naive agent — dense search top-5 → single LLM call → answer.

Без agent loop, без tools, без reranking.
Показывает baseline: что даёт простой RAG (retrieve + generate).
"""

from __future__ import annotations

import json
import time
import urllib.request

from benchmarks.config import LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_URL
from benchmarks.naive.retriever import NaiveRetriever
from benchmarks.protocols import AgentResult


SYSTEM_PROMPT = (
    "Ты помощник по новостям AI/ML из русскоязычных Telegram-каналов. "
    "Отвечай на русском языке. Используй информацию из предоставленных документов. "
    "Если информации недостаточно — скажи об этом."
)


def _format_context(results) -> str:
    """Форматирует результаты retrieval в контекст для LLM."""
    parts = []
    for i, r in enumerate(results, 1):
        text = r.text or "(текст недоступен)"
        # Обрезаем длинные тексты
        if len(text) > 1500:
            text = text[:1500] + "..."
        parts.append(f"[{i}] ({r.doc_id})\n{text}")
    return "\n\n".join(parts)


def _call_llm(system: str, user: str) -> str:
    """Single LLM call через llama-server /v1/chat/completions."""
    body = json.dumps({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{LLM_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


class NaiveAgent:
    """Dense search → format context → single LLM call."""

    def __init__(self):
        self._retriever = NaiveRetriever()

    def run(self, query: str) -> AgentResult:
        t0 = time.time()

        # Retrieve top-5
        docs = self._retriever.retrieve(query, top_k=5)

        # Format context
        context = _format_context(docs)
        user_msg = f"Контекст из базы новостей:\n\n{context}\n\nВопрос: {query}"

        # Single LLM call
        answer = _call_llm(SYSTEM_PROMPT, user_msg)

        latency = time.time() - t0
        return AgentResult(
            answer=answer,
            docs=docs,
            tool_calls=[],
            latency=latency,
        )
