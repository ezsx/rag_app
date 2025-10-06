"""Тесты для инструмента search."""

from __future__ import annotations

import os
import sys
from typing import List

# Добавляем src в PYTHONPATH для прямого импорта проектных модулей
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from schemas.search import Candidate


def run_search(**kwargs):
    # Импортируем модуль лениво, чтобы избежать тяжёлых зависимостей при импорте теста
    from services.tools.search import search as _search

    return _search(**kwargs)


class DummyHybridRetriever:
    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_plan = None

    def search_with_plan(self, query_text: str, plan) -> List[Candidate]:
        self.last_query = query_text
        self.last_plan = plan
        return [
            Candidate(
                id="channel:42",
                text="Example document",
                metadata={
                    "channel_id": "channel",
                    "message_id": "42",
                    "date": "2024-01-01",
                },
                bm25_score=None,
                dense_score=None,
                source="hybrid",
            )
        ]


def test_search_accepts_query_keyword_only():
    retriever = DummyHybridRetriever()

    result = run_search(query="Что нового?", hybrid_retriever=retriever)

    assert result["hits"], "Ожидались результаты поиска"
    assert result["route_used"] == "hybrid"
    assert retriever.last_query == "Что нового?"
    assert retriever.last_plan is not None
    assert retriever.last_plan.normalized_queries == ["Что нового?"]


def test_search_merges_query_and_queries_params():
    retriever = DummyHybridRetriever()

    result = run_search(
        query="основной",
        queries=[" дополнительный ", "основной"],
        hybrid_retriever=retriever,
    )

    assert result["hits"], "Ожидались результаты поиска"
    assert retriever.last_plan is not None
    # Параметр query должен быть первым, а дубликаты удаляются
    assert retriever.last_plan.normalized_queries == ["основной", "дополнительный"]
