import json
from fastapi.testclient import TestClient

from main import app
from core.deps import get_query_planner, get_retriever


client = TestClient(app)


def test_search_plan_endpoint(monkeypatch):
    # Мокаем планировщик в DI
    from services.query_planner_service import QueryPlannerService

    class DummyPlanner(QueryPlannerService):
        def __init__(self):
            pass

        def make_plan(self, query: str):
            from schemas.search import SearchPlan

            return SearchPlan(
                normalized_queries=[query, f"{query} уточнение"],
                must_phrases=[],
                should_phrases=[],
                metadata_filters=None,
                k_per_query=3,
                fusion="rrf",
            )

    # Подменяем зависимость get_query_planner
    app.dependency_overrides[get_query_planner] = lambda: DummyPlanner()

    resp = client.post("/v1/search/plan", json={"query": "курс доллара"})
    assert resp.status_code == 200
    data = resp.json()
    assert "normalized_queries" in data and len(data["normalized_queries"]) == 2


def test_search_with_rrf(monkeypatch):
    # Мокаем retriever.search, чтобы вернуть разные ранжированные результаты
    class DummyRetriever:
        def search(self, query: str, k: int, filters=None):
            if "уточнение" in query:
                docs = ["B", "C"]
                dists = [0.2, 0.3]
                metas = [{"id": 2}, {"id": 3}]
            else:
                docs = ["A", "B"]
                dists = [0.1, 0.4]
                metas = [{"id": 1}, {"id": 2}]
            return docs, dists, metas

    from schemas.search import SearchPlan

    class DummyPlanner:
        def make_plan(self, query: str):
            return SearchPlan(
                normalized_queries=[query, f"{query} уточнение"],
                must_phrases=[],
                should_phrases=[],
                metadata_filters=None,
                k_per_query=2,
                fusion="rrf",
            )

    # Подмена зависимостей
    app.dependency_overrides[get_retriever] = lambda: DummyRetriever()
    app.dependency_overrides[get_query_planner] = lambda: DummyPlanner()

    resp = client.post("/v1/search", json={"query": "курс доллара", "plan_debug": True})
    assert resp.status_code == 200
    data = resp.json()
    # Проверим, что RRF вернул объединение без дублей и в приоритете A, B, C
    assert data["documents"][0] == "A"
    assert data["documents"][1] == "B"
    assert data["documents"][2] == "C"
    assert "plan" in data

    # Сбрасываем overrides
    app.dependency_overrides.pop(get_retriever, None)
    app.dependency_overrides.pop(get_query_planner, None)
