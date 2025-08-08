import json
from unittest.mock import Mock

from services.query_planner_service import QueryPlannerService
from core.settings import Settings


def test_make_plan_parses_valid_json_and_limits_subqueries(monkeypatch):
    settings = Settings()
    settings.max_plan_subqueries = 3

    # Готовим LLM мок, который вернет валидный JSON с 5 подзапросами
    llm = Mock()
    raw_plan = {
        "normalized_queries": ["q1", "q2", "q3", "q4", "q5"],
        "must_phrases": ["a"],
        "should_phrases": ["b"],
        "metadata_filters": {
            "date_from": "2024-01-01",
            "date_to": "2024-02-01",
        },
        "k_per_query": 7,
        "fusion": "rrf",
    }
    llm.return_value = {"choices": [{"text": json.dumps(raw_plan)}]}

    planner = QueryPlannerService(llm, settings)
    plan = planner.make_plan("тестовый запрос")

    assert len(plan.normalized_queries) == 3
    assert plan.k_per_query == 7
    assert plan.fusion == "rrf"
    assert plan.metadata_filters is not None


def test_make_plan_fallback_on_bad_json():
    settings = Settings()
    llm = Mock()
    llm.return_value = {"choices": [{"text": "невалидный json"}]}

    planner = QueryPlannerService(llm, settings)
    plan = planner.make_plan("любой запрос")

    assert plan.normalized_queries == ["любой запрос"]
    assert plan.k_per_query == settings.search_k_per_query_default
    assert plan.fusion == "rrf"
