import json
from unittest.mock import Mock

from core.settings import Settings
from services.query_planner_service import QueryPlannerService


def _make_planner(llm_responses: list[dict]) -> tuple[QueryPlannerService, Mock]:
    """Создаёт планировщик с мок-LLM.

    llm_responses — список словарей в формате completions:
    {"choices": [{"text": "..."}]}
    """
    settings = Settings()
    settings.max_plan_subqueries = 3

    llm = Mock()
    # Удаляем chat_completion чтобы llm вызывался через __call__
    del llm.chat_completion
    llm.side_effect = llm_responses

    planner = QueryPlannerService(llm, settings)
    return planner, llm


def test_make_plan_parses_valid_json_and_limits_subqueries():
    raw_plan = {
        "normalized_queries": ["Q1 ", "q2.", "q3", "q4", "q5"],
        "must_phrases": ["A", "a"],
        "should_phrases": ["b"],
        "metadata_filters": {
            "date_from": "2024-01-30T10:00:00",
            "date_to": "2024-01-01T12:00:00",
            "channel_ids": ["10", 20, -1],
            "min_views": "5",
            "reply_to": 0,
        },
        "k_per_query": 7,
        "fusion": "rrf",
    }
    response = {"choices": [{"text": json.dumps(raw_plan)}]}
    planner, _ = _make_planner([response])

    plan = planner.make_plan("тестовый запрос")

    assert len(plan.normalized_queries) <= 3
    assert plan.k_per_query == 7
    assert plan.fusion == "rrf"
    assert plan.metadata_filters is not None
    assert plan.metadata_filters.date_from == "2024-01-01"  # swapped and trimmed
    assert plan.metadata_filters.date_to == "2024-01-30"
    assert plan.metadata_filters.channel_ids == [10, 20]


def test_make_plan_fallback_on_bad_json():
    settings = Settings()
    llm = Mock()
    del llm.chat_completion
    llm.side_effect = [
        {"choices": [{"text": "невалидный json"}]},
        {"choices": [{"text": "невалидный json"}]},
    ]

    planner = QueryPlannerService(llm, settings)
    plan = planner.make_plan("любой запрос")

    assert plan.normalized_queries == ["любой запрос"]
    assert plan.k_per_query == settings.search_k_per_query_default
    assert plan.fusion == "rrf"


def test_filters_sql_and_imperatives():
    settings = Settings()
    llm = Mock()
    del llm.chat_completion
    raw_plan = {
        "normalized_queries": [
            "select * from table where x=1",
            "вытяни все про deepseek-v3.1 цены api",
            "DeepSeek-V3.1 API тарифы и дата вступления",
        ],
        "must_phrases": [],
        "should_phrases": [],
        "metadata_filters": None,
        "k_per_query": 10,
        "fusion": "rrf",
    }
    llm.return_value = {"choices": [{"text": json.dumps(raw_plan)}]}

    planner = QueryPlannerService(llm, settings)
    plan = planner.make_plan("Вытяни цены DeepSeek-V3.1 API")
    # Должны остаться только семантические фразы, без SQL/императивов
    assert all("select" not in q and "вытяни" not in q for q in plan.normalized_queries)
    assert len(plan.normalized_queries) >= 1
