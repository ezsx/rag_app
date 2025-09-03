import json
from unittest.mock import Mock

from services.query_planner_service import QueryPlannerService
from core.settings import Settings


def test_make_plan_parses_valid_json_and_limits_subqueries(monkeypatch):
    settings = Settings()
    settings.max_plan_subqueries = 3

    # Готовим LLM мок, который вернет валидный JSON с 5 подзапросами (json_schema режим)
    llm = Mock()
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
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps(raw_plan)}}]
    }

    planner = QueryPlannerService(llm, settings)
    plan = planner.make_plan("тестовый запрос")

    assert len(plan.normalized_queries) == 3
    assert plan.k_per_query == 7
    assert plan.fusion == "rrf"
    assert plan.metadata_filters is not None
    assert plan.metadata_filters.date_from == "2024-01-01"  # swapped and trimmed
    assert plan.metadata_filters.date_to == "2024-01-30"
    assert plan.metadata_filters.channel_ids == [10, 20]


def test_make_plan_fallback_on_bad_json():
    settings = Settings()
    llm = Mock()
    # Оба вызова json_schema возвращают невалидный JSON
    llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": "невалидный json"}}]},
        {"choices": [{"message": {"content": "невалидный json"}}]},
    ]

    planner = QueryPlannerService(llm, settings)
    plan = planner.make_plan("любой запрос")

    assert plan.normalized_queries == ["любой запрос"]
    assert plan.k_per_query == settings.search_k_per_query_default
    assert plan.fusion == "rrf"


def test_post_validate_dogen_min_three():
    settings = Settings()
    # json_schema вызов вернет 1 подзапрос, затем доген до 3
    llm = Mock()
    first = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "normalized_queries": ["alpha"],
                            "must_phrases": [],
                            "should_phrases": [],
                            "metadata_filters": None,
                            "k_per_query": 10,
                            "fusion": "mmr",
                        }
                    )
                }
            }
        ]
    }
    # Догенерация вернет массив из 2 строк
    second = {"choices": [{"message": {"content": json.dumps(["beta", "gamma"])}}]}
    llm.create_chat_completion.side_effect = [first, second]

    planner = QueryPlannerService(llm, settings)
    plan = planner.make_plan("запрос")
    assert len(plan.normalized_queries) >= 3
    assert plan.fusion in ("rrf", "mmr")


def test_filters_sql_and_imperatives():
    settings = Settings()
    llm = Mock()
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
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps(raw_plan)}}]
    }

    planner = QueryPlannerService(llm, settings)
    plan = planner.make_plan("Вытяни цены DeepSeek-V3.1 API")
    # Должны остаться только семантические фразы, без SQL/императивов
    assert all("select" not in q and "вытяни" not in q for q in plan.normalized_queries)
    assert len(plan.normalized_queries) >= 1
