import json
import pytest


@pytest.mark.skip(
    reason="GBNF инт. тест — требует локальной модели и llama.cpp. Подготовлен как хелпер."
)
def test_query_planner_gbnf_smoke():
    from core.deps import get_planner_llm
    from services.query_planner_service import QueryPlannerService
    from core.settings import get_settings

    settings = get_settings()
    settings.use_gbnf_planner = True

    llm = get_planner_llm()
    svc = QueryPlannerService(llm, settings)

    plan = svc.make_plan("Новости DeepSeek-V3.1 API тарифы и дата вступления")
    assert plan is not None
    assert isinstance(plan.normalized_queries, list)
    assert 1 <= len(plan.normalized_queries) <= 6
