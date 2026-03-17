"""Базовые unit-тесты для AgentService helper-логики."""

from unittest.mock import Mock

import pytest

from core.settings import Settings
from schemas.agent import ToolMeta, ToolResponse
from services.agent_service import AgentService, AgentState
from services.tools.tool_runner import ToolRunner


@pytest.fixture
def mock_settings():
    """Минимальный набор настроек для тестов AgentService."""
    settings = Mock(spec=Settings)
    settings.agent_max_steps = 4
    settings.agent_default_steps = 3
    settings.agent_tool_timeout = 5.0
    settings.agent_tool_max_tokens = 256
    settings.agent_tool_temp = 0.7
    settings.agent_tool_top_p = 0.8
    settings.agent_tool_top_k = 20
    settings.agent_tool_presence_penalty = 1.5
    settings.agent_token_budget = 2000
    settings.enable_verify_step = False
    settings.coverage_threshold = 0.65
    settings.max_refinements = 2
    settings.search_k_per_query_default = 10
    settings.reranker_top_n = 20
    return settings


@pytest.fixture
def mock_llm():
    """Мок LLM-клиента."""
    llm = Mock()
    llm.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Готово."},
                "finish_reason": "stop",
            }
        ]
    }
    return llm


@pytest.fixture
def mock_tool_runner():
    """Мок ToolRunner."""
    return Mock(spec=ToolRunner)


@pytest.fixture
def agent_service(mock_settings, mock_llm, mock_tool_runner):
    """Экземпляр AgentService для тестов."""
    return AgentService(lambda: mock_llm, mock_tool_runner, mock_settings)


def test_extract_tool_calls_supports_openai_shape(agent_service):
    """Tool calls должны читаться из OpenAI-compatible структуры."""
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"queries": ["биткоин"], "k": 5}',
                },
            }
        ],
    }

    tool_calls = agent_service._extract_tool_calls(message)

    assert tool_calls == [
        {
            "id": "call_1",
            "name": "search",
            "arguments": {"queries": ["биткоин"], "k": 5},
        }
    ]


def test_normalize_compose_context_injects_query_and_dense_score(agent_service):
    """compose_context должен получать query и dense_score из last_search_hits."""
    agent_service._current_query = "что с биткоином"
    agent_service._last_search_hits = [
        {
            "id": "doc-1",
            "text": "Биткоин обновил максимум",
            "meta": {"channel": "news"},
            "dense_score": 0.82,
        }
    ]

    normalized = agent_service._normalize_tool_params(
        "compose_context", {"hit_ids": ["doc-1"]}
    )

    assert normalized["query"] == "что с биткоином"
    assert normalized["docs"][0]["id"] == "doc-1"
    assert normalized["docs"][0]["dense_score"] == 0.82


def test_format_observation_for_search(agent_service):
    """Observation для search должен включать route и ids."""
    response = ToolResponse(
        ok=True,
        data={
            "hits": [{"id": "doc-1"}, {"id": "doc-2"}],
            "route_used": "hybrid",
            "total_found": 2,
        },
        meta=ToolMeta(took_ms=15),
    )

    observation = agent_service._format_observation(response, "search")

    assert "Route: hybrid" in observation
    assert "doc-1" in observation


def test_get_available_tools_exposes_five_llm_tools(agent_service):
    """LLM-visible tools должны совпадать с function-calling schema."""
    tools = agent_service.get_available_tools()

    assert tools["total"] == 5
    assert set(tools["tools"]) == {
        "query_plan",
        "search",
        "rerank",
        "compose_context",
        "final_answer",
    }
    assert "verify" in tools["system_tools"]
    assert "fetch_docs" in tools["system_tools"]


def test_build_final_payload_adds_disclaimer(agent_service):
    """При low coverage финальный payload должен получить disclaimer."""
    agent_service._last_compose_citations = [{"id": "doc-1", "index": 1}]
    agent_service._last_search_route = "hybrid"
    agent_service._last_plan_summary = {"normalized_queries": ["биткоин"]}

    state = AgentState()
    state.coverage = 0.2
    state.refinement_count = 2
    state.low_coverage_disclaimer = True

    payload = agent_service._build_final_payload(
        base_payload={"answer": "Краткий ответ [1]"},
        answer="Краткий ответ [1]",
        verify_res={},
        agent_state=state,
        request_id="req-1",
        step=3,
    )

    assert payload["answer"].startswith("[Примечание:")
    assert payload["citations"][0]["id"] == "doc-1"
    assert payload["route"] == "hybrid"
