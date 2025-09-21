"""
Тесты для Agent API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from main import app


@pytest.fixture
def client():
    """Тестовый клиент FastAPI"""
    return TestClient(app)


@pytest.fixture
def mock_agent_service():
    """Мок AgentService"""
    service = Mock()
    service.get_available_tools.return_value = {
        "tools": {
            "math_eval": {
                "description": "Вычисляет математические выражения",
                "parameters": {"expression": "string"},
            }
        },
        "total": 1,
    }
    return service


def test_list_tools_endpoint(client):
    """Тест эндпоинта списка инструментов"""
    with patch("core.deps.get_agent_service") as mock_get_service:
        mock_service = Mock()
        mock_service.get_available_tools.return_value = {
            "tools": {
                "test_tool": {
                    "description": "Тестовый инструмент",
                    "parameters": {"param": "string"},
                }
            },
            "total": 1,
        }
        mock_get_service.return_value = mock_service

        response = client.get("/v1/agent/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "total" in data
        assert data["total"] == 1


def test_agent_status_endpoint(client):
    """Тест эндпоинта статуса агента"""
    with patch("core.deps.get_settings") as mock_get_settings:
        mock_settings = Mock()
        mock_settings.agent_max_steps = 4
        mock_settings.agent_token_budget = 2000
        mock_settings.agent_tool_timeout = 5.0
        mock_settings.current_collection = "test_collection"
        mock_settings.current_llm_key = "test_llm"
        mock_settings.enable_query_planner = True
        mock_get_settings.return_value = mock_settings

        response = client.get("/v1/agent/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "configuration" in data
        assert "features" in data
        assert data["configuration"]["max_steps"] == 4
        assert data["features"]["react_reasoning"] is True


@pytest.mark.asyncio
async def test_agent_stream_endpoint_structure():
    """Тест структуры SSE эндпоинта агента"""
    # Этот тест проверяет только структуру, не выполнение
    from api.v1.endpoints.agent import agent_stream
    from schemas.agent import AgentRequest
    from unittest.mock import Mock

    # Создаем мок запроса
    mock_request = Mock()
    mock_request.is_disconnected = AsyncMock(return_value=False)

    # Создаем мок агента
    mock_agent = Mock()

    async def mock_stream(request):
        from schemas.agent import AgentStepEvent

        yield AgentStepEvent(type="step_started", data={"step": 1})
        yield AgentStepEvent(type="final", data={"answer": "test"})

    mock_agent.stream_agent_response = mock_stream

    # Создаем мок настроек
    mock_settings = Mock()
    mock_settings.current_collection = "test"

    # Тестовый запрос
    agent_request = AgentRequest(query="test question")

    # Вызываем функцию
    response = await agent_stream(
        agent_request, mock_request, mock_agent, mock_settings
    )

    # Проверяем, что это EventSourceResponse
    from sse_starlette.sse import EventSourceResponse

    assert isinstance(response, EventSourceResponse)
