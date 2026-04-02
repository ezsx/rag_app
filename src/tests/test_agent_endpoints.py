"""
Тесты для Agent API endpoints — SSE структура.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from api.v1.endpoints.agent import agent_stream
from schemas.agent import AgentRequest


@pytest.mark.asyncio
async def test_agent_stream_endpoint_structure():
    """Тест структуры SSE эндпоинта агента."""
    from sse_starlette.sse import EventSourceResponse

    from schemas.agent import AgentStepEvent

    mock_request = Mock()
    mock_request.is_disconnected = AsyncMock(return_value=False)

    mock_agent = Mock()

    async def mock_stream(request):
        yield AgentStepEvent(type="step_started", data={"step": 1})
        yield AgentStepEvent(type="final", data={"answer": "test"})

    mock_agent.stream_agent_response = mock_stream

    mock_settings = Mock()
    mock_settings.current_collection = "test"

    agent_request = AgentRequest(query="test question")

    response = await agent_stream(
        agent_request, mock_request, mock_agent, mock_settings
    )

    assert isinstance(response, EventSourceResponse)
