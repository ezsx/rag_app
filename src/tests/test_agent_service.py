"""
Тесты для AgentService
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from services.agent_service import AgentService
from services.tools.tool_runner import ToolRunner
from schemas.agent import AgentRequest
from core.settings import Settings


@pytest.fixture
def mock_settings():
    """Мок настроек"""
    settings = Mock(spec=Settings)
    settings.agent_max_steps = 4
    settings.agent_token_budget = 2000
    settings.agent_tool_timeout = 5.0
    return settings


@pytest.fixture
def mock_llm():
    """Мок LLM"""
    llm = Mock()
    llm.return_value = {"choices": [{"text": "FinalAnswer: Это тестовый ответ"}]}
    return llm


@pytest.fixture
def mock_tool_runner():
    """Мок ToolRunner"""
    return Mock(spec=ToolRunner)


@pytest.fixture
def agent_service(mock_settings, mock_llm, mock_tool_runner):
    """Экземпляр AgentService для тестов"""
    llm_factory = lambda: mock_llm
    return AgentService(llm_factory, mock_tool_runner, mock_settings)


@pytest.mark.asyncio
async def test_parse_llm_response_with_final_answer(agent_service):
    """Тест парсинга ответа LLM с финальным ответом"""
    response = "Thought: Я думаю об этом\nFinalAnswer: Это мой финальный ответ"

    thought, action, final_answer = agent_service._parse_llm_response(response)

    assert thought == "Я думаю об этом"
    assert action is None
    assert final_answer == "Это мой финальный ответ"


@pytest.mark.asyncio
async def test_parse_llm_response_with_action(agent_service):
    """Тест парсинга ответа LLM с действием"""
    response = 'Thought: Нужно что-то сделать\nAction: math_eval {"expression": "2+2"}'

    thought, action, final_answer = agent_service._parse_llm_response(response)

    assert thought == "Нужно что-то сделать"
    assert action == 'math_eval {"expression": "2+2"}'
    assert final_answer is None


@pytest.mark.asyncio
async def test_format_observation_success(agent_service):
    """Тест форматирования успешного результата инструмента"""
    from schemas.agent import ToolResponse, ToolMeta

    response = ToolResponse(
        ok=True, data={"result": 4, "expression": "2+2"}, meta=ToolMeta(took_ms=100)
    )

    observation = agent_service._format_observation(response)

    assert "result: 4" in observation
    assert "expression: 2+2" in observation


@pytest.mark.asyncio
async def test_format_observation_error(agent_service):
    """Тест форматирования ошибки инструмента"""
    from schemas.agent import ToolResponse, ToolMeta

    response = ToolResponse(
        ok=False, data={}, meta=ToolMeta(took_ms=100, error="Деление на ноль")
    )

    observation = agent_service._format_observation(response)

    assert "Ошибка: Деление на ноль" in observation


@pytest.mark.asyncio
async def test_execute_action_valid_json(agent_service, mock_tool_runner):
    """Тест выполнения действия с валидным JSON"""
    from schemas.agent import AgentAction, ToolResponse, ToolMeta

    # Настраиваем мок
    mock_action = AgentAction(
        step=1,
        tool="math_eval",
        input={"expression": "2+2"},
        output=ToolResponse(ok=True, data={"result": 4}, meta=ToolMeta(took_ms=100)),
    )
    mock_tool_runner.run.return_value = mock_action

    action_text = 'math_eval {"expression": "2+2"}'
    result = await agent_service._execute_action(action_text, "test_req", 1)

    assert result is not None
    assert result.tool == "math_eval"
    assert result.input == {"expression": "2+2"}


@pytest.mark.asyncio
async def test_execute_action_invalid_json(agent_service, mock_tool_runner):
    """Тест выполнения действия с невалидным JSON"""
    from schemas.agent import AgentAction, ToolResponse, ToolMeta

    # Настраиваем мок для обработки невалидного JSON
    mock_action = AgentAction(
        step=1,
        tool="math_eval",
        input={"raw_input": "invalid json"},
        output=ToolResponse(ok=True, data={}, meta=ToolMeta(took_ms=50)),
    )
    mock_tool_runner.run.return_value = mock_action

    action_text = "math_eval invalid json"
    result = await agent_service._execute_action(action_text, "test_req", 1)

    assert result is not None
    assert result.tool == "math_eval"
    assert result.input == {"raw_input": "invalid json"}


def test_get_available_tools(agent_service):
    """Тест получения списка доступных инструментов"""
    tools = agent_service.get_available_tools()

    assert "tools" in tools
    assert "total" in tools
    assert tools["total"] > 0

    # Проверяем наличие основных инструментов
    tools_list = tools["tools"]
    expected_tools = [
        "router_select",
        "compose_context",
        "fetch_docs",
        "dedup_diversify",
        "verify",
        "math_eval",
        "time_now",
    ]

    for tool in expected_tools:
        assert tool in tools_list
        assert "description" in tools_list[tool]
        assert "parameters" in tools_list[tool]


@pytest.mark.asyncio
async def test_stream_agent_response_simple(agent_service, mock_llm):
    """Тест простого потока агента с финальным ответом"""
    request = AgentRequest(query="Тестовый вопрос", max_steps=1)

    # Настраиваем мок LLM для возврата финального ответа
    mock_llm.return_value = {"choices": [{"text": "FinalAnswer: Это тестовый ответ"}]}

    events = []
    async for event in agent_service.stream_agent_response(request):
        events.append(event)

    # Проверяем, что получили ожидаемые события
    assert len(events) >= 2  # Минимум step_started и final

    # Первое событие должно быть step_started
    assert events[0].type == "step_started"
    assert events[0].data["step"] == 1

    # Последнее событие должно быть final
    final_event = next(e for e in events if e.type == "final")
    assert "answer" in final_event.data
    assert final_event.data["answer"] == "Это тестовый ответ"
