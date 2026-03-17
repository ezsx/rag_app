"""Тесты native function-calling цикла AgentService."""

from unittest.mock import Mock

import pytest

from core.settings import Settings
from schemas.agent import AgentAction, AgentRequest, ToolMeta, ToolResponse
from services.agent_service import AgentService
from services.tools.tool_runner import ToolRunner


def _action(step: int, tool: str, payload: dict, data: dict, ok: bool = True) -> AgentAction:
    """Утилита для построения AgentAction."""
    return AgentAction(
        step=step,
        tool=tool,
        input=payload,
        output=ToolResponse(
            ok=ok,
            data=data if ok else {},
            meta=ToolMeta(took_ms=5, error=None if ok else "tool_failed"),
        ),
    )


@pytest.fixture
def mock_settings():
    """Минимальные настройки для function-calling тестов."""
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
    """Мок LLM с настраиваемым chat_completion."""
    return Mock()


@pytest.fixture
def mock_tool_runner():
    """Мок ToolRunner."""
    return Mock(spec=ToolRunner)


@pytest.fixture
def agent_service(mock_settings, mock_llm, mock_tool_runner):
    """Экземпляр AgentService."""
    return AgentService(lambda: mock_llm, mock_tool_runner, mock_settings)


@pytest.mark.asyncio
async def test_tool_call_parsing(agent_service, mock_llm, mock_tool_runner):
    """LLM tool_call должен приводить к вызову нужного инструмента."""
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"queries": ["биткоин"], "k": 3}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_final",
                                "type": "function",
                                "function": {
                                    "name": "final_answer",
                                    "arguments": '{"answer": "Ответ [1]", "sources": [1]}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    def run_side_effect(request_id, step, req):
        if req.tool == "search":
            return _action(
                step,
                "search",
                req.input,
                {
                    "hits": [
                        {
                            "id": "doc-1",
                            "text": "Биткоин вырос",
                            "meta": {"channel": "news"},
                            "dense_score": 0.72,
                        }
                    ],
                    "route_used": "hybrid",
                    "total_found": 1,
                },
            )
        if req.tool == "final_answer":
            return _action(
                step,
                "final_answer",
                req.input,
                {"answer": req.input["answer"]},
            )
        raise AssertionError(f"unexpected tool: {req.tool}")

    mock_tool_runner.run.side_effect = run_side_effect

    events = []
    async for event in agent_service.stream_agent_response(
        AgentRequest(query="Что с биткоином?", max_steps=2)
    ):
        events.append(event)

    assert any(
        event.type == "tool_invoked" and event.data["tool"] == "search"
        for event in events
    )
    first_tool_request = mock_tool_runner.run.call_args_list[0][0][2]
    assert first_tool_request.tool == "search"
    assert first_tool_request.input["queries"] == ["биткоин"]


@pytest.mark.asyncio
async def test_final_answer_stops_loop(agent_service, mock_llm, mock_tool_runner):
    """final_answer tool_call должен завершать цикл."""
    mock_llm.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_final",
                            "type": "function",
                            "function": {
                                "name": "final_answer",
                                "arguments": '{"answer": "Короткий ответ [1]", "sources": [1]}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    mock_tool_runner.run.return_value = _action(
        1,
        "final_answer",
        {"answer": "Короткий ответ [1]", "sources": [1]},
        {"answer": "Короткий ответ [1]"},
    )

    events = []
    async for event in agent_service.stream_agent_response(
        AgentRequest(query="Тест", max_steps=3)
    ):
        events.append(event)

    assert events[-1].type == "final"
    assert events[-1].data["answer"] == "Короткий ответ [1]"
    assert mock_llm.chat_completion.call_count == 1


@pytest.mark.asyncio
async def test_text_response_as_thought(agent_service, mock_llm):
    """Обычный content без tool_calls должен эмититься как thought."""
    mock_llm.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Сначала соберу план поиска.",
                },
                "finish_reason": "length",
            }
        ]
    }

    events = []
    async for event in agent_service.stream_agent_response(
        AgentRequest(query="Тест", max_steps=1)
    ):
        events.append(event)

    assert any(
        event.type == "thought" and "Сначала соберу план" in event.data["content"]
        for event in events
    )


@pytest.mark.asyncio
async def test_multi_step_flow(agent_service, mock_llm, mock_tool_runner):
    """Полный flow search -> rerank -> compose_context -> final_answer."""
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"queries": ["биткоин"], "k": 2}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_rerank",
                                "type": "function",
                                "function": {
                                    "name": "rerank",
                                    "arguments": '{"query": "биткоин", "top_n": 2}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_compose",
                                "type": "function",
                                "function": {
                                    "name": "compose_context",
                                    "arguments": '{"hit_ids": ["doc-1", "doc-2"]}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_final",
                                "type": "function",
                                "function": {
                                    "name": "final_answer",
                                    "arguments": '{"answer": "Ответ [1][2]", "sources": [1, 2]}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    def run_side_effect(request_id, step, req):
        if req.tool == "search":
            return _action(
                step,
                "search",
                req.input,
                {
                    "hits": [
                        {
                            "id": "doc-1",
                            "text": "Биткоин вырос на 5%",
                            "meta": {"channel": "a"},
                            "dense_score": 0.81,
                        },
                        {
                            "id": "doc-2",
                            "text": "ETF поддержали рост",
                            "meta": {"channel": "b"},
                            "dense_score": 0.77,
                        },
                    ],
                    "route_used": "hybrid",
                    "total_found": 2,
                },
            )
        if req.tool == "rerank":
            return _action(
                step,
                "rerank",
                req.input,
                {"indices": [0, 1], "scores": [0.95, 0.88], "top_n": 2},
            )
        if req.tool == "compose_context":
            return _action(
                step,
                "compose_context",
                req.input,
                {
                    "prompt": "[1] Биткоин вырос\n\n[2] ETF поддержали рост",
                    "citations": [{"id": "doc-1", "index": 1}, {"id": "doc-2", "index": 2}],
                    "contexts": ["Биткоин вырос", "ETF поддержали рост"],
                    "citation_coverage": 0.78,
                },
            )
        if req.tool == "final_answer":
            return _action(step, "final_answer", req.input, {"answer": req.input["answer"]})
        raise AssertionError(f"unexpected tool: {req.tool}")

    mock_tool_runner.run.side_effect = run_side_effect

    events = []
    async for event in agent_service.stream_agent_response(
        AgentRequest(query="Что с биткоином?", max_steps=4)
    ):
        events.append(event)

    assert any(e.type == "tool_invoked" and e.data["tool"] == "search" for e in events)
    assert any(e.type == "tool_invoked" and e.data["tool"] == "rerank" for e in events)
    assert any(e.type == "citations" for e in events)
    assert events[-1].type == "final"


@pytest.mark.asyncio
async def test_refinement_triggers(agent_service, mock_llm, mock_tool_runner):
    """Низкий coverage должен запускать системный refinement."""
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_search_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"queries": ["биткоин"], "k": 2}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_compose_1",
                                "type": "function",
                                "function": {
                                    "name": "compose_context",
                                    "arguments": '{"hit_ids": ["doc-1"]}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_final",
                                "type": "function",
                                "function": {
                                    "name": "final_answer",
                                    "arguments": '{"answer": "Осторожный ответ [1]", "sources": [1]}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    call_log = []

    def run_side_effect(request_id, step, req):
        call_log.append(req.tool)
        if req.tool == "search" and len([x for x in call_log if x == "search"]) == 1:
            return _action(
                step,
                "search",
                req.input,
                {
                    "hits": [
                        {
                            "id": "doc-1",
                            "text": "Короткая заметка",
                            "meta": {"channel": "a"},
                            "dense_score": 0.44,
                        }
                    ],
                    "route_used": "hybrid",
                    "total_found": 1,
                },
            )
        if req.tool == "compose_context" and len([x for x in call_log if x == "compose_context"]) == 1:
            return _action(
                step,
                "compose_context",
                req.input,
                {
                    "prompt": "[1] Короткая заметка",
                    "citations": [{"id": "doc-1", "index": 1}],
                    "contexts": ["Короткая заметка"],
                    "citation_coverage": 0.32,
                },
            )
        if req.tool == "search":
            return _action(
                step,
                "search",
                req.input,
                {
                    "hits": [
                        {
                            "id": "doc-2",
                            "text": "Подробный материал про рынок",
                            "meta": {"channel": "b"},
                            "dense_score": 0.71,
                        }
                    ],
                    "route_used": "hybrid",
                    "total_found": 1,
                },
            )
        if req.tool == "compose_context":
            return _action(
                step,
                "compose_context",
                req.input,
                {
                    "prompt": "[1] Подробный материал про рынок",
                    "citations": [{"id": "doc-2", "index": 1}],
                    "contexts": ["Подробный материал про рынок"],
                    "citation_coverage": 0.74,
                },
            )
        if req.tool == "final_answer":
            return _action(step, "final_answer", req.input, {"answer": req.input["answer"]})
        raise AssertionError(f"unexpected tool: {req.tool}")

    mock_tool_runner.run.side_effect = run_side_effect

    events = []
    async for event in agent_service.stream_agent_response(
        AgentRequest(query="Что с биткоином?", max_steps=3)
    ):
        events.append(event)

    assert any(
        event.type == "thought" and event.data.get("refinement") is True
        for event in events
    )
    assert sum(1 for event in events if event.type == "tool_invoked" and event.data["tool"] == "search") >= 2
    assert events[-1].type == "final"
