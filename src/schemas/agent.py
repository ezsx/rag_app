from __future__ import annotations

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class ToolRequest(BaseModel):
    """Единый контракт запроса к инструменту.

    - tool: имя инструмента из реестра
    - input: произвольные параметры (JSON-совместимые)
    """

    tool: str = Field(..., description="Имя инструмента")
    input: Dict[str, Any] = Field(
        default_factory=dict, description="Параметры инструмента"
    )


class ToolMeta(BaseModel):
    """Метаданные выполнения инструмента."""

    took_ms: int = Field(..., ge=0, description="Время выполнения в миллисекундах")
    error: Optional[str] = Field(
        default=None, description="Текст ошибки, если произошла"
    )


class ToolResponse(BaseModel):
    """Структура ответа инструмента."""

    ok: bool = Field(..., description="Успешно ли выполнился инструмент")
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Результат работы инструмента"
    )
    meta: ToolMeta = Field(..., description="Метаданные выполнения")


class AgentAction(BaseModel):
    """Описывает одно действие агента для трейс-лога."""

    step: int = Field(..., ge=1, description="Порядковый номер шага")
    tool: str = Field(..., description="Имя инструмента")
    input: Dict[str, Any] = Field(default_factory=dict, description="Входные параметры")
    output: ToolResponse = Field(..., description="Ответ инструмента")


# === Схемы для ReAct Agent API ===


class AgentRequest(BaseModel):
    """Запрос для ReAct агента"""

    query: str = Field(
        ..., description="Вопрос пользователя", min_length=1, max_length=1000
    )
    collection: Optional[str] = Field(
        None, description="Название коллекции (опционально)"
    )
    model_profile: Optional[str] = Field(
        None, description="Профиль модели (опционально)"
    )
    tools_allowlist: Optional[List[str]] = Field(
        None, description="Разрешенные инструменты"
    )
    planner: bool = Field(True, description="Использовать ли планировщик запросов")
    max_steps: int = Field(4, description="Максимальное количество шагов", ge=1, le=10)


class AgentResponse(BaseModel):
    """Ответ ReAct агента"""

    answer: str = Field(..., description="Итоговый ответ агента")
    steps: List[AgentAction] = Field(..., description="Выполненные шаги")
    request_id: str = Field(..., description="Идентификатор запроса")


class AgentStepEvent(BaseModel):
    """Событие SSE для пошагового выполнения агента"""

    type: str = Field(
        ...,
        description="Тип события: step_started, thought, tool_invoked, observation, token, final",
    )
    data: Dict[str, Any] = Field(..., description="Данные события")
