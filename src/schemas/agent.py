from __future__ import annotations

from typing import Any, Dict, Optional
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
