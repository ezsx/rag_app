from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QARequest(BaseModel):
    """Запрос для QA API"""

    query: str = Field(
        ..., description="Вопрос пользователя", min_length=1, max_length=1000
    )
    include_context: bool = Field(False, description="Включить ли контекст в ответ")


class ContextItem(BaseModel):
    """Элемент контекста с метаданными"""

    document: str = Field(..., description="Текст документа")
    metadata: Dict[str, Any] = Field({}, description="Метаданные документа")
    distance: float = Field(..., description="Расстояние в векторном пространстве")


class QAResponse(BaseModel):
    """Ответ QA API"""

    answer: str = Field(..., description="Ответ на вопрос")
    query: str = Field(..., description="Исходный вопрос")


class QAResponseWithContext(QAResponse):
    """Расширенный ответ QA API с контекстом"""

    context: List[ContextItem] = Field([], description="Использованный контекст")
    context_count: int = Field(0, description="Количество найденных документов")
