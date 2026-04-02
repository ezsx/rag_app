from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    """Запрос для QA API"""

    query: str = Field(
        ..., description="Вопрос пользователя", min_length=1, max_length=1000
    )
    include_context: bool = Field(False, description="Включить ли контекст в ответ")
    collection: str | None = Field(
        None, description="Название коллекции (опционально)"
    )


class ContextItem(BaseModel):
    """Элемент контекста с метаданными"""

    document: str = Field(..., description="Текст документа")
    metadata: dict[str, Any] = Field({}, description="Метаданные документа")
    distance: float = Field(..., description="Расстояние в векторном пространстве")


class QAResponse(BaseModel):
    """Ответ QA API"""

    answer: str = Field(..., description="Ответ на вопрос")
    query: str = Field(..., description="Исходный вопрос")


class QAResponseWithContext(QAResponse):
    """Расширенный ответ QA API с контекстом"""

    context: list[ContextItem] = Field([], description="Использованный контекст")
    context_count: int = Field(0, description="Количество найденных документов")


# === Схемы для семантического поиска ===


class SearchRequest(BaseModel):
    """Запрос для семантического поиска"""

    query: str = Field(
        ..., description="Поисковый запрос", min_length=1, max_length=1000
    )
    k: int = Field(5, description="Количество результатов", ge=1, le=50)
    collection: str | None = Field(
        None, description="Название коллекции (опционально)"
    )


class SearchResponse(BaseModel):
    """Ответ семантического поиска"""

    documents: list[str] = Field(..., description="Найденные документы")
    distances: list[float] = Field(
        ..., description="Расстояния в векторном пространстве"
    )
    metadatas: list[dict[str, Any]] = Field(..., description="Метаданные документов")
    query: str = Field(..., description="Исходный запрос")
    total_results: int = Field(..., description="Общее количество результатов")
    collection_used: str = Field(..., description="Использованная коллекция")


# === Схемы для управления моделями ===


class ModelType(str, Enum):
    """Типы моделей"""

    LLM = "llm"
    EMBEDDING = "embedding"


class ModelInfo(BaseModel):
    """Информация о модели"""

    key: str = Field(..., description="Ключ модели")
    name: str = Field(..., description="Полное название модели")
    description: str = Field(..., description="Описание модели")
    type: ModelType = Field(..., description="Тип модели")


class AvailableModelsResponse(BaseModel):
    """Список доступных моделей"""

    llm_models: list[ModelInfo] = Field(..., description="Доступные LLM модели")
    embedding_models: list[ModelInfo] = Field(
        ..., description="Доступные embedding модели"
    )
    current_llm: str = Field(..., description="Текущая LLM модель")
    current_embedding: str = Field(..., description="Текущая embedding модель")


class SelectModelRequest(BaseModel):
    """Запрос для выбора модели"""

    model_key: str = Field(..., description="Ключ модели", min_length=1)
    model_type: ModelType = Field(..., description="Тип модели (llm или embedding)")


class SelectModelResponse(BaseModel):
    """Ответ на выбор модели"""

    model_key: str = Field(..., description="Выбранная модель")
    model_type: ModelType = Field(..., description="Тип модели")
    message: str = Field(..., description="Сообщение об успехе")


# === Схемы для Telegram Ingestion ===


class IngestJobStatus(str, Enum):
    """Статусы задач ingestion"""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TelegramIngestRequest(BaseModel):
    """Запрос для запуска Telegram ingestion"""

    # Совместимость: один канал или список каналов
    channel: str | None = Field(
        None, description="Один Telegram канал (@username или ID)", min_length=1
    )
    channels: list[str] | None = Field(
        None, description="Список каналов (@username или ID)"
    )
    since: str = Field(..., description="Дата начала в формате ISO (YYYY-MM-DD)")
    until: str = Field(..., description="Дата окончания в формате ISO (YYYY-MM-DD)")
    collection: str = Field(
        ..., description="Название коллекции ChromaDB", min_length=1
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        "auto", description="Устройство для обработки"
    )
    max_messages: int | None = Field(
        None, description="Максимум сообщений (для тестирования)", ge=1
    )
    chunk_size: int | None = Field(
        0, description="Размер чанка для длинных сообщений (0 = без разбиения)", ge=0
    )


class IngestJobResponse(BaseModel):
    """Ответ на запуск ingestion job"""

    job_id: str = Field(..., description="Уникальный идентификатор задачи")
    status: IngestJobStatus = Field(..., description="Статус задачи")
    message: str = Field(..., description="Сообщение")
    estimated_time: str | None = Field(
        None, description="Примерное время выполнения"
    )


class IngestJobStatusResponse(BaseModel):
    """Статус и прогресс ingestion job"""

    job_id: str = Field(..., description="ID задачи")
    status: IngestJobStatus = Field(..., description="Текущий статус")
    progress: float = Field(
        0.0, description="Прогресс выполнения (0.0-1.0)", ge=0.0, le=1.0
    )
    messages_processed: int = Field(0, description="Обработано сообщений", ge=0)
    total_messages: int | None = Field(None, description="Всего сообщений", ge=0)
    error_message: str | None = Field(None, description="Сообщение об ошибке")
    started_at: datetime | None = Field(None, description="Время начала")
    completed_at: datetime | None = Field(None, description="Время завершения")
    log_messages: list[str] = Field([], description="Последние логи")
