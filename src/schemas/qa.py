from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum


class QARequest(BaseModel):
    """Запрос для QA API"""

    query: str = Field(
        ..., description="Вопрос пользователя", min_length=1, max_length=1000
    )
    include_context: bool = Field(False, description="Включить ли контекст в ответ")
    collection: Optional[str] = Field(
        None, description="Название коллекции (опционально)"
    )


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


# === Схемы для семантического поиска ===


class SearchRequest(BaseModel):
    """Запрос для семантического поиска"""

    query: str = Field(
        ..., description="Поисковый запрос", min_length=1, max_length=1000
    )
    k: int = Field(5, description="Количество результатов", ge=1, le=50)
    collection: Optional[str] = Field(
        None, description="Название коллекции (опционально)"
    )


class SearchResponse(BaseModel):
    """Ответ семантического поиска"""

    documents: List[str] = Field(..., description="Найденные документы")
    distances: List[float] = Field(
        ..., description="Расстояния в векторном пространстве"
    )
    metadatas: List[Dict[str, Any]] = Field(..., description="Метаданные документов")
    query: str = Field(..., description="Исходный запрос")
    total_results: int = Field(..., description="Общее количество результатов")
    collection_used: str = Field(..., description="Использованная коллекция")


# === Схемы для управления коллекциями ===


class CollectionInfo(BaseModel):
    """Информация о коллекции ChromaDB"""

    name: str = Field(..., description="Название коллекции")
    count: int = Field(..., description="Количество документов")
    metadata: Dict[str, Any] = Field({}, description="Метаданные коллекции")


class CollectionsResponse(BaseModel):
    """Список коллекций"""

    collections: List[CollectionInfo] = Field(..., description="Список коллекций")
    current_collection: str = Field(..., description="Текущая активная коллекция")


class SelectCollectionRequest(BaseModel):
    """Запрос для выбора коллекции"""

    collection_name: str = Field(..., description="Название коллекции", min_length=1)


class SelectCollectionResponse(BaseModel):
    """Ответ на выбор коллекции"""

    collection_name: str = Field(..., description="Выбранная коллекция")
    document_count: int = Field(..., description="Количество документов в коллекции")
    message: str = Field(..., description="Сообщение об успехе")


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

    llm_models: List[ModelInfo] = Field(..., description="Доступные LLM модели")
    embedding_models: List[ModelInfo] = Field(
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

    channel: str = Field(
        ..., description="Telegram канал (@username или ID)", min_length=1
    )
    since: str = Field(..., description="Дата начала в формате ISO (YYYY-MM-DD)")
    until: str = Field(..., description="Дата окончания в формате ISO (YYYY-MM-DD)")
    collection: str = Field(
        ..., description="Название коллекции ChromaDB", min_length=1
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        "auto", description="Устройство для обработки"
    )
    max_messages: Optional[int] = Field(
        None, description="Максимум сообщений (для тестирования)", ge=1
    )


class IngestJobResponse(BaseModel):
    """Ответ на запуск ingestion job"""

    job_id: str = Field(..., description="Уникальный идентификатор задачи")
    status: IngestJobStatus = Field(..., description="Статус задачи")
    message: str = Field(..., description="Сообщение")
    estimated_time: Optional[str] = Field(
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
    total_messages: Optional[int] = Field(None, description="Всего сообщений", ge=0)
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    started_at: Optional[datetime] = Field(None, description="Время начала")
    completed_at: Optional[datetime] = Field(None, description="Время завершения")
    log_messages: List[str] = Field([], description="Последние логи")
