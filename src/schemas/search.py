from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime


class MetadataFilters(BaseModel):
    channel_usernames: Optional[List[str]] = Field(
        None, description="Список @username каналов Telegram"
    )
    channel_ids: Optional[List[int]] = Field(
        None, description="Список ID каналов Telegram"
    )
    date_from: Optional[str] = Field(
        None, description="Начальная дата (ISO, YYYY-MM-DD или YYYY-MM-DDTHH:MM:SS)"
    )
    date_to: Optional[str] = Field(
        None, description="Конечная дата (ISO, YYYY-MM-DD или YYYY-MM-DDTHH:MM:SS)"
    )
    min_views: Optional[int] = Field(None, ge=0, description="Минимум просмотров")
    reply_to: Optional[int] = Field(
        None, description="ID сообщения, на которое был ответ"
    )

    @validator("date_from", "date_to")
    def validate_iso_date(cls, v: Optional[str]):
        if v is None:
            return v
        try:
            # допускаем дату или дату-время
            if len(v) == 10:
                datetime.strptime(v, "%Y-%m-%d")
            else:
                # Попытаемся распарсить ISO-8601 форматы
                datetime.fromisoformat(v)
        except Exception as exc:
            raise ValueError(f"Некорректный ISO формат даты: {v}") from exc
        return v


class SearchPlan(BaseModel):
    normalized_queries: List[str] = Field(
        ..., description="Нормализованные под-запросы"
    )
    must_phrases: List[str] = Field(default_factory=list)
    should_phrases: List[str] = Field(default_factory=list)
    metadata_filters: Optional[MetadataFilters] = None
    k_per_query: int = Field(..., gt=0)
    fusion: Literal["rrf", "mmr"] = Field("rrf")

    @validator("normalized_queries")
    def non_empty_queries(cls, v: List[str]):
        v = [q.strip() for q in v if q and q.strip()]
        if not v:
            raise ValueError("normalized_queries не может быть пустым")
        return v


class SearchPlanRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    include_context: bool = Field(False)
    plan_debug: bool = Field(False)


class SearchResponse(BaseModel):
    documents: List[str]
    distances: List[float]
    metadatas: List[Dict]
    plan: Optional[SearchPlan] = None
