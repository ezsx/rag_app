from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, validator


class MetadataFilters(BaseModel):
    channel_usernames: list[str] | None = Field(
        None, description="Список @username каналов Telegram"
    )
    channel_ids: list[int] | None = Field(
        None, description="Список ID каналов Telegram"
    )
    date_from: str | None = Field(
        None, description="Начальная дата (ISO, YYYY-MM-DD или YYYY-MM-DDTHH:MM:SS)"
    )
    date_to: str | None = Field(
        None, description="Конечная дата (ISO, YYYY-MM-DD или YYYY-MM-DDTHH:MM:SS)"
    )
    min_views: int | None = Field(None, ge=0, description="Минимум просмотров")
    reply_to: int | None = Field(
        None, description="ID сообщения, на которое был ответ"
    )

    @validator("date_from", "date_to")
    def validate_iso_date(cls, v: str | None):
        if v is None:
            return v
        try:
            # допускаем дату или дату-время
            if len(v) == 10:
                datetime.strptime(v, "%Y-%m-%d")
            else:
                # Попытаемся распарсить ISO-8601 форматы
                datetime.fromisoformat(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Некорректный ISO формат даты: {v}") from exc
        return v


class SearchPlan(BaseModel):
    normalized_queries: list[str] = Field(
        ..., description="Нормализованные под-запросы"
    )
    must_phrases: list[str] = Field(default_factory=list)
    should_phrases: list[str] = Field(default_factory=list)
    metadata_filters: MetadataFilters | None = None
    k_per_query: int = Field(..., gt=0)
    fusion: Literal["rrf", "mmr"] = Field("rrf")
    strategy: Literal["broad", "temporal", "channel", "entity"] = Field(
        "broad", description="Стратегия поиска: broad|temporal|channel|entity"
    )

    @validator("normalized_queries")
    def non_empty_queries(cls, v: list[str]):
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
    documents: list[str]
    distances: list[float]
    metadatas: list[dict]
    plan: SearchPlan | None = None


# === Hybrid/BM25 общий формат кандидата ===
class Candidate(BaseModel):
    id: str
    text: str
    metadata: dict
    bm25_score: float | None = None
    dense_score: float | None = None
    source: Literal["bm25", "dense", "hybrid"]
