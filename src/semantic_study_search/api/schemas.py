"""Форматы запросов и ответов поискового API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints

from semantic_study_search.limits import MAX_QUERY_CHARACTERS, MAX_RESULTS
from semantic_study_search.sources import LOCAL_SOURCE_PATTERN


class SearchRequest(BaseModel):
    """Описывает вопрос пользователя и размер поисковой выдачи."""

    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str,
        StringConstraints(
            strict=True, strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARACTERS
        ),
    ]
    top_k: int = Field(default=5, ge=1, le=MAX_RESULTS, strict=True)
    relaxed: bool = Field(default=False, strict=True)


class ChunkResponse(BaseModel):
    """Описывает найденный фрагмент и его источник."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    title: str
    source_url: HttpUrl | Annotated[str, StringConstraints(pattern=LOCAL_SOURCE_PATTERN)]
    file_name: str | None = None
    text: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class SearchHit(BaseModel):
    """Связывает фрагмент с оценкой косинусного сходства."""

    model_config = ConfigDict(from_attributes=True)

    chunk: ChunkResponse
    score: float = Field(ge=-1, le=1, allow_inf_nan=False)


class SearchResponse(BaseModel):
    """Содержит обработанный запрос и найденные фрагменты."""

    query: str
    results: list[SearchHit]


class HealthResponse(BaseModel):
    """Сообщает о готовности поиска и размере загруженной библиотеки."""

    status: Literal["ready", "unavailable"]
    documents: int = Field(ge=0)
    fragments: int = Field(ge=0)


class ErrorResponse(BaseModel):
    """Содержит краткое сообщение об ошибке запроса."""

    detail: str


class MaterialResponse(BaseModel):
    """Сообщает о ходе подготовки материала и результате загрузки."""

    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    file_name: str
    title: str
    status: Literal["queued", "extracting", "chunking", "embedding", "saving", "ready", "failed"]
    documents: int = Field(default=0, ge=0)
    fragments: int = Field(default=0, ge=0)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class MaterialListResponse(BaseModel):
    """Содержит состояние ранее загруженных материалов автора."""

    materials: list[MaterialResponse]
