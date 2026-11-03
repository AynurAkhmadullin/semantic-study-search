"""Модели данных учебной библиотеки и результатов поиска."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Хранит текст учебного документа, название и ссылку на источник."""

    id: str
    title: str
    source_url: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """Хранит фрагмент текста, его источник и символьные границы в документе."""

    id: str
    document_id: str
    title: str
    source_url: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SearchResult:
    """Связывает найденный фрагмент с оценкой сходства с запросом."""

    chunk: Chunk
    score: float
