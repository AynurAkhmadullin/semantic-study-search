"""Поиск и переключение библиотеки с одним экземпляром модели."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from semantic_study_search.api.settings import APISettings
from semantic_study_search.domain import Document, SearchResult
from semantic_study_search.ingestion.chunking import chunk_documents, token_count
from semantic_study_search.search.encoder import MAX_TOKENS, QUERY_PREFIX, E5Encoder, Encoder
from semantic_study_search.search.engine import SearchEngine
from semantic_study_search.search.relevance import default_min_score, validate_min_score
from semantic_study_search.sources import uploaded_source_id
from semantic_study_search.storage.index import IndexConfig, StoredIndex, load_index, save_index


class SearchBusyError(RuntimeError):
    """Сообщает, что модель уже выполняет вычисления."""


class QueryLimitError(ValueError):
    """Сообщает о превышении числа токенов в вопросе."""


class EmptyLibraryError(ValueError):
    """Сообщает об отсутствии материалов для поиска."""


@dataclass(frozen=True)
class LibrarySnapshot:
    """Объединяет согласованные данные и поиск одной версии библиотеки."""

    index: StoredIndex | None
    engine: SearchEngine | None


class SearchService:
    """Хранит модель и заменяет готовую библиотеку без перезапуска сервера."""

    def __init__(
        self,
        index: StoredIndex | None,
        encoder: Encoder,
        *,
        index_dir: Path | None = None,
        min_score: float | None = None,
    ):
        """Загружает готовый индекс или создаёт пустую библиотеку.

        Args:
            index: Сохранённая библиотека; None означает отсутствие материалов.
            encoder: Загруженная модель для вопросов и новых документов.
            index_dir: Каталог сохранения новых версий.
            min_score: Порог сходства; None использует начальное значение.
        """
        self.encoder = encoder
        self.index_dir = index_dir
        self.min_score = default_min_score() if min_score is None else validate_min_score(min_score)
        self._snapshot = LibrarySnapshot(
            index, SearchEngine.from_index(index, encoder) if index and index.chunks else None
        )
        self._lock = Lock()

    @property
    def index(self) -> StoredIndex | None:
        """Возвращает текущую согласованную версию библиотеки.

        Returns:
            Индекс, в том числе пустой после удаления, или None до первой загрузки.
        """
        return self._snapshot.index

    def counts(self) -> tuple[int, int]:
        """Получает размеры одной версии библиотеки.

        Returns:
            Количество документов и фрагментов.
        """
        index = self.index
        return (len(index.documents), len(index.chunks)) if index else (0, 0)

    def material_counts(self, material_id: str) -> tuple[int, int]:
        """Находит загруженный материал в активном индексе.

        Args:
            material_id: Идентификатор загрузки.

        Returns:
            Количество документов и фрагментов этого файла.
        """
        index = self.index
        if index is None:
            return 0, 0
        ids = {
            doc.id for doc in index.documents if uploaded_source_id(doc.source_url) == material_id
        }
        return len(ids), sum(chunk.document_id in ids for chunk in index.chunks)

    def search(self, query: str, top_k: int, *, relaxed: bool = False) -> list[SearchResult]:
        """Ищет по текущей библиотеке, если модель свободна.

        Args:
            query: Проверенный вопрос пользователя.
            top_k: Максимальное количество результатов.
            relaxed: Включить ближайшие фрагменты независимо от порога.

        Returns:
            Найденные фрагменты или пустой список.
        """
        if not self._lock.acquire(blocking=False):
            raise SearchBusyError(
                "Модель занята подготовкой материала или поиском. Попробуйте позже."
            )
        try:
            engine = self._snapshot.engine
            if engine is None:
                raise EmptyLibraryError(
                    "Библиотека пуста. Загрузите материал и дождитесь завершения обработки."
                )
            if token_count(self.encoder.tokenizer, QUERY_PREFIX + query) > MAX_TOKENS:
                raise QueryLimitError("Вопрос слишком длинный для модели. Сократите формулировку.")
            return engine.search(query, top_k=top_k, min_score=None if relaxed else self.min_score)
        finally:
            self._lock.release()

    def add_material(self, documents: Sequence[Document], on_stage: Callable[[str], None]) -> None:
        """Добавляет векторы нового материала и публикует готовую версию библиотеки.

        Args:
            documents: Извлечённый текст нового файла.
            on_stage: Обновление состояния фоновой обработки.
        """
        if self.index_dir is None:
            raise RuntimeError("Каталог индекса не настроен.")
        with self._lock:
            previous = self.index
            config = (
                previous.config
                if previous
                else IndexConfig(
                    model_id=self.encoder.model_id,
                    model_revision=self.encoder.revision,
                    dimension=self.encoder.dimension,
                )
            )
            on_stage("chunking")
            chunks = chunk_documents(
                documents,
                self.encoder.tokenizer,
                max_tokens=config.max_tokens,
                overlap=config.overlap,
            )
            if len(chunks) > 1000:
                raise ValueError("Материал слишком большой: больше 1000 фрагментов.")
            on_stage("embedding")
            new_engine = SearchEngine(chunks, self.encoder)
            candidate = StoredIndex(
                (previous.documents if previous else ()) + tuple(documents),
                (previous.chunks if previous else ()) + tuple(chunks),
                np.concatenate((previous.vectors, new_engine.vectors))
                if previous
                else new_engine.vectors,
                config,
            )
            prepared = LibrarySnapshot(candidate, SearchEngine.from_index(candidate, self.encoder))
            on_stage("saving")
            save_index(self.index_dir, candidate)
            self._snapshot = prepared

    def remove_material(self, identifier: str) -> None:
        """Исключает материал из сохранённого индекса и поиска без пересчёта векторов.

        Args:
            identifier: Идентификатор загруженного файла.
        """
        if self.index_dir is None:
            raise RuntimeError("Каталог индекса не настроен.")
        with self._lock:
            previous = self.index
            if previous is None:
                return
            removed = {
                doc.id
                for doc in previous.documents
                if uploaded_source_id(doc.source_url) == identifier
            }
            if not removed:
                return
            rows = [
                i for i, chunk in enumerate(previous.chunks) if chunk.document_id not in removed
            ]
            candidate = StoredIndex(
                tuple(doc for doc in previous.documents if doc.id not in removed),
                tuple(previous.chunks[i] for i in rows),
                previous.vectors[rows],
                previous.config,
            )
            prepared = LibrarySnapshot(
                candidate,
                SearchEngine.from_index(candidate, self.encoder) if candidate.chunks else None,
            )
            save_index(self.index_dir, candidate)
            self._snapshot = prepared


def load_service(settings: APISettings) -> SearchService:
    """Загружает модель и существующую или пустую библиотеку.

    Args:
        settings: Настройки индекса, модели и поиска.

    Returns:
        Сервис, готовый к поиску и загрузке материалов.
    """
    index = load_index(settings.index_dir) if (settings.index_dir / "CURRENT").exists() else None
    if index is not None:
        expected = IndexConfig()
        if (index.config.model_id, index.config.model_revision, index.config.dimension) != (
            expected.model_id,
            expected.model_revision,
            expected.dimension,
        ):
            raise ValueError("Индекс создан другой моделью; выполните semantic-study-index.")
    encoder = E5Encoder(cache_folder=settings.model_cache, offline=settings.offline)
    return SearchService(index, encoder, index_dir=settings.index_dir, min_score=settings.min_score)
