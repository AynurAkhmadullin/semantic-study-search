"""Поиск по косинусному сходству в библиотеке фрагментов в памяти."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from semantic_study_search.domain import Chunk, SearchResult
from semantic_study_search.search.encoder import Encoder, validate_query
from semantic_study_search.search.relevance import validate_min_score

if TYPE_CHECKING:
    from semantic_study_search.storage.index import StoredIndex


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Приводит каждый вектор матрицы к единичной длине.

    Args:
        vectors: Матрица векторов, расположенных по строкам.

    Returns:
        Матрица нормализованных векторов.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("Векторы должны быть конечной двумерной матрицей.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Невозможно нормализовать нулевой или некорректный вектор.")
    return vectors / norms


class SearchEngine:
    """Ищет фрагменты по сходству векторов запроса и учебных текстов."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        encoder: Encoder,
        *,
        vectors: NDArray[np.float32] | None = None,
    ):
        """Создаёт поисковый индекс фрагментов в памяти.

        Args:
            chunks: Учебные фрагменты с уникальными идентификаторами.
            encoder: Модель для вычисления векторов текстов и запросов.
            vectors: Готовые векторы; None вычисляет их по текстам фрагментов.
        """
        if not chunks:
            raise ValueError("Для поиска необходим хотя бы один фрагмент.")
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("Идентификаторы фрагментов должны быть уникальны.")
        self.chunks = tuple(chunks)
        self.encoder = encoder
        if vectors is None:
            vectors = encoder.encode_passages([chunk.text for chunk in chunks])
        self._vectors = _normalize(vectors)
        if len(self._vectors) != len(chunks):
            raise ValueError("Число векторов не совпадает с числом фрагментов.")

    @property
    def vectors(self) -> NDArray[np.float32]:
        """Возвращает копию векторов фрагментов для сохранения.

        Returns:
            Матрица векторов в порядке фрагментов.
        """
        return self._vectors.copy()

    @classmethod
    def from_index(cls, index: "StoredIndex", encoder: Encoder) -> "SearchEngine":
        """Создаёт поиск по сохранённым векторам без повторного кодирования документов.

        Args:
            index: Загруженный индекс библиотеки.
            encoder: Модель той же версии для кодирования запросов.

        Returns:
            Поиск по готовой библиотеке.
        """
        index.config.require_encoder(encoder)
        index.validate()
        return cls(index.chunks, encoder, vectors=index.vectors)

    def search(
        self, query: str, *, top_k: int = 5, min_score: float | None = None
    ) -> list[SearchResult]:
        """Находит учебные фрагменты по смыслу запроса.

        Args:
            query: Поисковый вопрос.
            top_k: Максимальное количество результатов.
            min_score: Нижняя граница сходства; None отключает фильтрацию.

        Returns:
            Результаты по убыванию сходства; список может быть пустым.
        """
        query = validate_query(query)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("Количество результатов должно быть положительным целым числом.")
        if min_score is not None:
            min_score = validate_min_score(min_score)
        vector = _normalize(np.asarray([self.encoder.encode_query(query)]))
        if vector.shape[1] != self._vectors.shape[1]:
            raise ValueError("Размерности векторов запроса и библиотеки не совпадают.")
        scores = np.clip(self._vectors @ vector[0], -1.0, 1.0)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchResult(self.chunks[index], float(scores[index]))
            for index in order
            if min_score is None or float(scores[index]) >= min_score
        ]
