"""Проверки косинусного ранжирования, векторов и параметров поиска."""

import numpy as np
import pytest

from semantic_study_search.domain import Chunk
from semantic_study_search.search.engine import SearchEngine


@pytest.fixture
def chunks():
    """Создаёт три фрагмента для проверки ранжирования.

    Returns:
        Список тестовых фрагментов.
    """
    return [
        Chunk(f"doc:{i}", "doc", "Title", "https://example.org", f"text {i}", 0, 6)
        for i in range(3)
    ]


def test_cosine_ranking_normalizes_vectors_and_keeps_sources(chunks, encoder):
    """Поиск нормализует векторы, ранжирует по косинусному сходству и сохраняет источники.

    Args:
        chunks: Три фрагмента для проверки ранжирования.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    results = SearchEngine(chunks, encoder).search("запрос", top_k=2)
    assert [result.chunk.id for result in results] == ["doc:0", "doc:1"]
    assert [result.score for result in results] == pytest.approx([1, 1 / np.sqrt(2)])
    assert results[0].chunk.source_url == "https://example.org"


def test_large_top_k_and_repeated_queries_do_not_reindex(chunks, encoder):
    """Повторные запросы используют готовые векторы и не превышают размер библиотеки.

    Args:
        chunks: Три фрагмента для проверки ранжирования.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    engine = SearchEngine(chunks, encoder)
    assert len(engine.search("первый", top_k=10)) == 3
    engine.search("второй")
    assert encoder.passage_calls == 1
    assert encoder.query_calls == 2


def test_equal_scores_keep_corpus_order(chunks, encoder):
    """Одинаковое сходство сохраняет исходный порядок фрагментов.

    Args:
        chunks: Три фрагмента для проверки ранжирования.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    encoder.passages = np.ones((3, 2))
    results = SearchEngine(chunks, encoder).search("запрос")
    assert [result.chunk for result in results] == chunks


@pytest.mark.parametrize("query", ["", " \n ", "я" * 2001])
def test_invalid_query_is_rejected_before_inference(chunks, encoder, query):
    """Некорректный вопрос отклоняется до вычисления вектора.

    Args:
        chunks: Три фрагмента для проверки ранжирования.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        query: Некорректный поисковый вопрос.
    """
    engine = SearchEngine(chunks, encoder)
    with pytest.raises(ValueError):
        engine.search(query)
    assert encoder.query_calls == 0


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k(chunks, encoder, top_k):
    """Недопустимое количество результатов отклоняется до кодирования вопроса.

    Args:
        chunks: Три фрагмента для проверки ранжирования.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        top_k: Недопустимое количество результатов.
    """
    engine = SearchEngine(chunks, encoder)
    with pytest.raises(ValueError):
        engine.search("запрос", top_k=top_k)
    assert encoder.query_calls == 0


@pytest.mark.parametrize(
    "vectors", [np.zeros((3, 2)), np.full((3, 2), np.nan), np.ones((2, 2)), np.ones(3)]
)
def test_invalid_passage_vectors(chunks, encoder, vectors):
    """Некорректные векторы фрагментов отклоняются при создании поиска.

    Args:
        chunks: Три фрагмента для проверки ранжирования.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        vectors: Некорректные векторы фрагментов.
    """
    encoder.passages = vectors
    with pytest.raises(ValueError):
        SearchEngine(chunks, encoder)


@pytest.mark.parametrize("vector", [np.zeros(2), np.array([np.inf, 1]), np.ones(4)])
def test_invalid_query_vectors(chunks, encoder, vector):
    """Некорректный вектор вопроса отклоняется при поиске.

    Args:
        chunks: Три фрагмента для проверки ранжирования.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        vector: Некорректный вектор вопроса.
    """
    engine = SearchEngine(chunks, encoder)
    encoder.query = vector
    with pytest.raises(ValueError):
        engine.search("запрос")


def test_empty_index_and_duplicate_chunk_ids(chunks, encoder):
    """Пустая библиотека и повторяющиеся идентификаторы фрагментов отклоняются.

    Args:
        chunks: Три фрагмента для проверки ранжирования.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    with pytest.raises(ValueError):
        SearchEngine([], encoder)
    with pytest.raises(ValueError):
        SearchEngine([chunks[0], chunks[0]], encoder)


def test_threshold_keeps_boundary_and_can_return_empty(chunks, encoder):
    """Порог включает граничное сходство и допускает пустую выдачу.

    Args:
        chunks: Тестовые фрагменты.
        encoder: Кодировщик с фиксированными векторами.
    """
    engine = SearchEngine(chunks, encoder)
    assert [r.chunk.id for r in engine.search("вопрос", min_score=1)] == ["doc:0"]
    assert len(engine.search("вопрос", min_score=-1)) == 3
    encoder.query = np.array([-1, 0], dtype=np.float32)
    assert engine.search("другая тема", min_score=0.1) == []
    assert encoder.passage_calls == 1


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -1.1, 1.1, True, "0.8"])
def test_invalid_threshold_fails_before_inference(chunks, encoder, threshold):
    """Некорректный порог отклоняется до вычисления вектора вопроса.

    Args:
        chunks: Тестовые фрагменты.
        encoder: Кодировщик со счётчиком вызовов.
        threshold: Недопустимый порог.
    """
    engine = SearchEngine(chunks, encoder)
    with pytest.raises(ValueError, match="Порог"):
        engine.search("вопрос", min_score=threshold)
    assert encoder.query_calls == 0
