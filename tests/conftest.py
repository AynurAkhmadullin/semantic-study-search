"""Общие фикстуры и детерминированные кодировщики для тестов."""

import re

import numpy as np
import pytest


class WordTokenizer:
    """Разбивает текст по пробелам для тестов без настоящего токенизатора E5."""

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, **kwargs):
        """Возвращает условные токены слов и их символьные границы.

        Args:
            text: Текст для разбиения на токены.
            add_special_tokens: Добавить два условных специальных токена.
            return_offsets_mapping: Включить символьные границы слов в ответ.
            **kwargs: Неиспользуемые именованные параметры совместимого вызова.

        Returns:
            Идентификаторы токенов и запрошенные границы слов.
        """
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        result = {"input_ids": list(range(len(offsets) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class FixedEncoder:
    """Возвращает заданные векторы и считает обращения к кодировщику."""

    model_id = "test/encoder"
    revision = "test-revision"
    dimension = 2

    def __init__(self):
        """Создаёт токенизатор, фиксированные векторы и счётчики вызовов."""
        self.tokenizer = WordTokenizer()
        self.passage_calls = 0
        self.query_calls = 0
        self.passages = np.array([[10, 0], [1, 1], [0, 5]], dtype=np.float32)
        self.query = np.array([1, 0], dtype=np.float32)

    def encode_passages(self, texts):
        """Возвращает заданные векторы фрагментов и учитывает вызов.

        Args:
            texts: Тексты фрагментов; содержимое не влияет на фиксированные векторы.

        Returns:
            Заданные векторы фрагментов.
        """
        self.passage_calls += 1
        return self.passages

    def encode_query(self, query):
        """Возвращает заданный вектор запроса и учитывает вызов.

        Args:
            query: Вопрос; содержимое не влияет на фиксированный вектор.

        Returns:
            Заданный вектор вопроса.
        """
        self.query_calls += 1
        return self.query


@pytest.fixture
def tokenizer():
    """Создаёт токенизатор для тестов без загрузки модели.

    Returns:
        Токенизатор по пробельным словам.
    """
    return WordTokenizer()


@pytest.fixture
def encoder():
    """Создаёт кодировщик с фиксированными векторами.

    Returns:
        Тестовый кодировщик со счётчиками вызовов.
    """
    return FixedEncoder()


@pytest.fixture
def library_documents():
    """Создаёт три документа с уникальными источниками.

    Returns:
        Список тестовых документов.
    """
    from semantic_study_search.domain import Document

    return [
        Document(str(i), f"Материал {i}", f"https://example.org/{i}", f"Текст {i}")
        for i in range(3)
    ]


@pytest.fixture
def index_config(encoder):
    """Создаёт настройки индекса для тестового кодировщика.

    Args:
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.

    Returns:
        Совместимые настройки индекса.
    """
    from semantic_study_search.storage.index import IndexConfig

    return IndexConfig(
        model_id=encoder.model_id, model_revision=encoder.revision, dimension=encoder.dimension
    )


@pytest.fixture
def sample_index(library_documents, encoder, index_config):
    """Собирает индекс из тестовых документов и готовых векторов.

    Args:
        library_documents: Три тестовых документа.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        index_config: Настройки индекса для тестового кодировщика.

    Returns:
        Индекс документов, фрагментов и векторов.
    """
    from semantic_study_search.ingestion.chunking import chunk_documents
    from semantic_study_search.search.engine import SearchEngine
    from semantic_study_search.storage.index import StoredIndex

    chunks = chunk_documents(library_documents, encoder.tokenizer)
    engine = SearchEngine(chunks, encoder)
    return StoredIndex(tuple(library_documents), tuple(chunks), engine.vectors, index_config)
