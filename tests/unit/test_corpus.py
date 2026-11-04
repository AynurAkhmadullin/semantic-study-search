"""Проверки загрузки и валидации пользовательских JSON-корпусов."""

import json

import pytest

from semantic_study_search.ingestion.corpus import load_corpus


@pytest.fixture
def row():
    """Создаёт корректную запись документа.

    Returns:
        Словарь с обязательными полями документа.
    """
    return {
        "id": "one",
        "title": "Документ",
        "source_url": "https://example.org/1",
        "text": "Текст",
    }


def write_corpus(tmp_path, payload):
    """Записывает JSON-корпус в UTF-8.

    Args:
        tmp_path: Временный каталог теста.
        payload: Проверяемое содержимое JSON-корпуса.

    Returns:
        Путь к созданному JSON-файлу.
    """
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_custom_corpus_preserves_unicode_and_source(tmp_path, row):
    """Загрузка корпуса сохраняет Unicode, название и источник документа.

    Args:
        tmp_path: Временный каталог теста.
        row: Корректная запись документа.
    """
    path = write_corpus(tmp_path, {"schema_version": 1, "documents": [row]})
    document = load_corpus(path)[0]
    assert document.text == "Текст"
    assert document.title == "Документ"
    assert document.source_url == row["source_url"]


@pytest.mark.parametrize("payload", [[], {}, {"schema_version": 2}, {"schema_version": 1}])
def test_invalid_schema(tmp_path, payload):
    """Неподдерживаемая или неполная схема корпуса отклоняется.

    Args:
        tmp_path: Временный каталог теста.
        payload: Проверяемое содержимое JSON-корпуса.
    """
    with pytest.raises(ValueError):
        load_corpus(write_corpus(tmp_path, payload))


@pytest.mark.parametrize("rows", [[], [None], [{"text": "x"}]])
def test_missing_documents_or_fields(tmp_path, rows):
    """Отсутствие документов или обязательных полей вызывает ошибку.

    Args:
        tmp_path: Временный каталог теста.
        rows: Некорректный список документов.
    """
    with pytest.raises(ValueError):
        load_corpus(write_corpus(tmp_path, {"schema_version": 1, "documents": rows}))


@pytest.mark.parametrize("change", [{"text": "  "}, {"id": 1}, {"source_url": "file:///etc/test"}])
def test_invalid_document(tmp_path, row, change):
    """Некорректные значения полей документа отклоняются.

    Args:
        tmp_path: Временный каталог теста.
        row: Корректная запись документа.
        change: Изменение, нарушающее контракт документа.
    """
    with pytest.raises(ValueError):
        load_corpus(write_corpus(tmp_path, {"schema_version": 1, "documents": [row | change]}))


def test_duplicate_document_id(tmp_path, row):
    """Повторяющиеся идентификаторы документов отклоняются.

    Args:
        tmp_path: Временный каталог теста.
        row: Корректная запись документа.
    """
    with pytest.raises(ValueError, match="Повторяющийся"):
        load_corpus(write_corpus(tmp_path, {"schema_version": 1, "documents": [row, row]}))
