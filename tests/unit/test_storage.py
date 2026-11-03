"""Проверки целостности и атомарного сохранения версий индекса."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from semantic_study_search.search.engine import SearchEngine
from semantic_study_search.storage.index import (
    corpus_fingerprint,
    file_digest,
    load_index,
    save_index,
)


def generation_path(path):
    """Определяет каталог активной версии индекса по указателю CURRENT.

    Args:
        path: Каталог сохранённого индекса.

    Returns:
        Каталог активной версии индекса.
    """
    pointer = json.loads((path / "CURRENT").read_text(encoding="utf-8"))
    return path / "versions" / pointer["generation"]


def edit_metadata(path, key, value):
    """Изменяет метаданные индекса и обновляет контрольную сумму базы.

    Args:
        path: Каталог сохранённого индекса.
        key: Ключ изменяемых метаданных.
        value: Новое значение метаданных.
    """
    database = generation_path(path) / "library.sqlite3"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE metadata SET value = ? WHERE key = ?", (json.dumps(value), key))
    pointer = json.loads((path / "CURRENT").read_text(encoding="utf-8"))
    pointer["database_sha256"] = file_digest(database)
    (path / "CURRENT").write_text(json.dumps(pointer), encoding="utf-8")


def test_roundtrip_preserves_alignment_and_search_without_document_encoding(
    tmp_path, sample_index, encoder
):
    """Сохранение и загрузка сохраняют индекс и позволяют искать без пересчёта документов.

    Args:
        tmp_path: Временный каталог теста.
        sample_index: Индекс трёх документов с готовыми векторами.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    path = tmp_path / "библиотека с пробелами"
    save_index(path, sample_index)
    loaded = load_index(path)
    assert loaded.documents == sample_index.documents
    assert loaded.chunks == sample_index.chunks
    assert loaded.config == sample_index.config
    np.testing.assert_array_equal(loaded.vectors, sample_index.vectors)
    encoder.passage_calls = 0
    results = SearchEngine.from_index(loaded, encoder).search("запрос")
    assert encoder.passage_calls == 0
    assert encoder.query_calls == 1
    assert [result.chunk.document_id for result in results] == ["0", "1", "2"]


def test_index_missing_does_not_create_empty_database(tmp_path):
    """Чтение отсутствующего индекса не создаёт пустое хранилище.

    Args:
        tmp_path: Временный каталог теста.
    """
    with pytest.raises(FileNotFoundError, match="semantic-study-index"):
        load_index(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("filename", ["library.sqlite3", "vectors.npy"])
def test_corrupted_files_are_rejected(tmp_path, sample_index, filename):
    """Повреждение файлов индекса обнаруживается по контрольным суммам.

    Args:
        tmp_path: Временный каталог теста.
        sample_index: Индекс трёх документов с готовыми векторами.
        filename: Имя повреждаемого файла индекса.
    """
    save_index(tmp_path, sample_index)
    target = generation_path(tmp_path) / filename
    target.write_bytes(target.read_bytes()[:10])
    with pytest.raises(ValueError, match="Контрольная сумма"):
        load_index(tmp_path)


def test_missing_vectors_rejected(tmp_path, sample_index):
    """Индекс без файла векторов отклоняется при загрузке.

    Args:
        tmp_path: Временный каталог теста.
        sample_index: Индекс трёх документов с готовыми векторами.
    """
    save_index(tmp_path, sample_index)
    (generation_path(tmp_path) / "vectors.npy").unlink()
    with pytest.raises(ValueError, match="повреждён"):
        load_index(tmp_path)


@pytest.mark.parametrize("pointer", [{"generation": "../../outside"}, [], "bad json"])
def test_invalid_pointer_rejected(tmp_path, pointer):
    """Некорректный указатель активной версии отклоняется.

    Args:
        tmp_path: Временный каталог теста.
        pointer: Некорректное содержимое указателя CURRENT.
    """
    (tmp_path / "CURRENT").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ValueError):
        load_index(tmp_path)


def test_failed_write_keeps_previous_generation(tmp_path, sample_index, monkeypatch):
    """Сбой записи векторов сохраняет предыдущую рабочую версию индекса.

    Args:
        tmp_path: Временный каталог теста.
        sample_index: Индекс трёх документов с готовыми векторами.
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    save_index(tmp_path, sample_index)
    previous = (tmp_path / "CURRENT").read_bytes()

    def fail_save(*args, **kwargs):
        """Имитирует отказ записи из-за отсутствия места на диске.

        Args:
            *args: Неиспользуемые позиционные параметры подменяемой функции.
            **kwargs: Неиспользуемые именованные параметры совместимого вызова.
        """
        raise OSError("disk full")

    monkeypatch.setattr("semantic_study_search.storage.index.np.save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        save_index(tmp_path, sample_index)
    assert (tmp_path / "CURRENT").read_bytes() == previous
    assert load_index(tmp_path).chunks == sample_index.chunks


def test_failed_pointer_switch_keeps_previous_generation(tmp_path, sample_index, monkeypatch):
    """Сбой переключения указателя сохраняет предыдущую рабочую версию индекса.

    Args:
        tmp_path: Временный каталог теста.
        sample_index: Индекс трёх документов с готовыми векторами.
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    save_index(tmp_path, sample_index)
    previous = (tmp_path / "CURRENT").read_bytes()

    def fail_replace(*args, **kwargs):
        """Имитирует сбой переключения активной версии индекса.

        Args:
            *args: Неиспользуемые позиционные параметры подменяемой функции.
            **kwargs: Неиспользуемые именованные параметры совместимого вызова.
        """
        raise OSError("switch interrupted")

    monkeypatch.setattr("semantic_study_search.storage.index.os.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        save_index(tmp_path, sample_index)
    assert (tmp_path / "CURRENT").read_bytes() == previous
    np.testing.assert_array_equal(load_index(tmp_path).vectors, sample_index.vectors)


def test_unsupported_format_and_pipeline_versions_rejected(tmp_path, sample_index):
    """Неподдерживаемые версии формата и обработки индекса отклоняются.

    Args:
        tmp_path: Временный каталог теста.
        sample_index: Индекс трёх документов с готовыми векторами.
    """
    save_index(tmp_path, sample_index)
    edit_metadata(tmp_path, "schema_version", 999)
    with pytest.raises(ValueError, match="формата"):
        load_index(tmp_path)
    save_index(tmp_path, sample_index)
    edit_metadata(tmp_path, "config", {**sample_index.config.__dict__, "chunker_version": 999})
    with pytest.raises(ValueError, match="обработки"):
        load_index(tmp_path)


def test_wrong_dimensions_and_object_arrays_rejected(tmp_path, sample_index):
    """Матрица другой размерности и массив Python-объектов отклоняются.

    Args:
        tmp_path: Временный каталог теста.
        sample_index: Индекс трёх документов с готовыми векторами.
    """
    save_index(tmp_path, sample_index)
    path = generation_path(tmp_path) / "vectors.npy"
    for array in (np.ones((3, 7), dtype=np.float32), np.array([{"unsafe": "data"}], dtype=object)):
        np.save(path, array)
        edit_metadata(tmp_path, "vectors_sha256", file_digest(path))
        with pytest.raises(ValueError):
            load_index(tmp_path)


def test_wrong_model_revision_rejected_before_query(sample_index, encoder):
    """Несовпадающая ревизия модели отклоняется до кодирования вопроса.

    Args:
        sample_index: Индекс трёх документов с готовыми векторами.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    encoder.revision = "different"
    with pytest.raises(ValueError, match="ревизия"):
        SearchEngine.from_index(sample_index, encoder)
    assert encoder.query_calls == 0


def test_invalid_chunk_mapping_cannot_be_saved(tmp_path, sample_index):
    """Фрагмент с текстом, отличным от исходного документа, не сохраняется.

    Args:
        tmp_path: Временный каталог теста.
        sample_index: Индекс трёх документов с готовыми векторами.
    """
    chunks = (replace(sample_index.chunks[0], text="Wrong source"), *sample_index.chunks[1:])
    with pytest.raises(ValueError, match="исходному документу"):
        save_index(tmp_path, replace(sample_index, chunks=chunks))
    assert not (tmp_path / "CURRENT").exists()


def test_corpus_fingerprint_covers_text_metadata_and_order(sample_index):
    """Отпечаток корпуса меняется при изменении текста, метаданных или порядка документов.

    Args:
        sample_index: Индекс трёх документов с готовыми векторами.
    """
    documents = sample_index.documents
    expected = corpus_fingerprint(documents)
    assert corpus_fingerprint(tuple(reversed(documents))) != expected
    for change in ({"text": "other"}, {"title": "other"}, {"source_url": "https://other.org"}):
        assert corpus_fingerprint((replace(documents[0], **change), *documents[1:])) != expected
