"""Проверки пересборки, повторного использования и восстановления индекса."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from semantic_study_search.ingestion.pipeline import prepare_index
from semantic_study_search.storage.index import load_index

pytestmark = pytest.mark.integration


def test_unchanged_corpus_reuses_index_without_loading_model(
    tmp_path, library_documents, encoder, index_config
):
    """Неизменная библиотека использует готовый индекс без повторной загрузки модели.

    Args:
        tmp_path: Временный каталог теста.
        library_documents: Три тестовых документа.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        index_config: Настройки индекса для тестового кодировщика.
    """
    factory = Mock(return_value=encoder)
    first, created = prepare_index(tmp_path, library_documents, factory, config=index_config)
    pointer = (tmp_path / "CURRENT").read_bytes()
    second, reused = prepare_index(tmp_path, library_documents, factory, config=index_config)
    assert created and not reused
    factory.assert_called_once()
    assert encoder.passage_calls == 1
    assert second.chunks == first.chunks
    assert (tmp_path / "CURRENT").read_bytes() == pointer


@pytest.mark.parametrize("change", ["text", "source", "chunking", "force", "model"])
def test_changed_inputs_create_new_generation(
    tmp_path, library_documents, encoder, index_config, change
):
    """Изменения библиотеки или настроек создают новую версию индекса.

    Args:
        tmp_path: Временный каталог теста.
        library_documents: Три тестовых документа.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        index_config: Настройки индекса для тестового кодировщика.
        change: Изменяемый параметр или режим принудительной пересборки.
    """
    factory = Mock(return_value=encoder)
    prepare_index(tmp_path, library_documents, factory, config=index_config)
    previous = (tmp_path / "CURRENT").read_bytes()
    if change == "text":
        library_documents[0] = replace(library_documents[0], text="Обновлённый текст")
    elif change == "source":
        library_documents[0] = replace(library_documents[0], source_url="https://example.org/new")
    elif change == "chunking":
        index_config = replace(index_config, max_tokens=300)
    elif change == "model":
        encoder.revision = "new-revision"
        index_config = replace(index_config, model_revision=encoder.revision)
    index, created = prepare_index(
        tmp_path, library_documents, factory, config=index_config, force=change == "force"
    )
    assert created
    assert factory.call_count == 2
    assert (tmp_path / "CURRENT").read_bytes() != previous
    assert load_index(tmp_path).documents == index.documents


def test_failed_inference_keeps_working_index(tmp_path, library_documents, encoder, index_config):
    """Ошибка вычисления векторов сохраняет рабочую версию индекса.

    Args:
        tmp_path: Временный каталог теста.
        library_documents: Три тестовых документа.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        index_config: Настройки индекса для тестового кодировщика.
    """
    prepare_index(tmp_path, library_documents, lambda: encoder, config=index_config)
    previous = (tmp_path / "CURRENT").read_bytes()
    encoder.encode_passages = Mock(side_effect=RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        prepare_index(tmp_path, library_documents, lambda: encoder, config=index_config, force=True)
    assert (tmp_path / "CURRENT").read_bytes() == previous
    assert len(load_index(tmp_path).chunks) == 3


def test_duplicate_documents_rejected_before_loading_model(
    tmp_path, library_documents, index_config
):
    """Повторяющиеся документы отклоняются до загрузки модели.

    Args:
        tmp_path: Временный каталог теста.
        library_documents: Три тестовых документа.
        index_config: Настройки индекса для тестового кодировщика.
    """
    factory = Mock()
    with pytest.raises(ValueError, match="Повторяющийся"):
        prepare_index(tmp_path, [library_documents[0]] * 2, factory, config=index_config)
    factory.assert_not_called()


def test_corrupted_index_requires_explicit_force(
    tmp_path, library_documents, encoder, index_config
):
    """Повреждённый индекс восстанавливается только при принудительной пересборке.

    Args:
        tmp_path: Временный каталог теста.
        library_documents: Три тестовых документа.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        index_config: Настройки индекса для тестового кодировщика.
    """
    prepare_index(tmp_path, library_documents, lambda: encoder, config=index_config)
    (tmp_path / "CURRENT").write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_index(tmp_path, library_documents, lambda: encoder, config=index_config)
    restored, created = prepare_index(
        tmp_path, library_documents, lambda: encoder, config=index_config, force=True
    )
    assert created
    assert load_index(tmp_path).chunks == restored.chunks
