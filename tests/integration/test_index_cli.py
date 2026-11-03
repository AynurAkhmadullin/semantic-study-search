"""Проверки команд подготовки индекса и поиска по сохранённой библиотеке."""

import json
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from semantic_study_search.cli import main as search_main
from semantic_study_search.index_cli import main as index_main
from semantic_study_search.search.encoder import EMBEDDING_DIMENSION, MODEL_ID, MODEL_REVISION
from semantic_study_search.storage.index import load_index, save_index

pytestmark = pytest.mark.integration


@pytest.fixture
def cli_model(monkeypatch, encoder):
    """Подменяет E5 кодировщиком с совместимыми настройками и размерностью.

    Args:
        monkeypatch: Фикстура для временной подмены объектов и окружения.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.

    Returns:
        Кодировщик и фабрика со счётчиком загрузок.
    """
    encoder.model_id = MODEL_ID
    encoder.revision = MODEL_REVISION
    encoder.dimension = EMBEDDING_DIMENSION
    encoder.passages = np.pad(encoder.passages, ((0, 0), (0, EMBEDDING_DIMENSION - 2)))
    encoder.query = np.pad(encoder.query, (0, EMBEDDING_DIMENSION - 2))
    factory = Mock(return_value=encoder)
    monkeypatch.setattr("semantic_study_search.search.encoder.E5Encoder", factory)
    return encoder, factory


@pytest.fixture
def local_manifest(tmp_path):
    """Создаёт три Markdown-материала и список их источников.

    Args:
        tmp_path: Временный каталог теста.

    Returns:
        Путь к списку источников.
    """
    rows = []
    for number in range(3):
        (tmp_path / f"lesson{number}.md").write_text(
            f"# Материал {number}\n\nТекст урока.", encoding="utf-8"
        )
        rows.append(
            {
                "id": str(number),
                "title": f"Материал {number}",
                "source_url": f"https://example.org/{number}",
                "path": f"lesson{number}.md",
            }
        )
    path = tmp_path / "library.json"
    path.write_text(json.dumps({"schema_version": 1, "documents": rows}), encoding="utf-8")
    return path


def test_build_reuse_and_search_from_disk(tmp_path, local_manifest, cli_model, capsys):
    """CLI повторно использует индекс и ищет без пересчёта векторов документов.

    Args:
        tmp_path: Временный каталог теста.
        local_manifest: Путь к списку трёх локальных Markdown-материалов.
        cli_model: Тестовый кодировщик и подменённая фабрика его загрузки.
        capsys: Перехват стандартного вывода и ошибок.
    """
    encoder, factory = cli_model
    directory = tmp_path / "index"
    arguments = ["--manifest", str(local_manifest), "--index-dir", str(directory)]
    assert index_main(arguments) == 0
    assert "Индекс сохранён" in capsys.readouterr().out
    assert index_main(arguments) == 0
    assert "пересчёт не требуется" in capsys.readouterr().out
    assert factory.call_count == 1
    encoder.encode_passages = Mock(side_effect=AssertionError("Documents must not be encoded"))
    assert (
        search_main(
            ["вопрос", "--index-dir", str(directory), "--json", "--top-k", "2", "--min-score", "-1"]
        )
        == 0
    )
    output = capsys.readouterr()
    results = json.loads(output.out)
    assert results[0]["chunk"]["document_id"] == "0"
    assert len(results) == 2
    assert "Загружен индекс" in output.err
    encoder.encode_passages.assert_not_called()


def test_missing_index_fails_before_model_load(tmp_path, cli_model, capsys):
    """CLI отклоняет отсутствующий индекс до загрузки модели.

    Args:
        tmp_path: Временный каталог теста.
        cli_model: Тестовый кодировщик и подменённая фабрика его загрузки.
        capsys: Перехват стандартного вывода и ошибок.
    """
    _, factory = cli_model
    assert search_main(["вопрос", "--index-dir", str(tmp_path / "missing")]) == 1
    assert "semantic-study-index" in capsys.readouterr().err
    factory.assert_not_called()


def test_incompatible_model_fails_before_model_load(tmp_path, local_manifest, cli_model, capsys):
    """CLI отклоняет индекс другой ревизии до загрузки модели.

    Args:
        tmp_path: Временный каталог теста.
        local_manifest: Путь к списку трёх локальных Markdown-материалов.
        cli_model: Тестовый кодировщик и подменённая фабрика его загрузки.
        capsys: Перехват стандартного вывода и ошибок.
    """
    _, factory = cli_model
    directory = tmp_path / "index"
    assert index_main(["--manifest", str(local_manifest), "--index-dir", str(directory)]) == 0
    index = load_index(directory)
    save_index(directory, replace(index, config=replace(index.config, model_revision="old")))
    factory.reset_mock()
    assert search_main(["вопрос", "--index-dir", str(directory)]) == 1
    assert "другой моделью" in capsys.readouterr().err
    factory.assert_not_called()


def test_bad_build_settings_fail_before_model_load(tmp_path, cli_model, capsys):
    """CLI проверяет параметры разбиения до загрузки модели.

    Args:
        tmp_path: Временный каталог теста.
        cli_model: Тестовый кодировщик и подменённая фабрика его загрузки.
        capsys: Перехват стандартного вывода и ошибок.
    """
    _, factory = cli_model
    assert index_main(["--chunk-tokens", "513", "--index-dir", str(tmp_path)]) == 1
    assert "Некорректные" in capsys.readouterr().err
    factory.assert_not_called()


def test_example_manifest_can_extend_demo_without_duplicate_ids():
    """Примеры дополняют встроенный корпус без повторяющихся идентификаторов."""
    from pathlib import Path

    from semantic_study_search.ingestion.corpus import load_corpus
    from semantic_study_search.ingestion.files import load_manifest

    root = Path(__file__).parents[2]
    documents = load_corpus() + load_manifest(root / "examples/library.json")
    assert len(documents) == 10
    assert len({doc.id for doc in documents}) == 10
