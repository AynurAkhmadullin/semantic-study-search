"""Проверки поиска из терминала по JSON-корпусу."""

import json

import numpy as np
import pytest

from semantic_study_search.cli import main

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus_file(tmp_path):
    """Записывает тестовый корпус из трёх документов.

    Args:
        tmp_path: Временный каталог теста.

    Returns:
        Путь к созданному корпусу.
    """
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "documents": [
                    {
                        "id": str(i),
                        "title": f"Материал {i}",
                        "text": f"Текст номер {i}",
                        "source_url": f"https://example.org/{i}",
                    }
                    for i in range(3)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_model(monkeypatch, encoder):
    """Подменяет загрузку E5 тестовым кодировщиком.

    Args:
        monkeypatch: Фикстура для временной подмены объектов и окружения.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.

    Returns:
        Подставленный тестовый кодировщик.
    """
    import semantic_study_search.search.encoder as module

    monkeypatch.setattr(module, "E5Encoder", lambda **kwargs: encoder)
    return encoder


def test_cli_json_connects_loading_chunking_and_search(corpus_file, fake_model, capsys):
    """CLI загружает корпус, разбивает его и возвращает результаты в JSON.

    Args:
        corpus_file: Путь к тестовому JSON-корпусу.
        fake_model: Фикстура подмены E5 тестовым кодировщиком.
        capsys: Перехват стандартного вывода и ошибок.
    """
    status = main(
        ["вопрос", "--corpus", str(corpus_file), "--top-k", "2", "--json", "--min-score", "-1"]
    )
    output = capsys.readouterr()
    assert status == 0
    results = json.loads(output.out)
    assert len(results) == 2
    assert results[0]["chunk"]["document_id"] == "0"
    assert results[0]["chunk"]["source_url"] == "https://example.org/0"
    assert results[0]["chunk"]["text"] == "Текст номер 0"
    assert results[1]["score"] == pytest.approx(1 / np.sqrt(2))
    assert "Индексация" in output.err


def test_cli_text(corpus_file, fake_model, capsys):
    """Текстовый вывод CLI содержит название, источник и сходство.

    Args:
        corpus_file: Путь к тестовому JSON-корпусу.
        fake_model: Фикстура подмены E5 тестовым кодировщиком.
        capsys: Перехват стандартного вывода и ошибок.
    """
    assert main(["вопрос", "--corpus", str(corpus_file)]) == 0
    output = capsys.readouterr().out
    assert "Материал 0" in output
    assert "https://example.org/0" in output
    assert "косинусная мера" in output


@pytest.mark.parametrize("arguments", [[" "], ["вопрос", "--top-k", "0"]])
def test_bad_cli_input_does_not_load_model(arguments, monkeypatch, capsys):
    """Некорректные аргументы CLI отклоняются до загрузки модели.

    Args:
        arguments: Некорректные аргументы команды.
        monkeypatch: Фикстура для временной подмены объектов и окружения.
        capsys: Перехват стандартного вывода и ошибок.
    """
    from unittest.mock import Mock

    model = Mock(side_effect=AssertionError("Model must not load"))
    monkeypatch.setattr("semantic_study_search.search.encoder.E5Encoder", model)
    assert main(arguments) == 1
    assert "Ошибка:" in capsys.readouterr().err
    model.assert_not_called()


def test_cli_explains_empty_results(corpus_file, fake_model, capsys):
    """CLI сообщает о пустой выдаче и сохраняет успешный код завершения.

    Args:
        corpus_file: Тестовый корпус.
        fake_model: Подменённый кодировщик E5.
        capsys: Перехват вывода команды.
    """
    fake_model.query = np.array([-1, 0], dtype=np.float32)
    assert main(["другая тема", "--corpus", str(corpus_file)]) == 0
    assert "не найдены" in capsys.readouterr().out
    assert main(["другая тема", "--corpus", str(corpus_file), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_missing_corpus_is_reported(tmp_path, fake_model, capsys):
    """CLI сообщает об отсутствующем файле корпуса.

    Args:
        tmp_path: Временный каталог теста.
        fake_model: Фикстура подмены E5 тестовым кодировщиком.
        capsys: Перехват стандартного вывода и ошибок.
    """
    assert main(["вопрос", "--corpus", str(tmp_path / "missing.json")]) == 1
    assert "Ошибка:" in capsys.readouterr().err
