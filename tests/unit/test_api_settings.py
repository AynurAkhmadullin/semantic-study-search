"""Проверки настроек API и отказа при отсутствии индекса."""

from pathlib import Path

import pytest

from semantic_study_search.api.service import load_service
from semantic_study_search.api.settings import APISettings


def test_settings_from_environment(monkeypatch):
    """Настройки API читаются из переменных окружения.

    Args:
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    monkeypatch.setenv("SSS_INDEX_DIR", "custom/index")
    monkeypatch.setenv("SSS_MODEL_CACHE", "custom/cache")
    monkeypatch.setenv("SSS_OFFLINE", "true")
    assert APISettings.from_env() == APISettings(Path("custom/index"), Path("custom/cache"), True)


def test_invalid_settings_are_rejected(monkeypatch):
    """Некорректный режим загрузки и пустой путь индекса отклоняются.

    Args:
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    monkeypatch.setenv("SSS_OFFLINE", "unknown")
    with pytest.raises(ValueError, match="SSS_OFFLINE"):
        APISettings.from_env()
    monkeypatch.setenv("SSS_OFFLINE", "0")
    monkeypatch.setenv("SSS_INDEX_DIR", " ")
    with pytest.raises(ValueError, match="SSS_INDEX_DIR"):
        APISettings.from_env()


def test_missing_index_rejected_before_model_load(tmp_path, monkeypatch):
    """Отсутствующий индекс отклоняется до загрузки модели.

    Args:
        tmp_path: Временный каталог теста.
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    from unittest.mock import Mock

    model = Mock(side_effect=AssertionError("Model should not load"))
    monkeypatch.setattr("semantic_study_search.api.service.E5Encoder", model)
    with pytest.raises(FileNotFoundError):
        load_service(APISettings(index_dir=tmp_path))
    model.assert_not_called()


@pytest.mark.parametrize("value", ["nan", "inf", "1.01", "-1.01", "", "abc"])
def test_invalid_score_environment(monkeypatch, value):
    """Некорректный порог окружения отклоняется с указанием настройки.

    Args:
        monkeypatch: Фикстура подмены окружения.
        value: Недопустимое значение порога.
    """
    monkeypatch.setenv("SSS_MIN_SCORE", value)
    with pytest.raises(ValueError, match="SSS_MIN_SCORE"):
        APISettings.from_env()


def test_score_environment_override(monkeypatch):
    """Порог API можно переопределить через окружение.

    Args:
        monkeypatch: Фикстура подмены окружения.
    """
    monkeypatch.setenv("SSS_MIN_SCORE", "0.9")
    assert APISettings.from_env().min_score == 0.9
