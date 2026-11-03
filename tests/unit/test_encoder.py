"""Проверки префиксов E5 и ограничений длины без загрузки весов."""

from unittest.mock import Mock

import numpy as np
import pytest

from semantic_study_search.search.encoder import E5Encoder


@pytest.fixture
def e5(tokenizer):
    """Создаёт E5 с тестовым токенизатором и подменённым вычислением векторов.

    Args:
        tokenizer: Тестовый токенизатор по пробельным словам.

    Returns:
        Кодировщик E5 без загруженных весов.
    """
    encoder = object.__new__(E5Encoder)
    encoder.tokenizer = tokenizer
    encoder._model = Mock()
    encoder._model.encode.return_value = np.ones((1, 384), dtype=np.float32)
    return encoder


def test_e5_prefixes_and_normalization_are_requested(e5):
    """Кодировщик передаёт префиксы E5 и запрашивает нормализацию векторов.

    Args:
        e5: Кодировщик E5 с подменённым вычислением векторов.
    """
    e5.encode_passages(["Учебный текст"])
    args, kwargs = e5._model.encode.call_args
    assert args[0] == ["passage: Учебный текст"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["prompt"] == ""
    vector = e5.encode_query("  Вопрос  ")
    assert e5._model.encode.call_args.args[0] == ["query: Вопрос"]
    assert vector.shape == (384,)


def test_overlong_passage_never_silently_truncated(e5):
    """Слишком длинный фрагмент отклоняется до вычисления вектора.

    Args:
        e5: Кодировщик E5 с подменённым вычислением векторов.
    """
    with pytest.raises(ValueError, match="512 токенов"):
        e5.encode_passages(["слово " * 512])
    e5._model.encode.assert_not_called()


def test_query_token_limit_includes_prefix_and_special_tokens(e5):
    """Лимит вопроса учитывает префикс и специальные токены.

    Args:
        e5: Кодировщик E5 с подменённым вычислением векторов.
    """
    with pytest.raises(ValueError, match="512 токенов"):
        e5.encode_query("я " * 510)
    e5._model.encode.assert_not_called()
