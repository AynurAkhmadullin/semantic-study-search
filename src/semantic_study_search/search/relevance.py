"""Проверка порога сходства и его начальное значение."""

import math

DEFAULT_MIN_SCORE = 0.8206596076488495


def validate_min_score(value: float) -> float:
    """Проверяет допустимый порог косинусного сходства.

    Args:
        value: Порог для включения фрагмента в выдачу.

    Returns:
        Конечное число от -1 до 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Порог сходства должен быть числом от -1 до 1.")
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Порог сходства должен быть конечным числом от -1 до 1.")
    return float(value)


def default_min_score() -> float:
    """Возвращает начальный порог, требующий проверки на материалах пользователя.

    Returns:
        Порог по умолчанию для API и CLI.
    """
    return DEFAULT_MIN_SCORE
