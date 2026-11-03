"""Проверка порога сходства и чтение его калиброванного значения."""

import json
import math
from importlib.resources import files


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
    """Читает порог, подобранный на калибровочных вопросах проекта.

    Returns:
        Порог по умолчанию для API и CLI.
    """
    resource = files("semantic_study_search").joinpath("resources/relevance_policy.json")
    return validate_min_score(json.loads(resource.read_text(encoding="utf-8"))["min_score"])
