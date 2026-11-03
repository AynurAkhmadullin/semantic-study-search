"""Настройки процесса поискового API."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from semantic_study_search.search.relevance import default_min_score, validate_min_score


@dataclass(frozen=True)
class APISettings:
    """Хранит пути библиотеки, режим загрузки и порог сходства."""

    index_dir: Path = Path("data/processed/library")
    model_cache: Path | None = None
    offline: bool = False
    min_score: float = field(default_factory=default_min_score)

    def __post_init__(self) -> None:
        """Проверяет порог до загрузки модели."""
        validate_min_score(self.min_score)

    @classmethod
    def from_env(cls) -> "APISettings":
        """Читает настройки API из переменных окружения SSS.

        Returns:
            Настройки для запуска поискового сервера.
        """
        offline = os.environ.get("SSS_OFFLINE", "0").strip().lower()
        if offline not in {"0", "1", "true", "false"}:
            raise ValueError("SSS_OFFLINE должен быть 0, 1, true или false.")
        cache = os.environ.get("SSS_MODEL_CACHE", "").strip()
        directory = os.environ.get("SSS_INDEX_DIR", "data/processed/library").strip()
        if not directory:
            raise ValueError("SSS_INDEX_DIR не должен быть пустым.")
        score = os.environ.get("SSS_MIN_SCORE")
        try:
            min_score = default_min_score() if score is None else validate_min_score(float(score))
        except ValueError as error:
            raise ValueError("SSS_MIN_SCORE должен быть конечным числом от -1 до 1.") from error
        return cls(
            Path(directory), Path(cache) if cache else None, offline in {"1", "true"}, min_score
        )
