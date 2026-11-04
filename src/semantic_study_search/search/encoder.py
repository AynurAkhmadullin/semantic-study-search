"""Вычисление векторов текста предобученной моделью E5 на CPU."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from semantic_study_search.ingestion.chunking import PASSAGE_PREFIX, Tokenizer, token_count
from semantic_study_search.limits import MAX_QUERY_CHARACTERS

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MAX_TOKENS = 512
EMBEDDING_DIMENSION = 384
QUERY_PREFIX = "query: "


def validate_query(query: str) -> str:
    """Удаляет пробелы по краям запроса и проверяет его длину.

    Args:
        query: Поисковый вопрос.

    Returns:
        Непустой запрос допустимой длины без крайних пробелов.
    """
    query = query.strip()
    if not query:
        raise ValueError("Введите непустой поисковый запрос.")
    if len(query) > MAX_QUERY_CHARACTERS:
        raise ValueError(f"Запрос должен содержать не более {MAX_QUERY_CHARACTERS} символов.")
    return query


class Encoder(Protocol):
    tokenizer: Tokenizer
    model_id: str
    revision: str
    dimension: int

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def encode_query(self, query: str) -> NDArray[np.float32]: ...


class E5Encoder:
    """Преобразует тексты и запросы в векторы с помощью E5."""

    model_id = MODEL_ID
    revision = MODEL_REVISION
    dimension = EMBEDDING_DIMENSION

    def __init__(self, *, cache_folder: Path | None = None, offline: bool = False):
        """Загружает закреплённую версию E5 для вычислений на CPU.

        Args:
            cache_folder: Каталог модели; None использует стандартный кэш Hugging Face.
            offline: Загружать модель только из кэша без обращения к сети.
        """
        # Heavy imports happen only when a real model is requested.
        import torch
        from sentence_transformers import SentenceTransformer

        # Small CPU batches suffer from oversubscription on many-core machines.
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self._model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            device="cpu",
            cache_folder=str(cache_folder) if cache_folder is not None else None,
            local_files_only=offline,
            trust_remote_code=False,
            model_kwargs={"use_safetensors": True},
        )
        self._model.max_seq_length = MAX_TOKENS
        self.tokenizer = self._model.tokenizer

    def _encode(self, texts: Sequence[str], prefix: str) -> NDArray[np.float32]:
        """Вычисляет векторы текстов с заданным префиксом E5.

        Args:
            texts: Непустые тексты для кодирования.
            prefix: Префикс перед каждым текстом.

        Returns:
            Матрица нормализованных векторов в порядке входных текстов.
        """
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Для вычисления векторов необходим непустой текст.")
        inputs = [prefix + text for text in texts]
        if any(token_count(self.tokenizer, text) > MAX_TOKENS for text in inputs):
            raise ValueError(f"Текст с префиксом превышает лимит модели: {MAX_TOKENS} токенов.")
        return np.asarray(
            self._model.encode(
                inputs,
                prompt="",
                batch_size=8,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Вычисляет векторы фрагментов с префиксом passage.

        Args:
            texts: Тексты фрагментов.

        Returns:
            Матрица нормализованных векторов в порядке фрагментов.
        """
        return self._encode(texts, PASSAGE_PREFIX)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """Вычисляет вектор поискового запроса с префиксом query.

        Args:
            query: Поисковый вопрос.

        Returns:
            Нормализованный вектор запроса.
        """
        return self._encode([validate_query(query)], QUERY_PREFIX)[0]
