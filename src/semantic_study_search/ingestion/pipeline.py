"""Подготовка и повторное использование индекса библиотеки."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from semantic_study_search.domain import Document
from semantic_study_search.ingestion.chunking import chunk_documents
from semantic_study_search.ingestion.corpus import parse_documents
from semantic_study_search.search.encoder import Encoder
from semantic_study_search.search.engine import SearchEngine
from semantic_study_search.storage.index import (
    IndexConfig,
    StoredIndex,
    corpus_fingerprint,
    load_index,
    save_index,
)


def prepare_index(
    path: Path,
    documents: Sequence[Document],
    encoder_factory: Callable[[], Encoder],
    *,
    config: IndexConfig,
    force: bool = False,
) -> tuple[StoredIndex, bool]:
    """Подготавливает индекс или использует сохранённую версию того же корпуса.

    Args:
        path: Каталог индекса.
        documents: Документы общей библиотеки.
        encoder_factory: Функция загрузки модели при необходимости пересчёта.
        config: Версия модели и параметры разбиения.
        force: Пересчитать индекс даже при совпадении корпуса и настроек.

    Returns:
        Индекс и признак создания новой версии.
    """
    config.validate()
    documents = tuple(documents)
    parse_documents({"schema_version": 1, "documents": [asdict(doc) for doc in documents]})
    if not force and (path / "CURRENT").exists():
        previous = load_index(path)
        if previous.config == config and corpus_fingerprint(
            previous.documents
        ) == corpus_fingerprint(documents):
            return previous, False
    encoder = encoder_factory()
    config.require_encoder(encoder)
    chunks = chunk_documents(
        documents, encoder.tokenizer, max_tokens=config.max_tokens, overlap=config.overlap
    )
    engine = SearchEngine(chunks, encoder)
    index = StoredIndex(documents, tuple(chunks), engine.vectors, config)
    save_index(path, index)
    return index, True
