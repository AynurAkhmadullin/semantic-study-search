"""Сохранение согласованных версий библиотеки в SQLite и NumPy."""

import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from semantic_study_search.domain import Chunk, Document
from semantic_study_search.ingestion.chunking import CHUNKER_VERSION, PASSAGE_PREFIX
from semantic_study_search.ingestion.corpus import parse_documents
from semantic_study_search.search.encoder import (
    EMBEDDING_DIMENSION,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_PREFIX,
    Encoder,
)

SCHEMA_VERSION = 1
DEFAULT_INDEX_DIR = Path("data/processed/library")


@dataclass(frozen=True)
class IndexConfig:
    """Хранит версию модели и параметры подготовки фрагментов."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dimension: int = EMBEDDING_DIMENSION
    max_tokens: int = 320
    overlap: int = 48
    chunker_version: int = CHUNKER_VERSION
    encoder_version: int = 1
    model_max_tokens: int = MAX_TOKENS
    passage_prefix: str = PASSAGE_PREFIX
    query_prefix: str = QUERY_PREFIX

    def validate(self) -> None:
        """Проверяет параметры индекса и поддерживаемые версии обработки."""
        if not all(
            isinstance(value, str) and value for value in (self.model_id, self.model_revision)
        ):
            raise ValueError("В индексе отсутствует название или ревизия модели.")
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("Размерность векторов должна быть положительной.")
        if (
            type(self.max_tokens) is not int
            or type(self.overlap) is not int
            or not 1 <= self.max_tokens <= MAX_TOKENS
            or not 0 <= self.overlap < self.max_tokens
        ):
            raise ValueError("Некорректные размер фрагмента или перекрытие.")
        if (
            self.chunker_version != CHUNKER_VERSION
            or self.encoder_version != 1
            or self.model_max_tokens != MAX_TOKENS
            or self.passage_prefix != PASSAGE_PREFIX
            or self.query_prefix != QUERY_PREFIX
        ):
            raise ValueError("Версия обработки индекса несовместима; пересоздайте индекс.")

    def require_encoder(self, encoder: Encoder) -> None:
        """Проверяет соответствие модели сохранённому индексу.

        Args:
            encoder: Модель для вычисления вектора запроса.
        """
        self.validate()
        if (self.model_id, self.model_revision, self.dimension) != (
            encoder.model_id,
            encoder.revision,
            encoder.dimension,
        ):
            raise ValueError("Модель или её ревизия не соответствует индексу; пересоздайте индекс.")


@dataclass(frozen=True)
class StoredIndex:
    """Объединяет документы, фрагменты, векторы и настройки индекса."""

    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]
    vectors: NDArray[np.float32]
    config: IndexConfig

    def validate(self) -> None:
        """Проверяет соответствие документов, фрагментов и строк матрицы."""
        self.config.validate()
        parse_documents({"schema_version": 1, "documents": [asdict(doc) for doc in self.documents]})
        documents = {doc.id: doc for doc in self.documents}
        if not self.chunks or len({chunk.id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("Индекс должен содержать фрагменты с уникальными идентификаторами.")
        if {chunk.document_id for chunk in self.chunks} != set(documents):
            raise ValueError("Документы и фрагменты индекса не соответствуют друг другу.")
        for chunk in self.chunks:
            doc = documents[chunk.document_id]
            if (
                type(chunk.start) is not int
                or type(chunk.end) is not int
                or not 0 <= chunk.start < chunk.end <= len(doc.text)
                or chunk.text != doc.text[chunk.start : chunk.end]
                or chunk.title != doc.title
                or chunk.source_url != doc.source_url
            ):
                raise ValueError(f"Фрагмент не соответствует исходному документу: {chunk.id}")
        if (
            self.vectors.dtype != np.float32
            or self.vectors.shape != (len(self.chunks), self.config.dimension)
            or not np.isfinite(self.vectors).all()
            or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1.0, atol=1e-5)
        ):
            raise ValueError("Матрица индекса должна содержать нормализованные векторы фрагментов.")


def corpus_fingerprint(documents: tuple[Document, ...] | list[Document]) -> str:
    """Вычисляет контрольную сумму содержимого и порядка документов.

    Args:
        documents: Документы подготовленной библиотеки.

    Returns:
        Контрольная сумма SHA-256.
    """
    data = json.dumps([asdict(doc) for doc in documents], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """Вычисляет контрольную сумму файла.

    Args:
        path: Путь к файлу индекса.

    Returns:
        Контрольная сумма SHA-256.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_index(path: Path, index: StoredIndex) -> None:
    """Сохраняет новую версию индекса и переключает указатель после успешной записи.

    Args:
        path: Каталог библиотеки.
        index: Проверяемые документы, фрагменты и векторы.
    """
    index.validate()
    generation = uuid4().hex
    directory = path / "versions" / generation
    directory.mkdir(parents=True)
    vectors_path = directory / "vectors.npy"
    with vectors_path.open("wb") as stream:
        np.save(stream, index.vectors, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(index.config),
        "corpus_sha256": corpus_fingerprint(index.documents),
        "vectors_sha256": file_digest(vectors_path),
    }
    database = directory / "library.sqlite3"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents (
                position INTEGER UNIQUE NOT NULL, id TEXT PRIMARY KEY,
                title TEXT NOT NULL, source_url TEXT NOT NULL, text TEXT NOT NULL
            );
            CREATE TABLE chunks (
                position INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL REFERENCES documents(id),
                text TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL
            );
        """)
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
        )
        connection.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
            [
                (i, doc.id, doc.title, doc.source_url, doc.text)
                for i, doc in enumerate(index.documents)
            ],
        )
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            [(i, c.id, c.document_id, c.text, c.start, c.end) for i, c in enumerate(index.chunks)],
        )
    pointer = json.dumps({"generation": generation, "database_sha256": file_digest(database)})
    temporary = path / f"CURRENT.{generation}.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(pointer)
        stream.flush()
        os.fsync(stream.fileno())
    # Readers continue using the old immutable generation until this single replace.
    os.replace(temporary, path / "CURRENT")


def load_index(path: Path) -> StoredIndex:
    """Загружает согласованный индекс и проверяет целостность его файлов.

    Args:
        path: Каталог сохранённой библиотеки.

    Returns:
        Проверенный индекс с настройками подготовки.
    """
    if not (path / "CURRENT").exists():
        raise FileNotFoundError("Индекс не найден. Выполните semantic-study-index.")
    try:
        pointer = json.loads((path / "CURRENT").read_text(encoding="utf-8"))
        generation = pointer["generation"]
        if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("Некорректный идентификатор версии индекса.")
        directory = path / "versions" / generation
        database = directory / "library.sqlite3"
        if file_digest(database) != pointer["database_sha256"]:
            raise ValueError("Контрольная сумма SQLite не совпадает.")
        with closing(
            sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            metadata = {
                key: json.loads(value)
                for key, value in connection.execute("SELECT key, value FROM metadata")
            }
            if metadata["schema_version"] != SCHEMA_VERSION:
                raise ValueError("Версия формата индекса не поддерживается.")
            config = IndexConfig(**metadata["config"])
            config.validate()
            documents = tuple(
                Document(*row)
                for row in connection.execute(
                    "SELECT id, title, source_url, text FROM documents ORDER BY position"
                )
            )
            rows = connection.execute("""
                SELECT c.position, c.id, c.document_id, d.title, d.source_url,
                       c.text, c.start, c.end
                FROM chunks c JOIN documents d ON c.document_id = d.id ORDER BY c.position
            """).fetchall()
            count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if len(rows) != count or [row[0] for row in rows] != list(range(count)):
                raise ValueError("Нарушен порядок фрагментов индекса.")
            chunks = tuple(Chunk(*row[1:]) for row in rows)
        vectors_path = directory / "vectors.npy"
        if file_digest(vectors_path) != metadata["vectors_sha256"]:
            raise ValueError("Контрольная сумма векторов не совпадает.")
        with vectors_path.open("rb") as stream:
            vectors = np.load(stream, allow_pickle=False)
        if not isinstance(vectors, np.ndarray):
            raise ValueError("Ожидается матрица векторов NumPy.")
        index = StoredIndex(documents, chunks, vectors, config)
        index.validate()
        if corpus_fingerprint(documents) != metadata["corpus_sha256"]:
            raise ValueError("Контрольная сумма корпуса не совпадает.")
        return index
    except (OSError, ValueError, TypeError, KeyError, EOFError, sqlite3.Error) as error:
        raise ValueError(f"Индекс повреждён или несовместим: {error}") from error
