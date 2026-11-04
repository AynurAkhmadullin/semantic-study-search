"""Загрузка и проверка корпуса материалов."""

import json
from pathlib import Path
from urllib.parse import urlparse

from semantic_study_search.domain import Document
from semantic_study_search.sources import uploaded_source_id


def load_corpus(path: Path) -> list[Document]:
    """Загружает и проверяет документы из JSON-корпуса.

    Args:
        path: Путь к предоставленному пользователем корпусу.

    Returns:
        Проверенные документы в порядке исходного корпуса.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_documents(payload)


def parse_documents(payload: object) -> list[Document]:
    """Проверяет структуру корпуса и создаёт документы.

    Args:
        payload: Прочитанные данные JSON-корпуса.

    Returns:
        Проверенные документы с уникальными идентификаторами.
    """
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Ожидается JSON-корпус с schema_version = 1.")
    rows = payload.get("documents")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Корпус должен содержать непустой список documents.")
    documents = []
    seen = set()
    for row in rows:
        keys = ("id", "title", "source_url", "text")
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip() for key in keys
        ):
            raise ValueError("У документа должны быть непустые id, title, source_url и text.")
        document = Document(**{key: row[key].strip() for key in keys})
        if document.id in seen:
            raise ValueError(f"Повторяющийся id документа: {document.id}")
        url = urlparse(document.source_url)
        if uploaded_source_id(document.source_url) is None and (
            url.scheme not in {"http", "https"} or not url.netloc
        ):
            raise ValueError(f"Некорректная ссылка на источник: {document.id}")
        seen.add(document.id)
        documents.append(document)
    return documents
