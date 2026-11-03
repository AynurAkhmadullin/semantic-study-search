"""Импорт локальных учебных файлов по списку источников."""

import json
import re
import warnings
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from markdown_it import MarkdownIt
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from semantic_study_search.domain import Document
from semantic_study_search.ingestion.corpus import parse_documents


def clean_text(text: str) -> str:
    """Убирает служебные символы и лишние пробелы из текста.

    Args:
        text: Извлечённый текст документа.

    Returns:
        Текст с сохранёнными абзацами.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace("\ufeff", "").replace("\u00ad", "")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def markdown_text(text: str) -> str:
    """Извлекает текст и примеры кода из Markdown.

    Args:
        text: Содержимое Markdown-файла.

    Returns:
        Текст без разметки, изображений и адресов ссылок.
    """
    # YAML front matter contains metadata, not learning material.
    text = re.sub(r"\A---\r?\n.*?\r?\n(?:---|\.\.\.)\s*\r?\n", "", text, flags=re.DOTALL)
    parts = []
    for token in MarkdownIt("commonmark").enable("table").parse(text):
        if token.type in {"fence", "code_block"}:
            parts.append(token.content)
        elif token.type == "inline":
            parts.append(
                "".join(
                    "\n" if child.type in {"softbreak", "hardbreak"} else child.content
                    for child in token.children or []
                    if child.type in {"text", "code_inline", "softbreak", "hardbreak"}
                )
            )
    return clean_text("\n\n".join(parts))


def read_pdf(path: Path, metadata: dict[str, str]) -> list[dict[str, str]]:
    """Извлекает текстовые страницы PDF с отдельными ссылками на страницы.

    Args:
        path: Путь к PDF-файлу.
        metadata: Идентификатор, название и ссылка на документ.

    Returns:
        Записи страниц с извлечённым текстом.
    """
    rows = []
    try:
        with path.open("rb") as stream:
            reader = PdfReader(stream)
            if reader.is_encrypted:
                raise ValueError(f"PDF защищён паролем: {path.name}")
            for number, page in enumerate(reader.pages, 1):
                text = clean_text(page.extract_text() or "")
                if not text:
                    warnings.warn(
                        f"{path.name}: страница {number} не содержит текста; OCR не выполняется.",
                        stacklevel=2,
                    )
                    continue
                url = urlsplit(metadata["source_url"])
                rows.append(
                    {
                        "id": f"{metadata['id']}:page-{number}",
                        "title": f"{metadata['title']} — стр. {number}",
                        "source_url": urlunsplit(url._replace(fragment=f"page={number}")),
                        "text": text,
                    }
                )
    except PyPdfError as error:
        raise ValueError(f"Не удалось прочитать PDF {path.name}: {error}") from error
    if not rows:
        raise ValueError(f"В PDF {path.name} нет текстового слоя; требуется OCR.")
    return rows


def load_manifest(path: Path) -> list[Document]:
    """Загружает материалы TXT, Markdown и PDF по JSON-списку источников.

    Args:
        path: Путь к списку с полями id, title, source_url и path.

    Returns:
        Очищенные документы и текстовые страницы PDF.
    """
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Ожидается список источников с schema_version = 1.")
    rows = payload.get("documents")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Список источников documents не должен быть пустым.")
    imported = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("id", "title", "source_url", "path")
        ):
            raise ValueError("Источник должен содержать id, title, source_url и path.")
        metadata = {key: row[key].strip() for key in ("id", "title", "source_url")}
        parse_documents({"schema_version": 1, "documents": [metadata | {"text": "validation"}]})
        if metadata["id"] in seen:
            raise ValueError(f"Повторяющийся id источника: {metadata['id']}")
        seen.add(metadata["id"])
        source = path.parent / row["path"]
        extension = source.suffix.lower()
        if extension == ".pdf":
            imported.extend(read_pdf(source, metadata))
        elif extension in {".txt", ".md", ".markdown"}:
            text = source.read_text(encoding="utf-8-sig")
            text = clean_text(text) if extension == ".txt" else markdown_text(text)
            if not text:
                raise ValueError(f"Файл не содержит текста: {source.name}")
            imported.append(metadata | {"text": text})
        else:
            raise ValueError(f"Формат {extension} не поддерживается: {source.name}")
    return parse_documents({"schema_version": 1, "documents": imported})
