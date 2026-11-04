"""Проверки импорта TXT, Markdown и PDF из списка источников."""

import json

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from semantic_study_search.ingestion.files import clean_text, load_manifest, markdown_text


def write_manifest(tmp_path, rows):
    """Записывает список источников в JSON-файл.

    Args:
        tmp_path: Временный каталог теста.
        rows: Проверяемые записи источников.

    Returns:
        Путь к списку источников.
    """
    path = tmp_path / "library.json"
    path.write_text(json.dumps({"schema_version": 1, "documents": rows}), encoding="utf-8")
    return path


def source_row(path, *, identifier="lesson"):
    """Создаёт запись файла для списка источников.

    Args:
        path: Путь материала относительно списка источников.
        identifier: Идентификатор источника.

    Returns:
        Словарь с описанием файла.
    """
    return {
        "id": identifier,
        "title": "Урок",
        "source_url": "https://example.org/lesson",
        "path": path,
    }


def write_pdf(path, pages):
    """Создаёт PDF с текстовыми или пустыми страницами для проверки импорта.

    Args:
        path: Путь создаваемого PDF-файла.
        pages: Тексты страниц в ASCII; None создаёт пустую страницу.
    """
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        if text is not None:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): writer._add_object(font)}
                    )
                }
            )
            content = DecodedStreamObject()
            content.set_data(f"BT /F1 12 Tf 30 250 Td ({text}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(path)


def test_clean_text_preserves_paragraphs():
    """Очистка текста удаляет служебные символы и сохраняет абзацы."""
    assert clean_text("\ufeff Раз\t два\r\n\r\n\r\nтри\x00\u00ad ") == "Раз два\n\nтри"


def test_markdown_retains_meaning_and_code_without_markup():
    """Преобразование Markdown сохраняет текст и код, удаляя разметку и HTML."""
    text = "---\ntitle: hidden\n---\n# Документ\n\n**Важный** [пример](https://example.org).\n\n![рисунок](x.png)\n\n```python\nx = 1\n```\n\n<script>ignore()</script>"
    rendered = markdown_text(text)
    assert "Документ" in rendered
    assert "Важный пример." in rendered
    assert "x = 1" in rendered
    assert not any(
        value in rendered for value in ("hidden", "https://", "рисунок", "ignore", "**", "```")
    )


def test_manifest_loads_relative_utf8_files_independent_of_cwd(tmp_path, monkeypatch):
    """Пути материалов отсчитываются от списка источников независимо от рабочей папки.

    Args:
        tmp_path: Временный каталог теста.
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    (tmp_path / "lesson.txt").write_text("\ufeffТекст\tурока.", encoding="utf-8")
    (tmp_path / "lesson.md").write_text("# Раздел\n\n**Пояснение**.", encoding="utf-8")
    path = write_manifest(
        tmp_path, [source_row("lesson.txt"), source_row("lesson.md", identifier="md")]
    )
    monkeypatch.chdir(tmp_path.parent)
    documents = load_manifest(path)
    assert [doc.text for doc in documents] == ["Текст урока.", "Раздел\n\nПояснение."]
    assert documents[0].title == "Урок"
    assert documents[0].source_url == "https://example.org/lesson"


def test_pdf_import_preserves_page_numbers_and_links(tmp_path):
    """Импорт PDF пропускает пустые страницы и сохраняет номера в названиях и ссылках.

    Args:
        tmp_path: Временный каталог теста.
    """
    write_pdf(tmp_path / "lesson.pdf", ["First lesson", None, "Third lesson"])
    path = write_manifest(tmp_path, [source_row("lesson.pdf")])
    with pytest.warns(UserWarning, match="страница 2"):
        documents = load_manifest(path)
    assert [doc.id for doc in documents] == ["lesson:page-1", "lesson:page-3"]
    assert documents[1].text == "Third lesson"
    assert documents[1].source_url.endswith("#page=3")
    assert documents[1].title.endswith("стр. 3")


def test_pdf_without_text_is_rejected(tmp_path):
    """PDF без текстового слоя отклоняется с указанием необходимости OCR.

    Args:
        tmp_path: Временный каталог теста.
    """
    write_pdf(tmp_path / "scan.pdf", [None])
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="требуется OCR"):
        load_manifest(write_manifest(tmp_path, [source_row("scan.pdf")]))


def test_password_protected_pdf_is_rejected(tmp_path):
    """PDF под паролем отклоняется при импорте.

    Args:
        tmp_path: Временный каталог теста.
    """
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    writer.write(tmp_path / "locked.pdf")
    with pytest.raises(ValueError, match="паролем"):
        load_manifest(write_manifest(tmp_path, [source_row("locked.pdf")]))


def test_broken_pdf_is_reported(tmp_path):
    """Повреждённый PDF вызывает понятную ошибку чтения.

    Args:
        tmp_path: Временный каталог теста.
    """
    (tmp_path / "broken.pdf").write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="Не удалось прочитать PDF"):
        load_manifest(write_manifest(tmp_path, [source_row("broken.pdf")]))


@pytest.mark.parametrize(
    "rows", [[], [None], [{"id": "missing"}], [source_row("file.txt") | {"source_url": "bad"}]]
)
def test_invalid_manifest(tmp_path, rows):
    """Некорректные записи списка источников отклоняются.

    Args:
        tmp_path: Временный каталог теста.
        rows: Проверяемые записи источников.
    """
    with pytest.raises(ValueError):
        load_manifest(write_manifest(tmp_path, rows))


def test_duplicate_source_id_is_rejected(tmp_path):
    """Повторяющиеся идентификаторы источников отклоняются.

    Args:
        tmp_path: Временный каталог теста.
    """
    (tmp_path / "lesson.txt").write_text("text", encoding="utf-8")
    row = source_row("lesson.txt")
    with pytest.raises(ValueError, match="Повторяющийся id"):
        load_manifest(write_manifest(tmp_path, [row, row]))


def test_empty_source_and_unsupported_format(tmp_path):
    """Пустой материал и неподдерживаемый формат вызывают ошибки.

    Args:
        tmp_path: Временный каталог теста.
    """
    (tmp_path / "empty.txt").write_text(" \n ", encoding="utf-8")
    with pytest.raises(ValueError, match="не содержит текста"):
        load_manifest(write_manifest(tmp_path, [source_row("empty.txt")]))
    with pytest.raises(ValueError, match="не поддерживается"):
        load_manifest(write_manifest(tmp_path, [source_row("unsupported.docx")]))


def test_missing_file_is_not_silently_skipped(tmp_path):
    """Отсутствующий файл вызывает ошибку импорта.

    Args:
        tmp_path: Временный каталог теста.
    """
    with pytest.raises(FileNotFoundError):
        load_manifest(write_manifest(tmp_path, [source_row("missing.txt")]))
