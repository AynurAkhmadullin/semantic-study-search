"""Подготовка выдержек из восьми разделов закреплённой версии курса Hugging Face."""

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.request import urlopen

REVISION = "8b893f0ede6c781045692a7f892f25a0c92b4805"
BASE_URL = f"https://raw.githubusercontent.com/huggingface/course/{REVISION}"
OUTPUT = Path(__file__).resolve().parents[1] / "src/semantic_study_search/resources"
SECTIONS = [
    ("chapter1/2", "Обработка естественного языка", [(10, 26)]),
    (
        "chapter1/4",
        "Как работают трансформеры?",
        [(41, 47), (87, 117), (125, 151)],
    ),
    ("chapter1/5", "Модели-кодировщики", [(10, 22)]),
    ("chapter1/6", "Модели-декодировщики", [(10, 21)]),
    ("chapter1/7", "Модели seq2seq", [(10, 21)]),
    (
        "chapter2/4",
        "Токенизаторы",
        [(27, 27), (35, 35), (61, 71), (75, 79), (87, 99), (108, 110), (222, 224), (235, 237)],
    ),
    (
        "chapter2/5",
        "Обработка нескольких последовательностей",
        [(186, 188), (199, 199), (261, 265), (319, 326)],
    ),
    ("chapter4/4", "Создание карточки модели", [(8, 54), (66, 72)]),
]


def clean_excerpt(text: str) -> str:
    """Удаляет разметку и лишние пустые строки из выбранного отрывка курса.

    Args:
        text: Отрывок курса с разметкой MDX и Markdown.

    Returns:
        Очищенный текст отрывка.
    """
    text = re.sub(r"<[^>]*>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[[^\]]+\]\]", "", text)
    text = re.sub(r"\[([^\]]+)\]\s*\(([^)]+)\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\* ", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = text.replace("`", "")
    text = "\n".join(line.rstrip() for line in html.unescape(text).splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    """Собирает демонстрационный корпус и сохраняет его вместе с лицензией."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()

    def read_source(remote_path: str, local_name: str) -> bytes:
        """Читает исходный файл из указанного каталога или репозитория курса.

        Args:
            remote_path: Путь к файлу в закреплённой версии репозитория курса.
            local_name: Имя файла в каталоге исходников.

        Returns:
            Содержимое файла в байтах.
        """
        if args.source_dir is not None:
            return (args.source_dir / local_name).read_bytes()
        with urlopen(f"{BASE_URL}/{remote_path}", timeout=60) as response:
            return response.read()

    documents = []
    for section, title, ranges in SECTIONS:
        filename = f"chapters/ru/{section}.mdx"
        raw = read_source(filename, section.replace("/", "-") + ".mdx")
        lines = raw.decode("utf-8").splitlines()
        if any(start < 1 or end > len(lines) for start, end in ranges):
            raise ValueError(f"Unexpected source length: {filename}")
        excerpts = ["\n".join(lines[start - 1 : end]) for start, end in ranges]
        documents.append(
            {
                "id": "hf-" + section.replace("/", "-"),
                "title": title,
                "source_url": f"https://huggingface.co/learn/llm-course/ru/{section}",
                "source_file": filename,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source_lines": ranges,
                "text": clean_excerpt("\n\n".join(excerpts)),
            }
        )
    license_bytes = read_source("LICENSE", "LICENSE")
    payload = {
        "schema_version": 1,
        "source_repository": "https://github.com/huggingface/course",
        "source_revision": REVISION,
        "license": "Apache-2.0",
        "license_sha256": hashlib.sha256(license_bytes).hexdigest(),
        "modifications": "Selected Russian prose excerpts; removed MDX/HTML, images, video, "
        "code examples and Markdown formatting. Source line numbers are inclusive, one-based.",
        "documents": documents,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "corpus.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (OUTPUT / "LICENSE.huggingface-course.txt").write_bytes(license_bytes)
    print(f"Prepared {len(documents)} documents in {OUTPUT}")


if __name__ == "__main__":
    main()
