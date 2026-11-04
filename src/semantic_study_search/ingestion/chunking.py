"""Разбиение документов на перекрывающиеся фрагменты по токенам."""

from collections.abc import Iterable
from typing import Any, Protocol

from semantic_study_search.domain import Chunk, Document

PASSAGE_PREFIX = "passage: "
CHUNKER_VERSION = 1


class Tokenizer(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Any: ...


def token_count(tokenizer: Tokenizer, text: str) -> int:
    """Считает токены текста с учётом специальных токенов.

    Args:
        tokenizer: Токенизатор модели.
        text: Текст для подсчёта.

    Returns:
        Количество токенов без усечения текста.
    """
    return len(
        tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
    )


def chunk_documents(
    documents: Iterable[Document],
    tokenizer: Tokenizer,
    *,
    max_tokens: int = 320,
    overlap: int = 48,
) -> list[Chunk]:
    """Разбивает документы на перекрывающиеся фрагменты с сохранением исходного текста.

    Args:
        documents: Документы.
        tokenizer: Токенизатор с поддержкой символьных смещений.
        max_tokens: Лимит токенов фрагмента с префиксом E5 и специальными токенами.
        overlap: Целевое перекрытие в токенах до выравнивания по границам слов.

    Returns:
        Фрагменты с источниками и символьными границами в документах.
    """
    budget = max_tokens - token_count(tokenizer, PASSAGE_PREFIX)
    if budget <= 0 or not 0 <= overlap < budget:
        raise ValueError("Размер фрагмента должен превышать префикс и перекрытие.")
    chunks = []
    for document in documents:
        offsets = tokenizer(
            document.text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )["offset_mapping"]
        offsets = [(start, end) for start, end in offsets if end > start]
        if not offsets:
            raise ValueError(f"Документ не содержит токенов: {document.id}")
        first = 0
        number = 0
        while first < len(offsets):
            last = min(first + budget, len(offsets))
            while last > first:
                start, end = offsets[first][0], offsets[last - 1][1]
                # Keep whole words in the excerpts displayed to the reader.
                while start > 0 and not document.text[start - 1].isspace():
                    start -= 1
                while end < len(document.text) and not document.text[end].isspace():
                    end += 1
                text = document.text[start:end]
                start += len(text) - len(text.lstrip())
                end -= len(text) - len(text.rstrip())
                text = document.text[start:end]
                if text and token_count(tokenizer, PASSAGE_PREFIX + text) <= max_tokens:
                    break
                last -= 1
            if last == first:
                raise ValueError(f"Токен не помещается во фрагмент: {document.id}")
            chunks.append(
                Chunk(
                    id=f"{document.id}:{number}",
                    document_id=document.id,
                    title=document.title,
                    source_url=document.source_url,
                    text=text,
                    start=start,
                    end=end,
                )
            )
            if last == len(offsets):
                break
            first = max(first + 1, last - overlap)
            number += 1
    return chunks
