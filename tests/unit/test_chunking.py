"""Проверки границ, перекрытия и допустимого размера фрагментов."""

from itertools import pairwise

import pytest

from semantic_study_search.domain import Document
from semantic_study_search.ingestion.chunking import PASSAGE_PREFIX, chunk_documents, token_count


def test_windows_overlap_and_preserve_all_source_text(tokenizer):
    """Фрагменты перекрываются, сохраняют исходный текст и укладываются в лимит токенов.

    Args:
        tokenizer: Тестовый токенизатор по пробельным словам.
    """
    text = "  Раз два три\nчетыре пять шесть семь восемь девять десять!  "
    document = Document("doc", "Название", "https://example.org", text)
    chunks = chunk_documents([document], tokenizer, max_tokens=8, overlap=2)
    assert len(chunks) > 1
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    covered = set()
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert chunk.document_id == "doc"
        assert chunk.title == document.title
        assert chunk.source_url == document.source_url
        assert token_count(tokenizer, PASSAGE_PREFIX + chunk.text) <= 8
        covered.update(range(chunk.start, chunk.end))
    assert all(index in covered for index, char in enumerate(text) if not char.isspace())
    assert all(left.start < right.start < left.end for left, right in pairwise(chunks))


def test_short_document_is_one_chunk(tokenizer):
    """Короткий документ сохраняется в одном фрагменте.

    Args:
        tokenizer: Тестовый токенизатор по пробельным словам.
    """
    document = Document("doc", "Title", "https://example.org", "Короткий текст.")
    chunks = chunk_documents([document], tokenizer)
    assert len(chunks) == 1
    assert chunks[0].text == document.text


@pytest.mark.parametrize("max_tokens,overlap", [(3, 0), (10, -1), (10, 7)])
def test_invalid_chunk_sizes(tokenizer, max_tokens, overlap):
    """Недопустимые размеры фрагмента и перекрытия отклоняются.

    Args:
        tokenizer: Тестовый токенизатор по пробельным словам.
        max_tokens: Проверяемый предел токенов во фрагменте.
        overlap: Проверяемый размер перекрытия в токенах.
    """
    with pytest.raises(ValueError):
        chunk_documents([], tokenizer, max_tokens=max_tokens, overlap=overlap)


def test_empty_text_rejected(tokenizer):
    """Документ без токенов отклоняется при разбиении.

    Args:
        tokenizer: Тестовый токенизатор по пробельным словам.
    """
    with pytest.raises(ValueError, match="не содержит токенов"):
        chunk_documents([Document("doc", "Title", "https://example.org", " ")], tokenizer)
