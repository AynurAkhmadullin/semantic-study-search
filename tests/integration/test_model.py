"""Проверки настоящей модели E5, токенизатора и поиска по материалам."""

import os
from pathlib import Path

import numpy as np
import pytest

from semantic_study_search.domain import Document
from semantic_study_search.ingestion.chunking import PASSAGE_PREFIX, chunk_documents, token_count
from semantic_study_search.search.encoder import E5Encoder
from semantic_study_search.search.engine import SearchEngine

pytestmark = [pytest.mark.integration, pytest.mark.model]


@pytest.fixture(scope="module")
def real_encoder():
    """Загружает настоящую E5 с настройками кэша из окружения.

    Returns:
        Загруженный кодировщик E5.
    """
    cache = os.environ.get("SSS_MODEL_CACHE")
    return E5Encoder(
        cache_folder=Path(cache) if cache else None,
        offline=os.environ.get("HF_HUB_OFFLINE") == "1",
    )


@pytest.fixture(scope="module")
def text_documents():
    """Создаёт изолированные документы только для тестов модели.

    Returns:
        Два текста с различными темами.
    """
    return [
        Document(
            "vectors",
            "Сходство векторов",
            "https://example.org/vectors",
            "Косинусное сходство измеряет близость направлений двух векторов. "
            "Для нормализованных векторов оно равно скалярному произведению. "
            "Семантический поиск сравнивает вектор вопроса с векторами текстов.",
        ),
        Document(
            "botany",
            "Ботаника",
            "https://example.org/botany",
            "Гладиолус — многолетнее клубнелуковичное растение семейства Ирисовые. "
            "Его также называют шпажником из-за длинных узких листьев. "
            "Цветки располагаются вдоль высокого стебля и образуют соцветие колос.",
        ),
    ]


@pytest.fixture(scope="module")
def real_engine(real_encoder, text_documents):
    """Создаёт поиск по изолированным тестовым текстам с настоящей E5.

    Args:
        real_encoder: Настоящая модель E5 с токенизатором.
        text_documents: Документы, созданные только для теста.

    Returns:
        Поисковый движок с векторами тестовых материалов.
    """
    chunks = chunk_documents(text_documents, real_encoder.tokenizer)
    assert all(
        token_count(real_encoder.tokenizer, PASSAGE_PREFIX + chunk.text) <= 320 for chunk in chunks
    )
    return SearchEngine(chunks, real_encoder)


def test_real_model_shape_and_normalization(real_encoder):
    """E5 возвращает конечные нормализованные векторы размерности 384.

    Args:
        real_encoder: Настоящая модель E5 с токенизатором.
    """
    vectors = real_encoder.encode_passages(["Текст на русском языке.", "Другой материал."])
    assert vectors.shape == (2, 384)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1], atol=1e-5)


def test_real_tokenizer_preserves_unicode_and_respects_budget(real_encoder):
    """Разбиение настоящим токенизатором сохраняет Unicode и соблюдает лимит токенов.

    Args:
        real_encoder: Настоящая модель E5 с токенизатором.
    """
    text = "Ёж изучает NLP 🤗 — attention_mask, e\u0301, 中文!\n" * 80
    document = Document("unicode", "Юникод", "https://example.org", text)
    chunks = chunk_documents([document], real_encoder.tokenizer, max_tokens=80, overlap=10)
    covered = set()
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert token_count(real_encoder.tokenizer, PASSAGE_PREFIX + chunk.text) <= 80
        covered.update(range(chunk.start, chunk.end))
    assert all(index in covered for index, char in enumerate(text) if not char.isspace())


def test_real_model_persistent_index_retrieves_added_material(
    tmp_path, real_encoder, text_documents, monkeypatch
):
    """Поиск находит добавленный материал без пересчёта сохранённых векторов.

    Args:
        tmp_path: Временный каталог теста.
        real_encoder: Настоящая модель E5 с токенизатором.
        text_documents: Документы, созданные только для теста.
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    from unittest.mock import Mock

    from semantic_study_search.ingestion.pipeline import prepare_index
    from semantic_study_search.storage.index import IndexConfig, load_index

    documents = text_documents
    config = IndexConfig()
    index, created = prepare_index(tmp_path, documents, lambda: real_encoder, config=config)
    assert created and len(index.documents) == len(documents)
    factory = Mock(side_effect=AssertionError("Model must not reload for an unchanged library"))
    _, created = prepare_index(tmp_path, documents, factory, config=config)
    assert not created
    factory.assert_not_called()
    passages = Mock(side_effect=AssertionError("Saved document vectors must be reused"))
    monkeypatch.setattr(real_encoder, "encode_passages", passages)
    engine = SearchEngine.from_index(load_index(tmp_path), real_encoder)
    results = engine.search("Как вычислить косинусное сходство нормализованных векторов?", top_k=3)
    assert results[0].chunk.document_id == "vectors"
    passages.assert_not_called()


def test_real_api_keeps_model_and_document_vectors_loaded(
    tmp_path, real_encoder, real_engine, text_documents, monkeypatch
):
    """API повторно использует настоящую модель и сохранённые векторы документов.

    Args:
        tmp_path: Временный каталог теста.
        real_encoder: Настоящая модель E5 с токенизатором.
        real_engine: Поиск по изолированным текстам с настоящей E5.
        text_documents: Документы, созданные только для теста.
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    from unittest.mock import Mock

    from fastapi.testclient import TestClient

    from semantic_study_search.api.app import create_app
    from semantic_study_search.api.settings import APISettings
    from semantic_study_search.storage.index import IndexConfig, StoredIndex, save_index

    index = StoredIndex(
        tuple(text_documents), real_engine.chunks, real_engine.vectors, IndexConfig()
    )
    save_index(tmp_path, index)
    model = Mock(return_value=real_encoder)
    passages = Mock(side_effect=AssertionError("Document vectors should come from disk"))
    query = Mock(wraps=real_encoder.encode_query)
    monkeypatch.setattr("semantic_study_search.api.service.E5Encoder", model)
    monkeypatch.setattr(real_encoder, "encode_passages", passages)
    monkeypatch.setattr(real_encoder, "encode_query", query)
    with TestClient(create_app(APISettings(index_dir=tmp_path, offline=True))) as client:
        assert client.get("/health").json()["documents"] == len(index.documents)
        for _ in range(2):
            response = client.post(
                "/search",
                json={
                    "query": "Как вычислить косинусное сходство нормализованных векторов?",
                    "top_k": 5,
                },
            )
            assert response.status_code == 200
            assert "vectors" in {row["chunk"]["document_id"] for row in response.json()["results"]}
        response = client.post("/search", json={"query": "Как испечь яблочный пирог?"})
        assert response.status_code == 200
        assert response.json()["results"] == []
    model.assert_called_once()
    passages.assert_not_called()
    assert query.call_count == 3


def test_real_model_upload_is_immediately_searchable(tmp_path, real_encoder, monkeypatch):
    """Загрузка и удаление меняют поиск настоящей модели без пересчёта прежних векторов.

    Args:
        tmp_path: Временная библиотека.
        real_encoder: Настоящая модель E5.
        monkeypatch: Подмена повторной загрузки весов.
    """
    from unittest.mock import Mock

    from fastapi.testclient import TestClient
    from tests.integration.test_uploads import post_material, wait_material

    from semantic_study_search.api.app import create_app
    from semantic_study_search.api.settings import APISettings

    model = Mock(return_value=real_encoder)
    monkeypatch.setattr("semantic_study_search.api.service.E5Encoder", model)
    settings = APISettings(index_dir=tmp_path, offline=True)
    text = (
        "Гладиолус — многолетнее клубнелуковичное растение семейства Ирисовые. "
        "Его также называют шпажником из-за длинных узких листьев. "
        "Цветки располагаются вдоль высокого стебля и образуют соцветие колос."
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["documents"] == 0
        assert client.post("/search", json={"query": "Что такое гладиолус?"}).status_code == 409
        identifier = post_material(client, text, "botany.txt")
        assert wait_material(client, identifier, timeout=30)["status"] == "ready"
        response = client.post("/search", json={"query": "Что такое гладиолус?"})
        assert response.status_code == 200
        assert response.json()["results"][0]["chunk"]["text"] == text
        assert client.get(f"/materials/{identifier}/file").content == text.encode()
        model.assert_called_once()
    # Повторный запуск читает сохранённые векторы без кодирования документа.
    passages = Mock(side_effect=AssertionError("Saved vectors must be reused"))
    monkeypatch.setattr(real_encoder, "encode_passages", passages)
    with TestClient(create_app(settings)) as client:
        result = client.post("/search", json={"query": "Что такое гладиолус?"}).json()
        assert result["results"][0]["chunk"]["document_id"] == identifier
        assert client.get(f"/materials/{identifier}/file").content == text.encode()
        assert client.delete(f"/materials/{identifier}").status_code == 204
        assert client.post("/search", json={"query": "Что такое гладиолус?"}).status_code == 409
        assert client.get(f"/materials/{identifier}/file").status_code == 404
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["documents"] == 0
        assert client.get("/materials").json() == {"materials": []}
        assert client.post("/search", json={"query": "Что такое гладиолус?"}).status_code == 409
    passages.assert_not_called()
