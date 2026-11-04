"""Проверки настоящей модели E5, токенизатора и поиска по учебным материалам."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from semantic_study_search.domain import Document
from semantic_study_search.ingestion.chunking import PASSAGE_PREFIX, chunk_documents, token_count
from semantic_study_search.ingestion.corpus import load_corpus
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
def real_engine(real_encoder):
    """Создаёт поиск по встроенному корпусу с настоящей E5.

    Args:
        real_encoder: Настоящая модель E5 с токенизатором.

    Returns:
        Поисковый движок с векторами встроенных материалов.
    """
    chunks = chunk_documents(load_corpus(), real_encoder.tokenizer)
    assert all(
        token_count(real_encoder.tokenizer, PASSAGE_PREFIX + chunk.text) <= 320 for chunk in chunks
    )
    return SearchEngine(chunks, real_encoder)


def test_real_model_shape_and_normalization(real_encoder):
    """E5 возвращает конечные нормализованные векторы размерности 384.

    Args:
        real_encoder: Настоящая модель E5 с токенизатором.
    """
    vectors = real_encoder.encode_passages(["Текст на русском языке.", "Другой учебный материал."])
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


def test_demo_questions_retrieve_expected_sections(real_engine):
    """Ожидаемые источники входят в первые пять результатов контрольных вопросов.

    Args:
        real_engine: Поиск по встроенному корпусу с настоящей E5.
    """
    cases = json.loads(
        (Path(__file__).parents[1] / "fixtures/evaluation.json").read_text(encoding="utf-8")
    )
    failures = []
    for case in cases:
        results = real_engine.search(case["query"], top_k=5)
        found = {result.chunk.document_id for result in results}
        if not found.intersection(case["relevant_document_ids"]):
            failures.append({"query": case["query"], "found": sorted(found)})
    assert not failures, f"Expected source missing from top 5 fragments: {failures}"


def test_real_model_persistent_index_retrieves_added_material(tmp_path, real_encoder, monkeypatch):
    """Поиск находит добавленный материал без пересчёта сохранённых векторов.

    Args:
        tmp_path: Временный каталог теста.
        real_encoder: Настоящая модель E5 с токенизатором.
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    from unittest.mock import Mock

    from semantic_study_search.ingestion.files import load_manifest
    from semantic_study_search.ingestion.pipeline import prepare_index
    from semantic_study_search.storage.index import IndexConfig, load_index

    documents = load_corpus() + load_manifest(Path(__file__).parents[2] / "examples/library.json")
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
    assert "project-vector-search" in {result.chunk.document_id for result in results}
    passages.assert_not_called()


def test_real_api_keeps_model_and_document_vectors_loaded(
    tmp_path, real_encoder, real_engine, monkeypatch
):
    """API повторно использует настоящую модель и сохранённые векторы документов.

    Args:
        tmp_path: Временный каталог теста.
        real_encoder: Настоящая модель E5 с токенизатором.
        real_engine: Поиск по встроенному корпусу с настоящей E5.
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    from unittest.mock import Mock

    from fastapi.testclient import TestClient

    from semantic_study_search.api.app import create_app
    from semantic_study_search.api.settings import APISettings
    from semantic_study_search.storage.index import IndexConfig, StoredIndex, save_index

    index = StoredIndex(
        tuple(load_corpus()), real_engine.chunks, real_engine.vectors, IndexConfig()
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
                "/search", json={"query": "Зачем нужна маска внимания?", "top_k": 5}
            )
            assert response.status_code == 200
            assert "hf-chapter2-5" in {
                row["chunk"]["document_id"] for row in response.json()["results"]
            }
        response = client.post("/search", json={"query": "Как испечь яблочный пирог?"})
        assert response.status_code == 200
        assert response.json()["results"] == []
    model.assert_called_once()
    passages.assert_not_called()
    assert query.call_count == 3


def test_held_out_retrieval_and_rejection_with_real_model(real_encoder):
    """Фиксированный порог сохраняет поиск по темам и отсеивает посторонние вопросы.

    Args:
        real_encoder: Настоящая модель E5.
    """
    from semantic_study_search.evaluation import load_benchmark, score_cases, summarize
    from semantic_study_search.ingestion.files import load_manifest
    from semantic_study_search.search.relevance import default_min_score
    from semantic_study_search.storage.index import IndexConfig, StoredIndex

    root = Path(__file__).parents[2]
    # Контрольная разметка относится только к исходным материалам курса и двум примерам.
    documents = [doc for doc in load_corpus() if doc.id.startswith("hf-")]
    documents += load_manifest(root / "examples/library.json")
    engine = SearchEngine(chunk_documents(documents, real_encoder.tokenizer), real_encoder)
    index = StoredIndex(tuple(documents), engine.chunks, engine.vectors, IndexConfig())
    benchmark = load_benchmark(root / "tests/fixtures/relevance_test.json", index, "test")
    threshold = default_min_score()
    rows = score_cases(engine, benchmark["cases"], repeats=1)
    metrics = summarize(rows, threshold)
    assert metrics["hit_at_5"] >= 0.9
    assert metrics["unanswerable_rejection_rate"] >= 0.6
    assert metrics["by_category"]["out_of_scope"]["rejected"] == 10
    for row in rows:
        actual = engine.search(row["query"], top_k=5, min_score=threshold)
        assert [result.chunk.id for result in actual] == [
            result["chunk_id"] for result in row["retrieved"] if result["score"] >= threshold
        ]
    assert engine.search("Что такое гладиолус?", min_score=threshold) == []


def test_real_model_upload_is_immediately_searchable(tmp_path, real_encoder, monkeypatch):
    """Загруженный через API текст находится настоящей моделью без перезапуска.

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
    passages.assert_not_called()
