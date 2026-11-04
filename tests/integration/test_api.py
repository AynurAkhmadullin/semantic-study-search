"""Проверки жизненного цикла, контрактов и ошибок поискового API."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from semantic_study_search.api.app import create_app
from semantic_study_search.api.service import SearchService
from semantic_study_search.api.settings import APISettings

pytestmark = pytest.mark.integration


@pytest.fixture
def service(sample_index, encoder):
    """Создаёт поисковый сервис на готовом тестовом индексе.

    Args:
        sample_index: Индекс трёх документов с готовыми векторами.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.

    Returns:
        Готовый поисковый сервис.
    """
    encoder.passage_calls = 0
    return SearchService(sample_index, encoder, min_score=-1)


def test_lifespan_loads_once_and_reuses_vectors(service, encoder):
    """API загружает сервис один раз и повторно использует векторы документов.

    Args:
        service: Поисковый сервис с тестовым кодировщиком.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    loader = Mock(return_value=service)
    application = create_app(APISettings(), service_loader=loader)
    loader.assert_not_called()
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ready", "documents": 3, "fragments": 3}
        for _ in range(2):
            response = client.post("/search", json={"query": "  вопрос  ", "top_k": 2})
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "вопрос"
            assert len(data["results"]) == 2
            assert data["results"][0]["chunk"]["source_url"] == "https://example.org/0"
            assert data["results"][0]["score"] >= data["results"][1]["score"]
        loader.assert_called_once()
        assert encoder.passage_calls == 0
        assert encoder.query_calls == 2
    assert application.state.search_service is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": " \n "},
        {"query": 123},
        {"query": "а" * 2001},
        {"query": "вопрос", "top_k": 0},
        {"query": "вопрос", "top_k": 21},
        {"query": "вопрос", "top_k": True},
        {"query": "вопрос", "top_k": 1.5},
        {"query": "вопрос", "top_k": "5"},
        {"query": "вопрос", "unexpected": "field"},
    ],
)
def test_invalid_requests_fail_before_inference(service, encoder, payload):
    """API отклоняет некорректные параметры до обращения к модели.

    Args:
        service: Поисковый сервис с тестовым кодировщиком.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
        payload: Некорректное тело поискового запроса.
    """
    with TestClient(create_app(service_loader=lambda settings: service)) as client:
        response = client.post("/search", json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)
    assert encoder.query_calls == 0


def test_invalid_json(service):
    """API возвращает ошибку 422 для некорректного JSON.

    Args:
        service: Поисковый сервис с тестовым кодировщиком.
    """
    with TestClient(create_app(service_loader=lambda settings: service)) as client:
        response = client.post("/search", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "JSON" in response.json()["detail"]


def test_query_token_limit_is_reported_as_input_error(service, encoder):
    """Превышение лимита токенов даёт 422 и не блокирует следующий запрос.

    Args:
        service: Поисковый сервис с тестовым кодировщиком.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    with TestClient(create_app(service_loader=lambda settings: service)) as client:
        response = client.post("/search", json={"query": "я " * 510})
        assert response.status_code == 422
        assert "Сократите" in response.json()["detail"]
        assert encoder.query_calls == 0
        assert client.post("/search", json={"query": "короткий вопрос"}).status_code == 200


def test_startup_failure_returns_unavailable_without_internal_details():
    """Сбой загрузки даёт 503 без внутренних деталей и сохраняет доступ к Swagger."""
    loader = Mock(side_effect=FileNotFoundError("private/index/path"))
    with TestClient(create_app(service_loader=loader)) as client:
        health = client.get("/health")
        search = client.post("/search", json={"query": "вопрос"})
        assert health.status_code == search.status_code == 503
        assert health.json()["status"] == "unavailable"
        assert "private" not in search.text
        assert client.get("/docs").status_code == 200


def test_internal_search_failure_is_sanitized_and_lock_is_released(service, encoder):
    """Ошибка поиска скрывает внутренние детали и освобождает блокировку.

    Args:
        service: Поисковый сервис с тестовым кодировщиком.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    original = encoder.encode_query
    encoder.encode_query = Mock(side_effect=RuntimeError("secret cache path"))
    with TestClient(create_app(service_loader=lambda settings: service)) as client:
        response = client.post("/search", json={"query": "вопрос"})
        assert response.status_code == 500
        assert "secret" not in response.text
        encoder.encode_query = original
        assert client.post("/search", json={"query": "вопрос"}).status_code == 200


def test_concurrent_search_returns_busy_while_health_stays_available(service, encoder):
    """Занятый поиск отклоняет второй запрос, сохраняя доступность проверки готовности.

    Args:
        service: Поисковый сервис с тестовым кодировщиком.
        encoder: Кодировщик с фиксированными векторами и счётчиками вызовов.
    """
    started, release = Event(), Event()
    original = encoder.encode_query

    def slow_query(query):
        """Задерживает кодирование запроса до разрешающего сигнала.

        Args:
            query: Поисковый вопрос.

        Returns:
            Вектор вопроса после снятия задержки.
        """
        started.set()
        assert release.wait(5)
        return original(query)

    encoder.encode_query = slow_query
    with TestClient(create_app(service_loader=lambda settings: service)) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.post, "/search", json={"query": "первый вопрос"})
            try:
                assert started.wait(5)
                assert client.get("/health").status_code == 200
                response = client.post("/search", json={"query": "второй вопрос"})
                assert response.status_code == 503
                assert response.headers["retry-after"] == "1"
            finally:
                release.set()
            assert future.result(timeout=5).status_code == 200


def test_openapi_describes_search_contract(service):
    """OpenAPI описывает маршруты, ответы и ограничения поискового запроса.

    Args:
        service: Поисковый сервис с тестовым кодировщиком.
    """
    with TestClient(create_app(service_loader=lambda settings: service)) as client:
        schema = client.get("/openapi.json").json()
    assert {"/health", "/search"} <= schema["paths"].keys()
    assert {"200", "422", "500", "503"} <= schema["paths"]["/search"]["post"]["responses"].keys()
    assert schema["components"]["schemas"]["SearchRequest"]["properties"]["top_k"]["maximum"] == 20


def test_low_similarity_is_successful_empty_response(service, encoder):
    """Слабое сходство возвращает пустой список с HTTP 200.

    Args:
        service: Тестовый поисковый сервис.
        encoder: Кодировщик с изменяемым вектором вопроса.
    """
    import numpy as np

    service.min_score = 0.8
    encoder.query = np.array([-1, 0], dtype=np.float32)
    with TestClient(create_app(service_loader=lambda settings: service)) as client:
        response = client.post("/search", json={"query": "другая тема"})
        assert response.status_code == 200
        assert response.json() == {"query": "другая тема", "results": []}
        assert client.get("/health").json()["status"] == "ready"
