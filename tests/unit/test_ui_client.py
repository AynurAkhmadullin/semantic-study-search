"""Проверки HTTP-клиента интерфейса без сетевых обращений."""

from unittest.mock import Mock

import httpx
import pytest

from semantic_study_search.ui import client


@pytest.fixture
def transport(monkeypatch):
    """Создаёт настройщик HTTP-транспорта с подменёнными ответами.

    Args:
        monkeypatch: Фикстура для временной подмены объектов и окружения.

    Returns:
        Функция установки обработчика HTTP-запросов.
    """
    original = httpx.Client

    def configure(handler):
        """Подключает обработчик запросов вместо сетевого транспорта.

        Args:
            handler: Обработчик тестовых HTTP-запросов.
        """
        monkeypatch.setattr(
            client.httpx,
            "Client",
            lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)),
        )

    return configure


def test_client_sends_search_body_and_parses_response(transport):
    """Клиент передаёт параметры поиска, сохраняет базовый путь и разбирает ответ.

    Args:
        transport: Функция установки подменённого HTTP-транспорта.
    """

    def handler(request):
        """Проверяет исходящий запрос и возвращает пустую поисковую выдачу.

        Args:
            request: Перехваченный HTTP-запрос.

        Returns:
            Успешный HTTP-ответ с пустой выдачей.
        """
        assert request.url == "http://search.local/api/search"
        assert request.method == "POST"
        assert b'"top_k":3' in request.content
        return httpx.Response(200, json={"query": "вопрос", "results": []})

    transport(handler)
    assert client.search("http://search.local/api", "вопрос", 3).results == []


def test_client_health(transport):
    """Клиент читает состояние и размер библиотеки из ответа API.

    Args:
        transport: Функция установки подменённого HTTP-транспорта.
    """
    transport(
        lambda request: httpx.Response(
            200, json={"status": "ready", "documents": 10, "fragments": 25}
        )
    )
    assert client.get_health("http://search.local").documents == 10


@pytest.mark.parametrize(
    "code,body,message",
    [
        (422, {"detail": "Сократите вопрос."}, "Сократите"),
        (503, {"detail": "busy"}, "недоступен или занят"),
        (500, {"detail": "private path"}, "не смог"),
        (302, {}, "не смог"),
    ],
)
def test_client_http_errors(transport, code, body, message):
    """Ошибки HTTP преобразуются в понятные сообщения клиенту.

    Args:
        transport: Функция установки подменённого HTTP-транспорта.
        code: Код HTTP-ответа.
        body: JSON-тело ошибочного ответа.
        message: Ожидаемый фрагмент сообщения об ошибке.
    """
    transport(lambda request: httpx.Response(code, json=body))
    with pytest.raises(client.APIClientError, match=message):
        client.search("http://search.local", "вопрос")


@pytest.mark.parametrize(
    "error,message",
    [
        (httpx.ConnectError("offline"), "Нет связи"),
        (httpx.ReadTimeout("slow"), "не ответил вовремя"),
    ],
)
def test_client_network_errors(transport, error, message):
    """Ошибки соединения и тайм-ауты получают понятные сообщения.

    Args:
        transport: Функция установки подменённого HTTP-транспорта.
        error: Имитируемая сетевая ошибка.
        message: Ожидаемый фрагмент сообщения об ошибке.
    """

    def handler(request):
        """Имитирует заданную сетевую ошибку при обработке запроса.

        Args:
            request: Неиспользуемый HTTP-запрос.
        """
        raise error

    transport(handler)
    with pytest.raises(client.APIClientError, match=message):
        client.get_health("http://search.local")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not JSON"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"unexpected": "format"}),
    ],
)
def test_client_invalid_responses(transport, response):
    """Клиент отклоняет некорректный JSON и ответы с нарушенным контрактом.

    Args:
        transport: Функция установки подменённого HTTP-транспорта.
        response: Ответ с некорректным содержимым.
    """
    transport(lambda request: response)
    with pytest.raises(client.APIClientError):
        client.search("http://search.local", "вопрос")


def test_client_rejects_empty_question_without_network(monkeypatch):
    """Клиент отклоняет пустой вопрос без сетевого запроса.

    Args:
        monkeypatch: Фикстура для временной подмены объектов и окружения.
    """
    request = Mock(side_effect=AssertionError("Network should not be used"))
    monkeypatch.setattr(client, "request_api", request)
    with pytest.raises(client.APIClientError, match="непустой"):
        client.search("http://search.local", " ")
    request.assert_not_called()


def test_client_rejects_unsafe_source_links(transport):
    """Клиент отклоняет результаты со ссылками, использующими javascript.

    Args:
        transport: Функция установки подменённого HTTP-транспорта.
    """
    transport(
        lambda request: httpx.Response(
            200,
            json={
                "query": "q",
                "results": [
                    {
                        "score": 0.9,
                        "chunk": {
                            "id": "x",
                            "document_id": "d",
                            "title": "x",
                            "text": "x",
                            "start": 0,
                            "end": 1,
                            "source_url": "javascript:alert(1)",
                        },
                    }
                ],
            },
        )
    )
    with pytest.raises(client.APIClientError, match="результаты"):
        client.search("http://search.local", "вопрос")


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp",
        "http://user:pass@host",
        "http://host?token=x",
        "http://[invalid",
        "http://host:bad",
    ],
)
def test_invalid_service_address(url):
    """Клиент отклоняет некорректный адрес API.

    Args:
        url: Некорректный адрес поискового API.
    """
    with pytest.raises(client.APIClientError, match="адрес"):
        client.get_health(url)
