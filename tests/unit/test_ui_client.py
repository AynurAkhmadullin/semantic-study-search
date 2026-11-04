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


def test_client_deletes_material_without_json_response(transport):
    """Удаление сохраняет префикс API и принимает пустой ответ HTTP 204.

    Args:
        transport: Подмена HTTP-транспорта.
    """
    identifier = "a" * 32
    requests = []
    transport(lambda request: requests.append(request) or httpx.Response(204))
    client.delete_material("http://search.local/api/v1", identifier)
    assert len(requests) == 1
    assert requests[0].method == "DELETE"
    assert requests[0].url.path == f"/api/v1/materials/{identifier}"
    assert "authorization" not in requests[0].headers


@pytest.mark.parametrize("identifier", ["../other", "unknown", "a" * 32 + "?query=x"])
def test_client_rejects_invalid_delete_identifier(identifier, monkeypatch):
    """Некорректный идентификатор отклоняется до отправки запроса.

    Args:
        identifier: Неподходящий идентификатор.
        monkeypatch: Подмена сетевого запроса.
    """
    request = Mock()
    monkeypatch.setattr(client, "send_request", request)
    with pytest.raises(client.APIClientError, match="идентификатор"):
        client.delete_material("http://search.local", identifier)
    request.assert_not_called()


def test_client_requires_delete_confirmation_from_api(transport):
    """Успешный HTTP-код без ожидаемого контракта не подтверждает удаление.

    Args:
        transport: Подмена HTTP-транспорта.
    """
    transport(lambda request: httpx.Response(200, json={}))
    with pytest.raises(client.APIClientError, match="не подтвердил"):
        client.delete_material("http://search.local", "a" * 32)


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
        (404, {"detail": "Материал не найден."}, "не найден"),
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


def test_client_uploads_multipart_without_authorization(transport):
    """Клиент передаёт файл и название без заголовка авторизации.

    Args:
        transport: Подменённый HTTP-транспорт.
    """
    identifier = "a" * 32

    def handler(request):
        """Проверяет содержимое загрузки и возвращает принятую задачу.

        Args:
            request: HTTP-запрос загрузки.

        Returns:
            Ответ о принятом материале.
        """
        assert request.url == "http://search.local/api/materials"
        assert "authorization" not in request.headers
        assert request.headers["content-type"].startswith("multipart/form-data;")
        assert "Текст".encode() in request.content
        assert "Лекция".encode() in request.content
        return httpx.Response(
            202,
            json={
                "id": identifier,
                "file_name": "lecture.txt",
                "title": "Лекция",
                "status": "queued",
            },
        )

    transport(handler)
    result = client.upload_material(
        "http://search.local/api", "lecture.txt", "Текст".encode(), "Лекция"
    )
    assert result.id == identifier


def test_client_download_keeps_api_prefix_and_omits_page_fragment(transport):
    """Внутренний источник скачивается через API с сохранением его базового пути.

    Args:
        transport: Подменённый HTTP-транспорт.
    """
    source = "/materials/" + "a" * 32 + "/file"

    def handler(request):
        """Проверяет адрес скачивания оригинала.

        Args:
            request: HTTP-запрос оригинального файла.

        Returns:
            Содержимое исходного файла.
        """
        assert str(request.url) == "http://search.local/api" + source
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"original")

    transport(handler)
    assert client.download_material("http://search.local/api", source + "#page=2") == b"original"


@pytest.mark.parametrize(
    "source", ["/materials/../file", "//evil.test/file", "https://evil.test/file"]
)
def test_client_rejects_invalid_download_source(source, monkeypatch):
    """Небезопасная ссылка не вызывает сетевого запроса скачивания.

    Args:
        source: Некорректная ссылка на оригинал.
        monkeypatch: Подмена сетевого вызова.
    """
    send = Mock()
    monkeypatch.setattr(client, "send_request", send)
    with pytest.raises(client.APIClientError):
        client.download_material("http://search.local", source)
    send.assert_not_called()
