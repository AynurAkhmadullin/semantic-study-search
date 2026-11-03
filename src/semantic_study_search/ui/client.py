"""Обращение веб-интерфейса к поисковому API без загрузки модели."""

from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from semantic_study_search.api.schemas import HealthResponse, SearchRequest, SearchResponse
from semantic_study_search.limits import MAX_QUERY_CHARACTERS


class APIClientError(RuntimeError):
    """Содержит понятное пользователю описание ошибки обращения к поиску."""


def request_api(api_url: str, method: str, path: str, payload: dict | None = None) -> dict:
    """Выполняет HTTP-запрос с ограничением времени ожидания.

    Args:
        api_url: Базовый адрес поискового API.
        method: Метод HTTP-запроса.
        path: Путь метода относительно адреса API.
        payload: Тело JSON для поискового запроса.

    Returns:
        Проверяемый JSON-объект ответа.
    """
    try:
        url = urlsplit(api_url)
        _ = url.port
    except ValueError as error:
        raise APIClientError(
            "Некорректный адрес поискового сервиса. Обратитесь к автору проекта."
        ) from error
    if (
        url.scheme not in {"http", "https"}
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise APIClientError("Некорректный адрес поискового сервиса. Обратитесь к автору проекта.")
    try:
        with httpx.Client(
            base_url=api_url.rstrip("/") + "/", timeout=httpx.Timeout(30, connect=3)
        ) as client:
            response = client.request(method, path, json=payload)
    except httpx.TimeoutException as error:
        raise APIClientError("Поиск не ответил вовремя. Попробуйте ещё раз.") from error
    except httpx.RequestError as error:
        raise APIClientError("Нет связи с поисковым сервисом. Попробуйте позже.") from error
    except httpx.InvalidURL as error:
        raise APIClientError(
            "Некорректный адрес поискового сервиса. Обратитесь к автору проекта."
        ) from error
    if response.status_code == 503:
        raise APIClientError("Поисковый сервис временно недоступен или занят. Попробуйте ещё раз.")
    if response.status_code == 422:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise APIClientError(
            detail if isinstance(detail, str) else "Проверьте вопрос и количество результатов."
        )
    if response.is_error or response.is_redirect:
        raise APIClientError("Сервис не смог выполнить запрос. Попробуйте позже.")
    try:
        data = response.json()
    except ValueError as error:
        raise APIClientError("Сервис вернул некорректный ответ. Попробуйте позже.") from error
    if not isinstance(data, dict):
        raise APIClientError("Сервис вернул некорректный ответ. Попробуйте позже.")
    return data


def get_health(api_url: str) -> HealthResponse:
    """Проверяет доступность поиска и получает размер библиотеки.

    Args:
        api_url: Базовый адрес поискового API.

    Returns:
        Состояние поискового сервиса.
    """
    try:
        return HealthResponse.model_validate(request_api(api_url, "GET", "health"))
    except ValidationError as error:
        raise APIClientError("Не удалось прочитать состояние поискового сервиса.") from error


def search(api_url: str, query: str, top_k: int = 5) -> SearchResponse:
    """Запрашивает подходящие фрагменты у поискового API.

    Args:
        api_url: Базовый адрес поискового API.
        query: Поисковый вопрос.
        top_k: Максимальное количество фрагментов.

    Returns:
        Обработанный вопрос и найденные фрагменты.
    """
    try:
        request = SearchRequest(query=query, top_k=top_k)
    except ValidationError as error:
        raise APIClientError(
            f"Введите непустой вопрос до {MAX_QUERY_CHARACTERS} символов и выберите количество результатов."
        ) from error
    try:
        return SearchResponse.model_validate(
            request_api(api_url, "POST", "search", request.model_dump())
        )
    except ValidationError as error:
        raise APIClientError("Не удалось прочитать результаты поиска.") from error
