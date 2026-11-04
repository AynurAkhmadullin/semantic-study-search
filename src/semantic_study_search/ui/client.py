"""Обращение веб-интерфейса к поисковому API без загрузки модели."""

from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from semantic_study_search.api.schemas import (
    HealthResponse,
    MaterialListResponse,
    MaterialResponse,
    SearchRequest,
    SearchResponse,
)
from semantic_study_search.limits import MAX_QUERY_CHARACTERS
from semantic_study_search.sources import uploaded_source_id


class APIClientError(RuntimeError):
    """Содержит понятное пользователю описание ошибки обращения к поиску."""


def send_request(
    api_url: str,
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    file: tuple[str, bytes] | None = None,
    title: str = "",
) -> httpx.Response:
    """Выполняет HTTP-запрос с ограничением времени ожидания.

    Args:
        api_url: Базовый адрес поискового API.
        method: Метод HTTP-запроса.
        path: Путь метода относительно адреса API.
        payload: Тело JSON для поискового запроса.
        file: Имя и содержимое загружаемого файла.
        title: Название материала.

    Returns:
        Успешный HTTP-ответ.
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
            response = client.request(
                method,
                path,
                json=payload,
                files={"file": file} if file else None,
                data={"title": title} if file else None,
            )
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
    if response.status_code in {404, 409, 413, 422}:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise APIClientError(
            detail if isinstance(detail, str) else "Проверьте вопрос и количество результатов."
        )
    if response.is_error or response.is_redirect:
        raise APIClientError("Сервис не смог выполнить запрос. Попробуйте позже.")
    return response


def request_api(api_url: str, method: str, path: str, payload: dict | None = None) -> dict:
    """Читает JSON-ответ API с проверкой формы данных.

    Args:
        api_url: Базовый адрес API.
        method: HTTP-метод.
        path: Относительный путь маршрута.
        payload: Тело поискового запроса.

    Returns:
        JSON-объект ответа.
    """
    response = send_request(api_url, method, path, payload)
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


def search(api_url: str, query: str, top_k: int = 5, *, relaxed: bool = False) -> SearchResponse:
    """Запрашивает подходящие фрагменты у поискового API.

    Args:
        api_url: Базовый адрес поискового API.
        query: Поисковый вопрос.
        top_k: Максимальное количество фрагментов.
        relaxed: Показать также результаты ниже порога сходства.

    Returns:
        Обработанный вопрос и найденные фрагменты.
    """
    try:
        request = SearchRequest(query=query, top_k=top_k, relaxed=relaxed)
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


def upload_material(api_url: str, filename: str, content: bytes, title: str) -> MaterialResponse:
    """Передаёт материал на фоновую подготовку.

    Args:
        api_url: Адрес API.
        filename: Исходное имя файла.
        content: Байты материала.
        title: Необязательное название.

    Returns:
        Состояние принятой загрузки.
    """
    try:
        response = send_request(api_url, "POST", "materials", file=(filename, content), title=title)
        return MaterialResponse.model_validate(response.json())
    except ValueError as error:
        raise APIClientError("Не удалось прочитать состояние загрузки.") from error


def get_material(api_url: str, identifier: str) -> MaterialResponse:
    """Получает текущий этап обработки материала.

    Args:
        api_url: Адрес API.
        identifier: Идентификатор загрузки.

    Returns:
        Состояние фоновой подготовки.
    """
    try:
        return MaterialResponse.model_validate(
            request_api(api_url, "GET", f"materials/{identifier}")
        )
    except ValidationError as error:
        raise APIClientError("Не удалось прочитать состояние материала.") from error


def list_materials(api_url: str) -> MaterialListResponse:
    """Получает историю загрузок общей библиотеки.

    Args:
        api_url: Адрес API.

    Returns:
        Готовые и обрабатываемые материалы.
    """
    try:
        return MaterialListResponse.model_validate(request_api(api_url, "GET", "materials"))
    except ValidationError as error:
        raise APIClientError("Не удалось прочитать список материалов.") from error


def delete_material(api_url: str, identifier: str) -> None:
    """Удаляет загруженный материал из общей библиотеки.

    Args:
        api_url: Адрес API.
        identifier: Идентификатор удаляемого материала.
    """
    if uploaded_source_id(f"/materials/{identifier}/file") != identifier:
        raise APIClientError("Некорректный идентификатор материала.")
    response = send_request(api_url, "DELETE", f"materials/{identifier}")
    if response.status_code != 204:
        raise APIClientError("Сервис не подтвердил удаление материала. Обновите список загрузок.")


def download_material(api_url: str, source: str) -> bytes:
    """Скачивает оригинал по проверенной внутренней ссылке.

    Args:
        api_url: Адрес API.
        source: Внутренняя ссылка найденного фрагмента.

    Returns:
        Байты оригинального файла.
    """
    identifier = uploaded_source_id(source)
    if identifier is None:
        raise APIClientError("Некорректная ссылка на загруженный материал.")
    return send_request(api_url, "GET", f"materials/{identifier}/file").content
