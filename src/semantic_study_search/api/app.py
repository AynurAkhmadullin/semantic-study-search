"""HTTP-доступ к семантическому поиску через FastAPI."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from semantic_study_search.api.schemas import (
    ErrorResponse,
    HealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from semantic_study_search.api.service import (
    QueryLimitError,
    SearchBusyError,
    SearchService,
    load_service,
)
from semantic_study_search.api.settings import APISettings
from semantic_study_search.limits import MAX_QUERY_CHARACTERS, MAX_RESULTS

logger = logging.getLogger(__name__)


def create_app(
    settings: APISettings | None = None,
    *,
    service_loader: Callable[[APISettings], SearchService] = load_service,
) -> FastAPI:
    """Создаёт API с загрузкой модели при запуске процесса.

    Args:
        settings: Настройки сервера; None читает переменные окружения.
        service_loader: Функция однократной загрузки библиотеки и модели.

    Returns:
        Приложение FastAPI.
    """
    settings = settings if settings is not None else APISettings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Загружает поисковый сервис на время работы API.

        Args:
            application: Запускаемое приложение.

        Returns:
            Контекст с загруженным поисковым сервисом.
        """
        try:
            application.state.search_service = service_loader(settings)
        except Exception:
            logger.exception("Не удалось загрузить поисковую библиотеку")
            application.state.search_service = None
        try:
            yield
        finally:
            application.state.search_service = None

    application = FastAPI(
        title="Semantic Study Search",
        description="Поиск фрагментов в общей библиотеке учебных материалов.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.search_service = None

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        """Возвращает понятное сообщение о неправильном запросе.

        Args:
            request: Входящий HTTP-запрос.
            error: Ошибки проверки тела запроса.

        Returns:
            Ответ 422 с кратким описанием ошибки.
        """
        detail = (
            f"Укажите непустой вопрос до {MAX_QUERY_CHARACTERS} символов "
            f"и целое количество результатов от 1 до {MAX_RESULTS}."
        )
        if any(item["type"] == "json_invalid" for item in error.errors()):
            detail = "Тело запроса должно содержать корректный JSON."
        return JSONResponse(status_code=422, content={"detail": detail})

    @application.get(
        "/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}}
    )
    def health(response: Response) -> HealthResponse:
        """Сообщает о готовности модели и индекса.

        Args:
            response: HTTP-ответ для установки статуса готовности.

        Returns:
            Состояние поиска и размер библиотеки.
        """
        service = application.state.search_service
        if service is None:
            response.status_code = 503
            return HealthResponse(status="unavailable", documents=0, fragments=0)
        return HealthResponse(
            status="ready", documents=service.document_count, fragments=service.fragment_count
        )

    @application.post(
        "/search",
        response_model=SearchResponse,
        responses={code: {"model": ErrorResponse} for code in (422, 500, 503)},
    )
    def search(payload: SearchRequest) -> SearchResponse:
        """Находит фрагменты по вопросу пользователя.

        Args:
            payload: Вопрос и максимальное количество результатов.

        Returns:
            Обработанный вопрос и ранжированные фрагменты.
        """
        service = application.state.search_service
        if service is None:
            raise HTTPException(503, "Поиск временно недоступен. Попробуйте позже.")
        try:
            results = service.search(payload.query, payload.top_k)
            return SearchResponse(
                query=payload.query,
                results=[SearchHit.model_validate(result) for result in results],
            )
        except QueryLimitError as error:
            raise HTTPException(422, str(error)) from error
        except SearchBusyError as error:
            raise HTTPException(503, str(error), headers={"Retry-After": "1"}) from error
        except Exception as error:
            logger.exception("Ошибка выполнения поиска")
            raise HTTPException(500, "Не удалось выполнить поиск. Попробуйте ещё раз.") from error

    return application


app = create_app()
