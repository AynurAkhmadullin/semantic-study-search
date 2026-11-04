"""HTTP-доступ к семантическому поиску через FastAPI."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from filelock import FileLock
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException

from semantic_study_search.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MaterialListResponse,
    MaterialResponse,
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
from semantic_study_search.api.uploads import MAX_UPLOAD_BYTES, UploadBusyError, UploadManager
from semantic_study_search.limits import MAX_QUERY_CHARACTERS, MAX_RESULTS
from semantic_study_search.sources import uploaded_source_id

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
        manager = None
        writer_lock = None
        try:
            if service_loader is load_service:
                settings.index_dir.mkdir(parents=True, exist_ok=True)
                writer_lock = FileLock(str(settings.index_dir / ".api.lock"))
                writer_lock.acquire(timeout=0)
            service = service_loader(settings)
            if service.index_dir is not None:
                if writer_lock is None:
                    service.index_dir.mkdir(parents=True, exist_ok=True)
                    writer_lock = FileLock(str(service.index_dir / ".api.lock"))
                    writer_lock.acquire(timeout=0)
                manager = UploadManager(
                    service, settings.upload_dir or service.index_dir / "uploads"
                )
            application.state.search_service = service
            application.state.upload_manager = manager
        except Exception:
            logger.exception("Не удалось загрузить поисковую библиотеку")
            application.state.search_service = None
        try:
            yield
        finally:
            if manager is not None:
                manager.close()
            if writer_lock is not None:
                writer_lock.release()
            application.state.search_service = None
            application.state.upload_manager = None

    application = FastAPI(
        title="Semantic Study Search",
        description="Поиск фрагментов в общей библиотеке учебных материалов.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.search_service = None
    application.state.upload_manager = None

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
        if request.url.path.startswith("/materials"):
            detail = "Проверьте параметры материала."
        elif any(item["type"] == "json_invalid" for item in error.errors()):
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
        documents, fragments = service.counts()
        return HealthResponse(status="ready", documents=documents, fragments=fragments)

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
            results = service.search(payload.query, payload.top_k, relaxed=payload.relaxed)
            hits = [SearchHit.model_validate(result) for result in results]
            manager = application.state.upload_manager
            if manager is not None:
                for hit in hits:
                    identifier = uploaded_source_id(str(hit.chunk.source_url))
                    if identifier is not None:
                        try:
                            hit.chunk.file_name = manager.get(identifier).file_name
                        except KeyError:
                            pass
            return SearchResponse(query=payload.query, results=hits)
        except QueryLimitError as error:
            raise HTTPException(422, str(error)) from error
        except SearchBusyError as error:
            raise HTTPException(503, str(error), headers={"Retry-After": "1"}) from error
        except Exception as error:
            logger.exception("Ошибка выполнения поиска")
            raise HTTPException(500, "Не удалось выполнить поиск. Попробуйте ещё раз.") from error

    def get_upload_manager() -> UploadManager:
        """Проверяет готовность обработки материалов.

        Returns:
            Работающий диспетчер загрузок.
        """
        manager = application.state.upload_manager
        if manager is None:
            raise HTTPException(503, "Загрузка временно недоступна.")
        return manager

    @application.post(
        "/materials",
        status_code=202,
        response_model=MaterialResponse,
        responses={code: {"model": ErrorResponse} for code in (409, 413, 422, 500, 503)},
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string", "format": "binary"},
                                "title": {"type": "string", "maxLength": 200},
                            },
                        }
                    }
                },
            }
        },
    )
    async def upload_material(request: Request) -> MaterialResponse:
        """Принимает файл и запускает фоновую подготовку библиотеки.

        Args:
            request: Multipart-запрос с файлом и необязательным названием.

        Returns:
            Состояние принятого материала.
        """
        manager = get_upload_manager()
        consumed = 0
        body_limit = MAX_UPLOAD_BYTES + 64 * 1024

        async def limited_receive() -> dict:
            """Ограничивает размер multipart-тела во время чтения.

            Returns:
                Следующее ASGI-сообщение тела запроса.
            """
            nonlocal consumed
            message = await request.receive()
            consumed += len(message.get("body", b""))
            if consumed > body_limit:
                raise MultiPartException("Размер файла превышает 20 МБ.")
            return message

        limited_request = Request(request.scope, receive=limited_receive)
        try:
            async with limited_request.form(max_files=1, max_fields=1, max_part_size=8192) as form:
                file = form.get("file")
                title = form.get("title", "")
                if not isinstance(file, UploadFile) or not isinstance(title, str):
                    raise HTTPException(422, "Выберите один файл и укажите текстовое название.")
                if set(form.keys()) - {"file", "title"}:
                    raise HTTPException(422, "Неизвестные поля загрузки.")
                content = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(content) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Размер файла превышает 20 МБ.")
                # Сохранение файла выполняется в потоке, чтобы не блокировать API.
                from starlette.concurrency import run_in_threadpool

                return await run_in_threadpool(manager.submit, file.filename or "", title, content)
        except UploadBusyError as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except HTTPException:
            raise
        except Exception as error:
            from starlette.exceptions import HTTPException as StarletteHTTPException

            if isinstance(error, StarletteHTTPException) and error.status_code == 400:
                status = 413 if consumed > body_limit else 422
                raise HTTPException(
                    status,
                    "Файл слишком большой."
                    if status == 413
                    else "Не удалось прочитать загруженный файл.",
                ) from error
            logger.exception("Ошибка приёма файла")
            raise HTTPException(500, "Не удалось сохранить файл. Попробуйте ещё раз.") from error

    @application.get("/materials", response_model=MaterialListResponse)
    def list_materials() -> MaterialListResponse:
        """Возвращает сохранённую историю загрузок общей библиотеки.

        Returns:
            Задачи подготовки и готовые материалы.
        """
        return MaterialListResponse(materials=get_upload_manager().list_materials())

    @application.get("/materials/{material_id}", response_model=MaterialResponse)
    def material_status(material_id: str) -> MaterialResponse:
        """Сообщает состояние обработки файла.

        Args:
            material_id: Идентификатор загрузки.

        Returns:
            Текущий этап или результат обработки.
        """
        try:
            return get_upload_manager().get(material_id)
        except KeyError as error:
            raise HTTPException(404, "Материал не найден.") from error

    @application.get("/materials/{material_id}/file")
    def download_material(material_id: str) -> FileResponse:
        """Отдаёт оригинал материала из общей поисковой библиотеки.

        Args:
            material_id: Идентификатор загруженного файла.

        Returns:
            Файл для скачивания.
        """
        manager = application.state.upload_manager
        if manager is None:
            raise HTTPException(503, "Библиотека временно недоступна.")
        try:
            path, filename, mime = manager.original(material_id)
        except KeyError as error:
            raise HTTPException(404, "Материал не найден.") from error
        return FileResponse(
            path, filename=filename, media_type=mime, headers={"X-Content-Type-Options": "nosniff"}
        )

    return application


app = create_app()
