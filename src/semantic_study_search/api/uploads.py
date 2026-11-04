"""Фоновая обработка файлов и сохранение состояния загрузок."""

import hashlib
import json
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import uuid4

from semantic_study_search.api.schemas import MaterialResponse
from semantic_study_search.api.service import SearchService
from semantic_study_search.ingestion.files import load_material

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FILE_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
}
ACTIVE_STATES = {"queued", "extracting", "chunking", "embedding", "saving"}
logger = logging.getLogger(__name__)


class UploadBusyError(RuntimeError):
    """Сообщает, что другой файл уже обрабатывается."""


class UploadManager:
    """Подготавливает по одному материалу и хранит состояние между запусками."""

    def __init__(self, service: SearchService, directory: Path):
        """Восстанавливает сохранённые загрузки и создаёт фоновый обработчик.

        Args:
            service: Поиск с загруженной моделью.
            directory: Каталог оригиналов и состояния обработки.
        """
        self.service = service
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._records: dict[str, dict] = {}
        self._active: str | None = None
        self._closed = False
        for path in directory.glob("*/metadata.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                response = MaterialResponse.model_validate(record)
                if (
                    response.id != path.parent.name
                    or Path(response.file_name).suffix.lower() not in FILE_TYPES
                ):
                    raise ValueError("Некорректные метаданные загрузки.")
                if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                    raise ValueError("Некорректная контрольная сумма.")
                documents, fragments = service.material_counts(response.id)
                if documents:
                    record.update(
                        status="ready", documents=documents, fragments=fragments, error=None
                    )
                elif response.status in ACTIVE_STATES or response.status == "ready":
                    record.update(
                        status="failed",
                        error="Обработка прервана или материал отсутствует в текущей библиотеке. Загрузите файл повторно.",
                    )
                self._records[response.id] = record
                self._persist(record)
            except (OSError, ValueError, TypeError, KeyError):
                logger.exception("Не удалось восстановить метаданные загрузки %s", path.parent.name)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="material-upload")

    def _persist(self, record: dict) -> None:
        """Атомарно записывает состояние одной загрузки.

        Args:
            record: Метаданные и состояние материала.
        """
        path = self.directory / record["id"] / "metadata.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def submit(self, filename: str, title: str, content: bytes) -> MaterialResponse:
        """Сохраняет файл и запускает обработку без ожидания индексации.

        Args:
            filename: Исходное имя файла.
            title: Название; пустое значение использует имя файла.
            content: Полученные байты материала.

        Returns:
            Новая задача или уже существующая загрузка тех же байтов.
        """
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
        extension = Path(filename).suffix.lower()
        if (
            extension not in FILE_TYPES
            or not filename
            or len(filename) > 180
            or any(ord(c) < 32 for c in filename)
        ):
            raise ValueError("Выберите файл PDF, TXT или Markdown с корректным именем.")
        title = title.strip() or Path(filename).stem
        if len(title) > 200 or any(ord(c) < 32 for c in title):
            raise ValueError("Название должно содержать не более 200 символов без переносов строк.")
        if not content:
            raise ValueError("Загруженный файл пуст.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("Размер файла превышает 20 МБ.")
        digest = hashlib.sha256(content).hexdigest()
        with self._lock:
            if self._closed:
                raise UploadBusyError("Сервис завершает работу. Попробуйте позже.")
            for record in self._records.values():
                if (
                    record["sha256"] == digest
                    and Path(record["file_name"]).suffix.lower() == extension
                    and record["status"] != "failed"
                ):
                    return MaterialResponse.model_validate(record)
            if self._active is not None:
                raise UploadBusyError(
                    "Другой материал уже обрабатывается. Дождитесь его готовности."
                )
            identifier = uuid4().hex
            directory = self.directory / identifier
            directory.mkdir()
            with (directory / ("source" + extension)).open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            record = MaterialResponse(
                id=identifier, file_name=filename, title=title, status="queued"
            ).model_dump() | {"sha256": digest}
            self._persist(record)
            self._records[identifier] = record
            self._active = identifier
            try:
                self._executor.submit(self._process, identifier)
            except Exception:
                self._active = None
                record.update(
                    status="failed",
                    error="Не удалось запустить обработку. Загрузите файл повторно.",
                )
                self._persist(record)
                raise
            return MaterialResponse.model_validate(record)

    def _stage(self, identifier: str, status: str) -> None:
        """Сохраняет текущий этап обработки.

        Args:
            identifier: Идентификатор загрузки.
            status: Новый этап.
        """
        with self._lock:
            record = self._records[identifier]
            record["status"] = status
            self._persist(record)

    def _process(self, identifier: str) -> None:
        """Извлекает текст, добавляет индекс и фиксирует результат задачи.

        Args:
            identifier: Идентификатор сохранённого файла.
        """
        record = self._records[identifier]
        try:
            self._stage(identifier, "extracting")
            path = (
                self.directory / identifier / ("source" + Path(record["file_name"]).suffix.lower())
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                documents = load_material(
                    path,
                    {
                        "id": identifier,
                        "title": record["title"],
                        "source_url": f"/materials/{identifier}/file",
                    },
                )
            with self._lock:
                record["warnings"] = [str(item.message) for item in caught[:20]]
            self.service.add_material(documents, lambda status: self._stage(identifier, status))
            with self._lock:
                count, fragments = self.service.material_counts(identifier)
                record.update(status="ready", documents=count, fragments=fragments, error=None)
                self._persist(record)
                self._active = None
        except Exception as error:
            logger.exception("Ошибка обработки материала %s", identifier)
            with self._lock:
                count, fragments = self.service.material_counts(identifier)
                if count:
                    record.update(status="ready", documents=count, fragments=fragments, error=None)
                else:
                    if isinstance(error, UnicodeError):
                        message = "Не удалось прочитать текст. Сохраните файл в кодировке UTF-8."
                    elif isinstance(error, ValueError):
                        message = str(error)
                    else:
                        message = (
                            "Не удалось подготовить материал. Попробуйте загрузить его повторно."
                        )
                    record.update(status="failed", error=message)
                try:
                    self._persist(record)
                except OSError:
                    logger.exception("Не удалось сохранить состояние задачи %s", identifier)
                self._active = None
        finally:
            with self._lock:
                if self._active == identifier:
                    self._active = None

    def get(self, identifier: str) -> MaterialResponse:
        """Возвращает состояние задачи без ожидания обработки.

        Args:
            identifier: Идентификатор материала.

        Returns:
            Состояние подготовки материала.
        """
        with self._lock:
            return MaterialResponse.model_validate(self._records[identifier])

    def list_materials(self) -> list[MaterialResponse]:
        """Возвращает историю загрузок в обратном порядке.

        Returns:
            Сохранённые состояния материалов.
        """
        with self._lock:
            return [
                MaterialResponse.model_validate(record)
                for record in reversed(self._records.values())
            ]

    def original(self, identifier: str) -> tuple[Path, str, str]:
        """Находит оригинал только опубликованного материала.

        Args:
            identifier: Идентификатор загрузки.

        Returns:
            Путь, имя файла и MIME-тип.
        """
        record = self.get(identifier)
        if not self.service.material_counts(identifier)[0]:
            raise KeyError(identifier)
        extension = Path(record.file_name).suffix.lower()
        path = self.directory / identifier / ("source" + extension)
        if not path.is_file():
            raise KeyError(identifier)
        return path, record.file_name, FILE_TYPES[extension]

    def close(self) -> None:
        """Дожидается активной обработки перед завершением сервера."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)
