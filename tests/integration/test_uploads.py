"""Проверки загрузки, фоновой обработки и переключения живой библиотеки."""

import json
from threading import Event
from time import monotonic, sleep
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from semantic_study_search.api.app import create_app
from semantic_study_search.api.service import SearchService
from semantic_study_search.api.settings import APISettings
from semantic_study_search.api.uploads import UploadManager
from semantic_study_search.storage.index import load_index

pytestmark = pytest.mark.integration


@pytest.fixture
def upload_service(tmp_path, encoder):
    """Создаёт пустую библиотеку с кодировщиком произвольного числа фрагментов.

    Args:
        tmp_path: Временное хранилище.
        encoder: Тестовый кодировщик.

    Returns:
        Сервис с доступным для проверки кодировщиком.
    """
    encoder.encode_passages = Mock(
        side_effect=lambda texts: np.tile([1, 0], (len(texts), 1)).astype(np.float32)
    )
    return SearchService(None, encoder, index_dir=tmp_path, min_score=0.8)


@pytest.fixture
def upload_client(upload_service):
    """Запускает API с временной библиотекой.

    Args:
        upload_service: Пустая библиотека.

    Returns:
        HTTP-клиент работающего приложения.
    """
    settings = APISettings(index_dir=upload_service.index_dir)
    with TestClient(create_app(settings, service_loader=lambda _: upload_service)) as client:
        yield client


def wait_material(client, identifier, timeout=10):
    """Дожидается завершения фоновой задачи через публичный API.

    Args:
        client: HTTP-клиент приложения.
        identifier: Идентификатор загрузки.
        timeout: Максимальное ожидание в секундах.

    Returns:
        Итоговое состояние материала.
    """
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        response = client.get(f"/materials/{identifier}")
        assert response.status_code == 200
        state = response.json()
        if state["status"] in {"ready", "failed"}:
            return state
        sleep(0.01)
    pytest.fail("Материал не завершил обработку за отведённое время")


def post_material(client, text="Текст о семантическом поиске.", filename="lecture.txt"):
    """Отправляет текстовый материал и проверяет принятие задачи.

    Args:
        client: HTTP-клиент приложения.
        text: Текст документа.
        filename: Имя и формат файла.

    Returns:
        Идентификатор фоновой задачи.
    """
    response = client.post("/materials", files={"file": (filename, text.encode())})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_upload_search_download_without_restart(upload_client, upload_service):
    """Новый файл сразу находится и скачивается без перезапуска API.

    Args:
        upload_client: Работающий API.
        upload_service: Пустая библиотека.
    """
    assert upload_client.get("/health").json()["documents"] == 0
    response = upload_client.post("/search", json={"query": "поиск по смыслу"})
    assert response.status_code == 409
    assert "Загрузите материал" in response.json()["detail"]
    assert upload_service.encoder.query_calls == 0
    text = "Семантический поиск сравнивает смысл запроса и текста."
    identifier = post_material(upload_client, text, "лекция.txt")
    state = wait_material(upload_client, identifier)
    assert state["status"] == "ready"
    assert state["title"] == "лекция"
    assert state["documents"] == state["fragments"] == 1
    results = upload_client.post("/search", json={"query": "поиск по смыслу"}).json()["results"]
    assert results[0]["chunk"]["text"] == text
    assert results[0]["chunk"]["file_name"] == "лекция.txt"
    source = results[0]["chunk"]["source_url"]
    assert source == f"/materials/{identifier}/file"
    original = upload_client.get(source)
    assert original.content == text.encode()
    assert original.headers["content-disposition"].startswith("attachment;")
    assert original.headers["x-content-type-options"] == "nosniff"
    assert upload_service.encoder.encode_passages.call_count == 1


def test_append_reuses_old_vectors_and_avoids_duplicates(upload_client, upload_service):
    """Добавление сохраняет старые векторы, а повтор файла не размножает документы.

    Args:
        upload_client: Работающий API.
        upload_service: Пустая библиотека.
    """
    first = post_material(upload_client)
    assert wait_material(upload_client, first)["status"] == "ready"
    vectors = upload_service.index.vectors.copy()
    assert post_material(upload_client) == first
    second = post_material(upload_client, "# Новая лекция\n\nДругой текст.", "lecture.md")
    assert wait_material(upload_client, second)["status"] == "ready"
    assert upload_service.counts() == (2, 2)
    np.testing.assert_array_equal(upload_service.index.vectors[:1], vectors)
    assert [
        len(call.args[0]) for call in upload_service.encoder.encode_passages.call_args_list
    ] == [1, 1]
    assert "#" not in upload_service.index.documents[1].text


@pytest.mark.parametrize("failure", ["extract", "encode", "save"])
def test_failed_upload_preserves_search_and_current(
    upload_client, upload_service, monkeypatch, failure
):
    """Ошибка подготовки не заменяет рабочий индекс и не публикует оригинал.

    Args:
        upload_client: Работающий API.
        upload_service: Библиотека с подменяемой моделью.
        monkeypatch: Подмена этапа обработки.
        failure: Этап с имитируемым сбоем.
    """
    first = post_material(upload_client)
    assert wait_material(upload_client, first)["status"] == "ready"
    pointer = (upload_service.index_dir / "CURRENT").read_bytes()
    if failure == "encode":
        upload_service.encoder.encode_passages.side_effect = RuntimeError("private model path")
    elif failure == "save":
        monkeypatch.setattr(
            "semantic_study_search.api.service.save_index",
            Mock(side_effect=OSError("private storage path")),
        )
    identifier = post_material(upload_client, "   " if failure == "extract" else "Новый текст")
    state = wait_material(upload_client, identifier)
    assert state["status"] == "failed"
    assert "private" not in state["error"]
    assert (upload_service.index_dir / "CURRENT").read_bytes() == pointer
    assert upload_service.counts() == (1, 1)
    assert upload_client.get(f"/materials/{identifier}/file").status_code == 404
    result = upload_client.post("/search", json={"query": "поиск"}).json()["results"]
    assert result[0]["chunk"]["document_id"] == first


def test_processing_keeps_status_available_and_rejects_parallel_upload(
    upload_client, upload_service
):
    """Во время вычислений API показывает прогресс и ограничивает доступ к модели.

    Args:
        upload_client: Работающий API.
        upload_service: Библиотека с задерживаемой моделью.
    """
    started, release = Event(), Event()

    def encode(texts):
        """Задерживает модель до завершения проверок.

        Args:
            texts: Фрагменты материала.

        Returns:
            Векторы фрагментов.
        """
        started.set()
        assert release.wait(5)
        return np.tile([1, 0], (len(texts), 1)).astype(np.float32)

    upload_service.encoder.encode_passages.side_effect = encode
    try:
        identifier = post_material(upload_client)
        assert started.wait(5)
        assert upload_client.get("/health").status_code == 200
        assert upload_client.get(f"/materials/{identifier}").json()["status"] == "embedding"
        assert upload_client.get(f"/materials/{identifier}/file").status_code == 404
        assert post_material(upload_client) == identifier
        response = upload_client.post("/materials", files={"file": ("new.txt", b"another")})
        assert response.status_code == 409
        assert upload_client.post("/search", json={"query": "текст"}).status_code == 503
        assert upload_client.delete(f"/materials/{identifier}").status_code == 409
    finally:
        release.set()
    assert wait_material(upload_client, identifier)["status"] == "ready"


def test_upload_endpoints_work_without_authorization(upload_client):
    """Загрузка, статус и общая история доступны без авторизации.

    Args:
        upload_client: Работающий API.
    """
    assert upload_client.get("/materials").json() == {"materials": []}
    identifier = post_material(upload_client)
    assert wait_material(upload_client, identifier)["status"] == "ready"
    assert upload_client.get("/materials").json()["materials"][0]["id"] == identifier
    assert upload_client.get("/materials/unknown").status_code == 404
    assert upload_client.post("/materials", content=b"broken").status_code == 422


def test_upload_endpoints_report_unavailable_service():
    """Сбой запуска возвращает 503 для загрузки, состояния и истории."""
    loader = Mock(side_effect=RuntimeError("model failed"))
    with TestClient(create_app(service_loader=loader)) as client:
        assert (
            client.post("/materials", files={"file": ("lecture.txt", b"text")}).status_code == 503
        )
        assert client.get("/materials").status_code == 503
        assert client.get("/materials/unknown").status_code == 503
        assert client.delete("/materials/unknown").status_code == 503


@pytest.mark.parametrize(
    "filename,content,title",
    [("file.exe", b"text", ""), ("empty.txt", b"", ""), ("ok.txt", b"text", "x" * 201)],
)
def test_invalid_upload_rejected_before_model(
    upload_client, upload_service, filename, content, title
):
    """Некорректные файлы и названия отклоняются до запуска модели.

    Args:
        upload_client: Работающий API.
        upload_service: Пустая библиотека.
        filename: Имя проверяемого файла.
        content: Байты файла.
        title: Название материала.
    """
    response = upload_client.post(
        "/materials", files={"file": (filename, content)}, data={"title": title}
    )
    assert response.status_code == 422
    upload_service.encoder.encode_passages.assert_not_called()


def test_oversized_stream_rejected(upload_client, monkeypatch):
    """Размер ограничивается по полученным байтам даже без Content-Length.

    Args:
        upload_client: Работающий API.
        monkeypatch: Уменьшение лимита для теста.
    """
    monkeypatch.setattr("semantic_study_search.api.app.MAX_UPLOAD_BYTES", 1024)
    body = b'--abc\r\nContent-Disposition: form-data; name="file"; filename="x.txt"\r\n\r\n'
    body += b"a" * 70000 + b"\r\n--abc--\r\n"
    response = upload_client.post(
        "/materials",
        content=iter([body[:100], body[100:]]),
        headers={"Content-Type": "multipart/form-data; boundary=abc"},
    )
    assert response.status_code == 413


def test_restart_restores_downloads_and_committed_status(upload_client, upload_service):
    """Сохранённый индекс восстанавливает готовность даже после сбоя записи статуса.

    Args:
        upload_client: Работающий API.
        upload_service: Временная библиотека.
    """
    identifier = post_material(upload_client)
    assert wait_material(upload_client, identifier)["status"] == "ready"
    manager = upload_client.app.state.upload_manager
    manager.close()
    metadata = manager.directory / identifier / "metadata.json"
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record["status"] = "saving"
    metadata.write_text(json.dumps(record), encoding="utf-8")
    restored = SearchService(load_index(upload_service.index_dir), upload_service.encoder)
    recovered = UploadManager(restored, manager.directory)
    try:
        assert recovered.get(identifier).status == "ready"
        assert (
            recovered.original(identifier)[0].read_bytes()
            == "Текст о семантическом поиске.".encode()
        )
        assert len(restored.search("поиск", 5)) == 1
        assert upload_service.encoder.encode_passages.call_count == 1
    finally:
        recovered.close()


def test_interrupted_job_recovers_as_failed(upload_client, upload_service):
    """Незавершённая загрузка после запуска разрешает повторную попытку.

    Args:
        upload_client: Работающий API.
        upload_service: Временная библиотека.
    """
    identifier = post_material(upload_client)
    assert wait_material(upload_client, identifier)["status"] == "ready"
    manager = upload_client.app.state.upload_manager
    manager.close()
    empty_service = SearchService(None, upload_service.encoder, index_dir=upload_service.index_dir)
    recovered = UploadManager(empty_service, manager.directory)
    try:
        assert recovered.get(identifier).status == "failed"
        assert (
            recovered.submit("lecture.txt", "", "Текст о семантическом поиске.".encode()).id
            != identifier
        )
    finally:
        recovered.close()


def test_second_writer_is_unavailable(upload_client, upload_service):
    """Два процесса не могут одновременно менять один каталог библиотеки.

    Args:
        upload_client: Первый работающий API.
        upload_service: Библиотека с занятым каталогом.
    """
    with TestClient(create_app(service_loader=lambda _: upload_service)) as second:
        assert second.get("/health").status_code == 503
        assert upload_client.get("/health").status_code == 200


def test_upload_contract_and_history(upload_client):
    """Swagger описывает публичную загрузку, а история содержит готовые файлы.

    Args:
        upload_client: Работающий API.
    """
    identifier = post_material(upload_client)
    assert wait_material(upload_client, identifier)["status"] == "ready"
    assert upload_client.get("/materials").json()["materials"][0]["id"] == identifier
    assert upload_client.get("/materials/missing").status_code == 404
    schema = upload_client.get("/openapi.json").json()
    route = schema["paths"]["/materials"]["post"]
    assert "multipart/form-data" in route["requestBody"]["content"]
    assert "securitySchemes" not in schema["components"]
    assert "security" not in route


def test_relaxed_search_is_explicit(upload_client, upload_service):
    """Слабые совпадения показываются только при явном расширении выдачи.

    Args:
        upload_client: Работающий API.
        upload_service: Библиотека с изменяемым вектором вопроса.
    """
    identifier = post_material(upload_client)
    assert wait_material(upload_client, identifier)["status"] == "ready"
    upload_service.encoder.query = np.array([0, 1], dtype=np.float32)
    assert upload_client.post("/search", json={"query": "другая тема"}).json()["results"] == []
    response = upload_client.post("/search", json={"query": "другая тема", "relaxed": True})
    assert len(response.json()["results"]) == 1


def test_pdf_upload_preserves_pages_and_original(upload_client, tmp_path):
    """PDF индексируется постранично и сохраняет исходный файл целиком.

    Args:
        upload_client: Работающий API.
        tmp_path: Каталог тестового PDF.
    """
    from tests.unit.test_files import write_pdf

    path = tmp_path / "lecture.pdf"
    write_pdf(
        path, ["Semantic search compares text meanings.", None, "Vectors represent fragments."]
    )
    original = path.read_bytes()
    response = upload_client.post(
        "/materials",
        files={"file": (path.name, original)},
        data={"title": "Лекция PDF"},
    )
    assert response.status_code == 202
    identifier = response.json()["id"]
    state = wait_material(upload_client, identifier)
    assert state["status"] == "ready"
    assert state["documents"] == 2
    assert len(state["warnings"]) == 1
    hits = upload_client.post("/search", json={"query": "text"}).json()["results"]
    assert [hit["chunk"]["source_url"] for hit in hits] == [
        f"/materials/{identifier}/file#page=1",
        f"/materials/{identifier}/file#page=3",
    ]
    assert upload_client.get(f"/materials/{identifier}/file").content == original


def test_wrong_text_encoding_reports_actionable_error(upload_client):
    """Неподходящая кодировка даёт понятную ошибку без изменения библиотеки.

    Args:
        upload_client: Работающий API.
    """
    response = upload_client.post(
        "/materials", files={"file": ("legacy.txt", "Текст".encode("cp1251"))}
    )
    state = wait_material(upload_client, response.json()["id"])
    assert state["status"] == "failed"
    assert "UTF-8" in state["error"]
    assert upload_client.get("/health").json()["documents"] == 0


def test_delete_pdf_preserves_other_material_and_vectors(upload_client, upload_service, tmp_path):
    """Удаление убирает все страницы PDF и сохраняет другой материал без пересчёта.

    Args:
        upload_client: API с временной библиотекой.
        upload_service: Сервис с проверяемым кодировщиком.
        tmp_path: Каталог тестового PDF.
    """
    from tests.unit.test_files import write_pdf

    kept_text = "Оставшийся материал о поиске."
    kept = post_material(upload_client, kept_text)
    assert wait_material(upload_client, kept)["status"] == "ready"
    vectors = upload_service.index.vectors.copy()
    pdf = tmp_path / "pages.pdf"
    write_pdf(pdf, ["First page.", "Second page."])
    response = upload_client.post("/materials", files={"file": (pdf.name, pdf.read_bytes())})
    identifier = response.json()["id"]
    assert wait_material(upload_client, identifier)["documents"] == 2

    response = upload_client.delete(f"/materials/{identifier}")
    assert response.status_code == 204
    assert response.content == b""
    assert upload_service.counts() == (1, 1)
    assert upload_service.material_counts(identifier) == (0, 0)
    assert upload_service.encoder.encode_passages.call_count == 2
    np.testing.assert_array_equal(upload_service.index.vectors, vectors)
    assert upload_client.get(f"/materials/{identifier}").status_code == 404
    assert upload_client.get(f"/materials/{identifier}/file").status_code == 404
    assert upload_client.get(f"/materials/{kept}/file").content == kept_text.encode()
    assert [row["id"] for row in upload_client.get("/materials").json()["materials"]] == [kept]
    hits = upload_client.post("/search", json={"query": "текст"}).json()["results"]
    assert hits and all(hit["chunk"]["source_url"] == f"/materials/{kept}/file" for hit in hits)
    path = upload_service.index_dir
    assert not (path / "uploads" / identifier).exists()
    assert len(list((path / "versions").iterdir())) == 1
    assert load_index(path).documents == upload_service.index.documents
    assert upload_client.delete(f"/materials/{identifier}").status_code == 404
    route = upload_client.get("/openapi.json").json()["paths"]["/materials/{material_id}"]["delete"]
    assert "204" in route["responses"]
    assert "security" not in route


@pytest.mark.parametrize("failed", [False, True])
def test_delete_last_upload_survives_restart_and_allows_reupload(
    upload_client, upload_service, failed
):
    """Пустая библиотека сохраняется после удаления, а тот же файл можно загрузить снова.

    Args:
        upload_client: API с временной библиотекой.
        upload_service: Поиск и каталог индекса.
        failed: Удалить ошибочную загрузку вместо готовой.
    """
    from semantic_study_search.api.service import EmptyLibraryError

    text = "   " if failed else "Текст о поиске."
    identifier = post_material(upload_client, text)
    assert wait_material(upload_client, identifier)["status"] == ("failed" if failed else "ready")
    assert upload_client.delete(f"/materials/{identifier}").status_code == 204
    assert upload_client.get("/materials").json() == {"materials": []}
    assert upload_client.get("/health").json()["documents"] == 0
    assert upload_client.post("/search", json={"query": "вопрос"}).status_code == 409
    assert upload_service.encoder.query_calls == 0
    manager = upload_client.app.state.upload_manager
    manager.close()
    path = upload_service.index_dir
    index = load_index(path) if (path / "CURRENT").exists() else None
    restored = SearchService(index, upload_service.encoder, index_dir=path)
    recovered = UploadManager(restored, manager.directory)
    try:
        assert recovered.list_materials() == []
        assert restored.counts() == (0, 0)
        with pytest.raises(EmptyLibraryError):
            restored.search("вопрос", 5)
        repeated = recovered.submit("lecture.txt", "", text.encode())
        assert repeated.id != identifier
    finally:
        recovered.close()
    assert recovered.get(repeated.id).status == ("failed" if failed else "ready")
    assert restored.counts() == ((0, 0) if failed else (1, 1))


def test_failed_delete_keeps_original_and_search(upload_client, upload_service, monkeypatch):
    """Сбой записи индекса сохраняет прежний поиск, оригинал и состояние загрузки.

    Args:
        upload_client: API с временной библиотекой.
        upload_service: Сервис с сохранённым индексом.
        monkeypatch: Подмена записи индекса.
    """
    identifier = post_material(upload_client)
    assert wait_material(upload_client, identifier)["status"] == "ready"
    path = upload_service.index_dir
    previous = (path / "CURRENT").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(
            "semantic_study_search.api.service.save_index", Mock(side_effect=OSError("disk full"))
        )
        assert upload_client.delete(f"/materials/{identifier}").status_code == 500
    assert (path / "CURRENT").read_bytes() == previous
    assert upload_client.get(f"/materials/{identifier}/file").status_code == 200
    assert upload_client.get(f"/materials/{identifier}").json()["status"] == "ready"
    assert upload_client.post("/search", json={"query": "текст"}).json()["results"]
    assert not list((path / "uploads" / ".deletions").iterdir())
    assert upload_client.delete(f"/materials/{identifier}").status_code == 204


@pytest.mark.parametrize("committed", [False, True])
def test_interrupted_delete_finishes_on_restart(upload_client, upload_service, committed):
    """Записанное намерение удаления завершается после перезапуска без восстановления файла.

    Args:
        upload_client: API с временной библиотекой.
        upload_service: Поиск и сохранённый индекс.
        committed: Переключить индекс до имитации прерывания.
    """
    identifier = post_material(upload_client)
    assert wait_material(upload_client, identifier)["status"] == "ready"
    manager = upload_client.app.state.upload_manager
    manager.close()
    marker = manager.directory / ".deletions" / identifier
    marker.touch()
    if committed:
        upload_service.remove_material(identifier)
    path = upload_service.index_dir
    restored = SearchService(load_index(path), upload_service.encoder, index_dir=path)
    recovered = UploadManager(restored, manager.directory)
    try:
        assert restored.counts() == (0, 0)
        assert recovered.list_materials() == []
        assert not (manager.directory / identifier).exists()
        assert not marker.exists()
        assert len(list((path / "versions").iterdir())) == 1
        assert load_index(path).documents == ()
    finally:
        recovered.close()


def test_delete_cleanup_can_be_retried(upload_client, upload_service, monkeypatch):
    """Сбой очистки не возвращает файл в поиск и допускает повторное удаление.

    Args:
        upload_client: API с временной библиотекой.
        upload_service: Поиск и каталог индекса.
        monkeypatch: Подмена удаления файлов с диска.
    """
    identifier = post_material(upload_client)
    assert wait_material(upload_client, identifier)["status"] == "ready"
    with monkeypatch.context() as patch:
        patch.setattr(
            "semantic_study_search.api.uploads.shutil.rmtree",
            Mock(side_effect=PermissionError("locked")),
        )
        assert upload_client.delete(f"/materials/{identifier}").status_code == 500
    assert upload_service.counts() == (0, 0)
    assert upload_client.get(f"/materials/{identifier}/file").status_code == 404
    assert upload_client.get(f"/materials/{identifier}").json()["status"] == "failed"
    marker = upload_service.index_dir / "uploads" / ".deletions" / identifier
    assert marker.is_file()
    assert upload_client.delete(f"/materials/{identifier}").status_code == 204
    assert not marker.exists()
    assert upload_client.get("/materials").json() == {"materials": []}


def test_delete_unknown_upload_does_not_change_library(upload_client, upload_service):
    """Неизвестный идентификатор не меняет файлы и индекс библиотеки.

    Args:
        upload_client: API с временной библиотекой.
        upload_service: Сервис с проверяемым индексом.
    """
    identifier = post_material(upload_client)
    assert wait_material(upload_client, identifier)["status"] == "ready"
    previous = (upload_service.index_dir / "CURRENT").read_bytes()
    assert upload_client.delete("/materials/unknown").status_code == 404
    assert upload_client.delete(f"/materials/{'0' * 32}").status_code == 404
    assert (upload_service.index_dir / "CURRENT").read_bytes() == previous
    assert upload_client.get(f"/materials/{identifier}/file").status_code == 200
