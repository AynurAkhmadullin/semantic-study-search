"""Проверки формы поиска и результатов Streamlit через AppTest."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from semantic_study_search.api.schemas import (
    HealthResponse,
    MaterialListResponse,
    MaterialResponse,
    SearchResponse,
)
from semantic_study_search.ui import client

pytestmark = pytest.mark.integration
APP_PATH = Path(__file__).parents[2] / "src/semantic_study_search/ui/app.py"


@pytest.fixture
def ui_backend(monkeypatch):
    """Подменяет HTTP-клиент готовыми ответами о состоянии и результатах поиска.

    Args:
        monkeypatch: Фикстура для временной подмены объектов и окружения.

    Returns:
        Подменённые функции проверки готовности и поиска.
    """
    health = Mock(return_value=HealthResponse(status="ready", documents=10, fragments=25))
    response = SearchResponse.model_validate(
        {
            "query": "тестовый вопрос",
            "results": [
                {
                    "score": 0.89,
                    "chunk": {
                        "id": "one:0",
                        "document_id": "one",
                        "title": "Токенизация",
                        "source_url": "https://example.org/material",
                        "text": "Токенизатор преобразует текст в последовательность токенов.",
                        "start": 0,
                        "end": 56,
                    },
                }
            ],
        }
    )
    search = Mock(return_value=response)
    monkeypatch.setattr(client, "get_health", health)
    monkeypatch.setattr(client, "search", search)
    return health, search


def submit(app, query):
    """Заполняет вопрос и отправляет форму поиска.

    Args:
        app: Тестовый экземпляр интерфейса Streamlit.
        query: Поисковый вопрос.

    Returns:
        Состояние интерфейса после отправки формы.
    """
    app.text_input[0].set_value(query)
    return next(button for button in app.button if button.label == "Найти").click().run()


def test_initial_form_does_not_search(ui_backend):
    """Открытие формы показывает начальные значения без выполнения поиска.

    Args:
        ui_backend: Подменённые вызовы проверки готовности и поиска.
    """
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert app.text_input[0].label == "Ваш вопрос"
    assert app.selectbox[0].value == 5
    ui_backend[1].assert_not_called()


def test_submit_displays_results_and_preserves_them_on_rerun(ui_backend):
    """Выдача сохраняется при повторном выполнении интерфейса без нового поиска.

    Args:
        ui_backend: Подменённые вызовы проверки готовности и поиска.
    """
    app = AppTest.from_file(APP_PATH).run()
    app.selectbox[0].set_value(3)
    submit(app, "тестовый вопрос")
    assert not app.exception
    ui_backend[1].assert_called_once_with("http://127.0.0.1:8000", "тестовый вопрос", 3)
    assert any("Токенизация" in item.value for item in app.subheader)
    assert any("Токенизатор преобразует" in item.value for item in app.markdown)
    app.run()
    assert ui_backend[1].call_count == 1
    assert any("Токенизация" in item.value for item in app.subheader)


def test_empty_question_shows_validation_message(ui_backend):
    """Пустой вопрос вызывает сообщение об ошибке без обращения к поиску.

    Args:
        ui_backend: Подменённые вызовы проверки готовности и поиска.
    """
    app = AppTest.from_file(APP_PATH).run()
    submit(app, "   ")
    assert not app.exception
    assert "Введите вопрос" in app.error[0].value
    ui_backend[1].assert_not_called()


def test_error_clears_old_results(ui_backend):
    """Ошибка нового поиска удаляет результаты предыдущего запроса.

    Args:
        ui_backend: Подменённые вызовы проверки готовности и поиска.
    """
    app = AppTest.from_file(APP_PATH).run()
    submit(app, "первый вопрос")
    ui_backend[1].side_effect = client.APIClientError("Нет связи с сервисом.")
    submit(app, "второй вопрос")
    assert not app.exception
    assert "Нет связи" in app.error[0].value
    assert not any("Токенизация" in item.value for item in app.subheader)


def test_service_unavailable_still_shows_form(ui_backend):
    """Недоступность API показывает предупреждение и сохраняет форму поиска.

    Args:
        ui_backend: Подменённые вызовы проверки готовности и поиска.
    """
    ui_backend[0].side_effect = client.APIClientError("offline")
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert "Нет связи" in app.warning[0].value
    assert app.text_input(key="query_input").label == "Ваш вопрос"


def test_example_only_prefills_question(ui_backend):
    """Выбор примера заполняет вопрос без выполнения поиска.

    Args:
        ui_backend: Подменённые вызовы проверки готовности и поиска.
    """
    app = AppTest.from_file(APP_PATH).run()
    next(button for button in app.button if button.label == "Что такое токенизация?").click().run()
    assert app.text_input[0].value == "Что такое токенизация?"
    ui_backend[1].assert_not_called()


def test_empty_results_message(ui_backend):
    """Пустая выдача сопровождается понятным сообщением.

    Args:
        ui_backend: Подменённые вызовы проверки готовности и поиска.
    """
    ui_backend[1].return_value = SearchResponse(query="вопрос", results=[])
    app = AppTest.from_file(APP_PATH).run()
    submit(app, "вопрос")
    assert not app.exception
    assert "не найдены" in app.info[0].value


def test_upload_from_ui_becomes_searchable(ui_backend, monkeypatch):
    """Форма отправляет файл и обновляет библиотеку после завершения обработки.

    Args:
        ui_backend: Подменённые вызовы поиска.
        monkeypatch: Подмена загрузки файла и фонового статуса.
    """
    file = Mock()
    file.name = "lecture.txt"
    file.getvalue.return_value = "Учебный материал".encode()
    monkeypatch.setattr("streamlit.file_uploader", Mock(return_value=file))
    material = MaterialResponse(id="a" * 32, file_name=file.name, title="Лекция", status="queued")
    upload = Mock(return_value=material)
    status = Mock(return_value=material.model_copy(update={"status": "embedding"}))
    monkeypatch.setattr(client, "upload_material", upload)
    monkeypatch.setattr(client, "get_material", status)
    app = AppTest.from_file(APP_PATH).run()
    assert [field.label for field in app.text_input] == [
        "Ваш вопрос",
        "Название материала (необязательно)",
    ]
    next(field for field in app.text_input if field.label.startswith("Название")).set_value(
        "Лекция"
    )
    next(button for button in app.button if button.label == "Загрузить материал").click().run()
    assert not app.exception
    upload.assert_called_once_with(
        "http://127.0.0.1:8000", "lecture.txt", file.getvalue(), "Лекция"
    )
    assert app.session_state["upload_job_id"] == material.id
    status.return_value = material.model_copy(
        update={"status": "ready", "documents": 1, "fragments": 2}
    )
    ui_backend[0].return_value = HealthResponse(status="ready", documents=11, fragments=27)
    app.run()
    assert not app.exception
    assert any("готов к поиску" in message.value for message in app.success)
    assert app.sidebar.metric[0].value == "11"
    submit(app, "Вопрос о лекции")
    ui_backend[1].assert_called_once_with("http://127.0.0.1:8000", "Вопрос о лекции", 5)


def test_upload_failure_is_shown(ui_backend, monkeypatch):
    """Ошибка фоновой обработки отображается и разрешает новую загрузку.

    Args:
        ui_backend: Подменённые вызовы поиска.
        monkeypatch: Подмена состояния материала.
    """
    material = MaterialResponse(
        id="b" * 32,
        file_name="scan.pdf",
        title="Скан",
        status="failed",
        error="Нет текстового слоя.",
    )
    monkeypatch.setattr(client, "get_material", Mock(return_value=material))
    app = AppTest.from_file(APP_PATH)
    app.session_state["upload_job_id"] = material.id
    app.run()
    assert not app.exception
    assert any("Нет текстового слоя" in message.value for message in app.error)
    assert not next(
        button for button in app.button if button.label == "Загрузить материал"
    ).disabled


def test_upload_requires_file_before_network(ui_backend, monkeypatch):
    """Форма без файла сообщает об ошибке без запроса загрузки.

    Args:
        ui_backend: Подменённые вызовы поиска.
        monkeypatch: Подмена HTTP-загрузки.
    """
    upload = Mock()
    monkeypatch.setattr(client, "upload_material", upload)
    app = AppTest.from_file(APP_PATH).run()
    next(button for button in app.button if button.label == "Загрузить материал").click().run()
    assert any("Выберите файл" in message.value for message in app.error)
    upload.assert_not_called()


def test_relaxed_checkbox_is_sent_to_api(ui_backend):
    """Переключатель расширенной выдачи передаётся в API.

    Args:
        ui_backend: Подменённые вызовы поиска.
    """
    app = AppTest.from_file(APP_PATH).run()
    app.checkbox[0].check()
    submit(app, "новая тема")
    ui_backend[1].assert_called_once_with("http://127.0.0.1:8000", "новая тема", 5, relaxed=True)


def test_upload_history_is_available_immediately(ui_backend, monkeypatch):
    """История загрузок открывается без ввода дополнительных данных.

    Args:
        ui_backend: Подменённые вызовы поиска.
        monkeypatch: Подмена загрузки истории.
    """
    history = Mock(return_value=MaterialListResponse(materials=[]))
    monkeypatch.setattr(client, "list_materials", history)
    app = AppTest.from_file(APP_PATH).run()
    button = next(button for button in app.button if button.label == "Загрузки")
    assert not button.disabled
    button.click().run()
    assert not app.exception
    history.assert_called_once_with("http://127.0.0.1:8000")
    assert any("Материалы ещё не загружены" in message.value for message in app.info)
