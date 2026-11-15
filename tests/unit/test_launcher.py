"""Проверки настройки и совместного жизненного цикла локальных сервисов."""

import os
import signal
import socket
import subprocess
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from semantic_study_search import launcher


def test_default_environment_needs_no_local_files(tmp_path, monkeypatch):
    """Настройки запуска работают без предварительной подготовки файлов.

    Args:
        tmp_path: Рабочий каталог запуска.
        monkeypatch: Подмена окружения.
    """
    monkeypatch.delenv("SSS_MODEL_CACHE", raising=False)
    monkeypatch.delenv("SSS_OFFLINE", raising=False)
    first = launcher.prepare_environment(tmp_path)
    second = launcher.prepare_environment(tmp_path, offline=True)
    assert list(tmp_path.iterdir()) == []
    assert first["SSS_MODEL_CACHE"] == str(tmp_path / ".cache/huggingface/hub")
    assert first["SSS_OFFLINE"] == "0"
    assert second["SSS_OFFLINE"] == "1"


def test_explicit_settings_are_preserved(tmp_path, monkeypatch):
    """Явные настройки модели и библиотеки сохраняются при запуске.

    Args:
        tmp_path: Рабочий каталог запуска.
        monkeypatch: Подмена окружения.
    """
    monkeypatch.setenv("SSS_INDEX_DIR", "custom/library")
    monkeypatch.setenv("SSS_MODEL_CACHE", "custom/cache")
    monkeypatch.setenv("SSS_OFFLINE", "1")
    environment = launcher.prepare_environment(tmp_path)
    assert environment["SSS_INDEX_DIR"] == "custom/library"
    assert environment["SSS_MODEL_CACHE"] == "custom/cache"
    assert environment["SSS_OFFLINE"] == "1"


@pytest.mark.parametrize("ports", [(8000, 8000), (0, 8501), (8000, 65536)])
def test_invalid_ports_rejected(ports):
    """Некорректные или совпадающие порты отклоняются до запуска.

    Args:
        ports: Проверяемые порты API и интерфейса.
    """
    with pytest.raises(ValueError):
        launcher.check_ports(*ports)


def test_occupied_port_rejected_without_stopping_owner():
    """Занятый порт вызывает понятную ошибку и остаётся у прежнего процесса."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        other_port = 8501 if port != 8501 else 8502
        with pytest.raises(ValueError, match=f"Порт {port}"):
            launcher.check_ports(port, other_port)
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass


@pytest.fixture
def runtime(monkeypatch):
    """Подменяет внешние процессы для проверки порядка запуска и остановки.

    Args:
        monkeypatch: Подмена окружения запуска.

    Returns:
        Процессы, функции запуска, ожидания и остановки.
    """
    api, ui = Mock(), Mock()
    api.poll.return_value = ui.poll.return_value = None
    start = Mock(side_effect=[api, ui])
    ready, stop = Mock(), Mock()
    monkeypatch.setattr(launcher, "find_spec", Mock(return_value=object()))
    monkeypatch.setattr(launcher, "check_ports", Mock())
    monkeypatch.setattr(
        launcher,
        "prepare_environment",
        Mock(
            return_value={
                "SSS_INDEX_DIR": "custom/library",
                "SSS_API_URL": "http://old.example",
            }
        ),
    )
    monkeypatch.setattr(launcher, "start_process", start)
    monkeypatch.setattr(launcher, "wait_ready", ready)
    monkeypatch.setattr(launcher, "stop_processes", stop)
    monkeypatch.setattr(launcher.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    return api, ui, start, ready, stop


def test_launch_waits_for_both_services_and_ctrl_c_stops_them(runtime):
    """Команда ждёт готовности сервисов и завершает оба при Ctrl+C.

    Args:
        runtime: Подменённые процессы и ожидание.
    """
    api, ui, start, ready, stop = runtime
    assert launcher.main(["--api-port", "8123", "--ui-port", "8543"]) == 0
    assert ready.call_args_list[0].args == (api, "http://127.0.0.1:8123/health")
    assert ready.call_args_list[1].args == (ui, "http://127.0.0.1:8543/_stcore/health")
    api_environment = start.call_args_list[0].args[1]
    ui_environment = start.call_args_list[1].args[1]
    assert api_environment["SSS_INDEX_DIR"] == ui_environment["SSS_INDEX_DIR"] == "custom/library"
    assert ui_environment["SSS_API_URL"] == "http://127.0.0.1:8123"
    assert start.call_args_list[0].args[0][3] == "127.0.0.1"
    assert start.call_args_list[1].args[0][4] == "127.0.0.1"
    stop.assert_called_once_with([api, ui])


def test_container_host_keeps_internal_health_and_api_urls(runtime):
    """Контейнер открывает порты наружу и связывает сервисы через loopback.

    Args:
        runtime: Подменённые процессы и ожидание.
    """
    api, ui, start, ready, _ = runtime
    assert launcher.main(["--host", "0.0.0.0"]) == 0
    launcher.check_ports.assert_called_once_with(8000, 8501, host="0.0.0.0")
    assert start.call_args_list[0].args[0][3] == "0.0.0.0"
    assert start.call_args_list[1].args[0][4] == "0.0.0.0"
    assert start.call_args_list[1].args[1]["SSS_API_URL"] == "http://127.0.0.1:8000"
    assert ready.call_args_list[0].args == (api, "http://127.0.0.1:8000/health")
    assert ready.call_args_list[1].args == (ui, "http://127.0.0.1:8501/_stcore/health")


@pytest.mark.parametrize("during_startup", [False, True])
def test_sigterm_stops_children_and_restores_handlers(runtime, monkeypatch, during_startup):
    """SIGTERM останавливает созданные процессы даже во время загрузки модели.

    Args:
        runtime: Подменённые процессы и ожидание.
        monkeypatch: Подмена этапа получения сигнала.
        during_startup: Получить сигнал до запуска интерфейса.
    """
    api, ui, _, ready, stop = runtime
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def terminate(*args):
        """Вызывает зарегистрированный обработчик остановки.

        Args:
            *args: Аргументы подменённой операции.
        """
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    if during_startup:
        ready.side_effect = terminate
    else:
        monkeypatch.setattr(launcher.time, "sleep", terminate)
    assert launcher.main(["--host", "0.0.0.0"]) == 0
    stop.assert_called_once_with([api] if during_startup else [api, ui])
    for signum, handler in handlers.items():
        assert signal.getsignal(signum) == handler


def test_api_failure_prevents_ui_launch_and_cleans_up(runtime):
    """Сбой API отменяет запуск интерфейса и завершает созданный процесс.

    Args:
        runtime: Подменённые процессы и ожидание.
    """
    api, _, start, ready, stop = runtime
    ready.side_effect = RuntimeError("API failed")
    assert launcher.main([]) == 1
    assert start.call_count == 1
    stop.assert_called_once_with([api])


def test_ui_failure_also_stops_api(runtime):
    """Сбой запуска интерфейса не оставляет API работать отдельно.

    Args:
        runtime: Подменённые процессы и ожидание.
    """
    api, _, start, _, stop = runtime
    start.side_effect = [api, OSError("UI failed")]
    assert launcher.main([]) == 1
    stop.assert_called_once_with([api])


def test_unexpected_exit_stops_other_service(runtime):
    """Самопроизвольное завершение сервиса останавливает оставшийся процесс.

    Args:
        runtime: Подменённые процессы и ожидание.
    """
    api, ui, _, _, stop = runtime
    ui.poll.return_value = 1
    assert launcher.main([]) == 1
    stop.assert_called_once_with([api, ui])


def test_health_retries_connection_errors_then_accepts_ready(monkeypatch):
    """Запуск ожидает открытия порта, после чего принимает успешный health.

    Args:
        monkeypatch: Подмена локальных HTTP-запросов.
    """
    from unittest.mock import MagicMock

    process = Mock()
    process.poll.return_value = None
    response = MagicMock()
    response.__enter__.return_value.status = 200
    opener = Mock()
    opener.open.side_effect = [URLError("connection refused"), response]
    monkeypatch.setattr(launcher, "build_opener", Mock(return_value=opener))
    monkeypatch.setattr(launcher.time, "sleep", Mock())
    launcher.wait_ready(process, "http://127.0.0.1:8000/health")
    assert opener.open.call_count == 2


def test_unavailable_health_fails_without_waiting_for_timeout(monkeypatch):
    """Ошибка инициализации API останавливает ожидание сразу после HTTP 503.

    Args:
        monkeypatch: Подмена локального HTTP-запроса.
    """
    process = Mock()
    process.poll.return_value = None
    opener = Mock()
    opener.open.side_effect = HTTPError("http://127.0.0.1/health", 503, "failed", {}, None)
    monkeypatch.setattr(launcher, "build_opener", Mock(return_value=opener))
    with pytest.raises(RuntimeError, match="503"):
        launcher.wait_ready(process, "http://127.0.0.1/health")


def test_shutdown_signals_only_own_running_processes(monkeypatch):
    """Остановка посылает сигнал только работающим дочерним сервисам.

    Args:
        monkeypatch: Подмена системного сигнала.
    """
    running, exited = Mock(), Mock()
    running.poll.return_value = None
    exited.poll.return_value = 0
    kill_group = Mock()
    monkeypatch.setattr(launcher.os, "killpg", kill_group, raising=False)
    launcher.stop_processes([running, exited])
    exited.send_signal.assert_not_called()
    if os.name == "nt":
        running.send_signal.assert_called_once()
    else:
        kill_group.assert_called_once_with(running.pid, launcher.signal.SIGTERM)
    running.wait.assert_called_once()


def test_shutdown_forces_stuck_process_tree(monkeypatch):
    """Зависший сервис завершается вместе с дочерними процессами после тайм-аута.

    Args:
        monkeypatch: Подмена системных средств завершения.
    """
    process = Mock(pid=12345)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("service", 30), 0]
    terminate = Mock()
    kill_group = Mock()
    monkeypatch.setattr(launcher.subprocess, "run", terminate)
    monkeypatch.setattr(launcher.os, "killpg", kill_group, raising=False)
    launcher.stop_processes([process])
    if os.name == "nt":
        assert terminate.call_args.args[0] == ["taskkill", "/PID", "12345", "/T", "/F"]
    else:
        assert kill_group.call_args.args == (12345, launcher.signal.SIGKILL)
    assert process.wait.call_count == 2
