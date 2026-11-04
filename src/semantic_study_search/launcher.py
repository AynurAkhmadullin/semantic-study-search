"""Совместный локальный запуск API и Streamlit из одного терминала."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from importlib.util import find_spec
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener


def check_ports(api_port: int, ui_port: int) -> None:
    """Проверяет доступность двух локальных портов до запуска процессов.

    Args:
        api_port: Порт поискового API.
        ui_port: Порт веб-интерфейса.
    """
    if api_port == ui_port:
        raise ValueError("API и интерфейсу нужны разные порты.")
    if any(not 1 <= port <= 65535 for port in (api_port, ui_port)):
        raise ValueError("Порт должен быть числом от 1 до 65535.")
    with ExitStack() as stack:
        for port in (api_port, ui_port):
            server = stack.enter_context(socket.socket())
            if os.name == "nt":
                server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                server.bind(("127.0.0.1", port))
            except OSError as error:
                raise ValueError(
                    f"Порт {port} недоступен. Остановите прежний процесс "
                    "или задайте другие --api-port и --ui-port."
                ) from error


def prepare_environment(directory: Path, *, offline: bool = False) -> dict[str, str]:
    """Подготавливает окружение API с настройками кэша модели.

    Args:
        directory: Рабочий каталог проекта.
        offline: Использовать только скачанные веса модели.

    Returns:
        Окружение дочернего процесса API.
    """
    environment = os.environ.copy()
    environment.setdefault("SSS_MODEL_CACHE", str(directory / ".cache/huggingface/hub"))
    environment.setdefault("SSS_OFFLINE", "0")
    environment.setdefault("PYTHONUTF8", "1")
    if offline:
        environment["SSS_OFFLINE"] = "1"
    return environment


def start_process(arguments: list[str], environment: dict[str, str]) -> subprocess.Popen:
    """Запускает сервис в отдельной группе процессов с общим выводом в терминал.

    Args:
        arguments: Аргументы запуска Python-модуля.
        environment: Окружение сервиса.

    Returns:
        Дочерний процесс сервиса.
    """
    options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    return subprocess.Popen([sys.executable, "-u", "-m", *arguments], env=environment, **options)


def wait_ready(process: subprocess.Popen, url: str, *, timeout: float = 900) -> None:
    """Дожидается готовности сервиса или сообщает об ошибке запуска.

    Args:
        process: Запущенный сервис.
        url: Локальный адрес проверки готовности.
        timeout: Максимальное ожидание в секундах.
    """
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Сервис завершился при запуске. Проверьте сообщения выше.")
        try:
            with opener.open(url, timeout=1) as response:
                if response.status == 200:
                    return
        except HTTPError as error:
            raise RuntimeError(
                f"Сервис не готов: HTTP {error.code}. Проверьте сообщения выше."
            ) from error
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Истекло время ожидания запуска сервиса. Проверьте загрузку модели.")


def stop_processes(processes: list[subprocess.Popen]) -> None:
    """Останавливает созданные сервисы и дожидается освобождения их ресурсов.

    Args:
        processes: Процессы, принадлежащие текущему запуску.
    """
    for process in reversed(processes):
        if process.poll() is None:
            try:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            except OSError:
                # Без консоли Windows групповое прерывание может быть недоступно.
                pass
    deadline = time.monotonic() + 30
    for process in reversed(processes):
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main(argv: list[str] | None = None) -> int:
    """Запускает локальное приложение и завершает оба сервиса по Ctrl+C.

    Args:
        argv: Аргументы команды; None использует командную строку.

    Returns:
        Код завершения: 0 при остановке пользователем, 1 при ошибке.
    """
    parser = argparse.ArgumentParser(
        description="Запуск API и интерфейса одним процессом управления."
    )
    parser.add_argument("--api-port", type=int, default=8000, help="Порт API, по умолчанию 8000")
    parser.add_argument("--ui-port", type=int, default=8501, help="Порт UI, по умолчанию 8501")
    parser.add_argument("--offline", action="store_true", help="Использовать скачанные веса модели")
    args = parser.parse_args(argv)
    processes = []
    try:
        if any(
            find_spec(module) is None
            for module in ("uvicorn", "streamlit", "sentence_transformers")
        ):
            raise RuntimeError("Установите зависимости: uv sync --locked --all-extras")
        check_ports(args.api_port, args.ui_port)
        environment = prepare_environment(Path.cwd(), offline=args.offline)
        api_url = f"http://127.0.0.1:{args.api_port}"
        ui_url = f"http://127.0.0.1:{args.ui_port}"
        print(
            "Запускаем API и загружаем модель. Первый запуск может занять несколько минут…",
            flush=True,
        )
        api = start_process(
            [
                "uvicorn",
                "semantic_study_search.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.api_port),
            ],
            environment,
        )
        processes.append(api)
        wait_ready(api, api_url + "/health")
        ui_environment = environment.copy()
        ui_environment["SSS_API_URL"] = api_url
        ui = start_process(
            [
                "streamlit",
                "run",
                str(Path(__file__).parent / "ui/app.py"),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(args.ui_port),
                "--server.headless",
                "true",
            ],
            ui_environment,
        )
        processes.append(ui)
        wait_ready(ui, ui_url + "/_stcore/health")
        print(
            f"\nПриложение: {ui_url}\nДокументация API: {api_url}/docs\nОстановка: Ctrl+C",
            flush=True,
        )
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("Один из сервисов завершился. Приложение останавливается.")
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nОстанавливаем приложение…", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if processes:
            # Повторное нажатие не должно прерывать освобождение дочерних процессов.
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                stop_processes(processes)
            finally:
                signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    raise SystemExit(main())
