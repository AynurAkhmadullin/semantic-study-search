# Semantic Study Search

Семантический поиск по загруженным PDF, TXT и Markdown. Находит фрагменты документов по смыслу запроса через веб-интерфейс или API.

В приложении используется модель [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)

## Требования

- Для запуска через Docker — Docker Desktop с Linux-контейнерами или Docker Engine на Linux.
- Для запуска без Docker — [uv](https://docs.astral.sh/uv/getting-started/installation/) версии `0.11.29` или новее и Python **3.13**.
- Windows x86-64 или Linux x86-64.
- Доступ к GitHub, PyPI и `download.pytorch.org` для установки зависимостей.
- Доступ к Hugging Face при первом скачивании модели (около 471 МБ).
- Постоянный диск для библиотеки и оригиналов, в том числе при размещении в облаке.

Поддерживаются TXT и Markdown в UTF-8, PDF с текстовым слоем: один файл до **20 МБ**,
до 300 страниц PDF, до 1 000 000 символов текста и до 1000 поисковых фрагментов.

## Инструкция установки

```shell
git clone https://github.com/AynurAkhmadullin/semantic-study-search.git
cd semantic-study-search
```

### Запуск в Google Colab

Откройте [готовый блокнот](notebooks/semantic-study-search-colab.ipynb) через меню
Colab **Файл → Загрузить блокнот**. После отправки файла в GitHub доступно
[открытие из репозитория](https://colab.research.google.com/github/AynurAkhmadullin/semantic-study-search/blob/main/notebooks/semantic-study-search-colab.ipynb).

Выполните ячейки **«Подготовка»** и **«Запуск»** по очереди. Библиотека восстановится
из Google Drive, а интерфейс откроется под ячейкой. Исправление XSRF уже включено.

После работы выполните **«Сохранение»**: приложение остановится, библиотека
сохранится архивом в **Мой диск → semantic-study-search**. Дождитесь сообщения
об успехе. Без сохранения новые данные могут потеряться при отключении среды.

### Запуск через Docker

Один `Dockerfile` собирает образ с API и интерфейсом. Команды выполняются из корня
репозитория при работающем Docker Engine; Docker Compose не требуется.
Для первой сборки нужен доступ к Docker Hub, `ghcr.io` и источникам Python-зависимостей.

Сборка образа:

```shell
docker build --platform linux/amd64 -t semantic-study-search:local .
```

Запуск приложения в фоне:

```shell
docker run -d --name semantic-study-search --stop-timeout 45 -p 127.0.0.1:8501:8501 -p 127.0.0.1:8000:8000 --mount type=volume,source=semantic-study-data,target=/data semantic-study-search:local
```

Интерфейс доступен на [порту 8501](http://127.0.0.1:8501), документация API — на
[порту 8000](http://127.0.0.1:8000/docs). Первая загрузка модели может занять несколько
минут; интерфейс запускается после готовности API. Ход запуска можно увидеть командой:

```shell
docker logs -f semantic-study-search
```

`Ctrl+C` завершает просмотр логов. Для остановки и повторного запуска приложения:

```shell
docker stop semantic-study-search
docker start semantic-study-search
```

Том `semantic-study-data` хранит оригиналы и индекс в `/data/library`, а модель —
в `/data/huggingface/hub`. Он сохраняет данные после остановки или удаления контейнера;
при создании нового контейнера используйте то же имя тома. Новая библиотека в Docker
пуста: локальные материалы из рабочей папки не входят в образ.
После скачивания модели можно добавить `-e SSS_OFFLINE=1` перед именем образа в `docker run`.

Контейнер проверяет готовность API и UI; состояние видно в `docker ps`.
Внутренние порты — 8000 и 8501. Для других внешних портов меняйте левую часть
соответствия в `-p`, например `127.0.0.1:8502:8501` для интерфейса.

### Запуск без Docker

```shell
uv sync --locked --all-extras
```

Для локального запуска API и интерфейса выполните команду из корня репозитория:

```shell
uv run --locked --all-extras semantic-study-start
```

Приложение останавливаются нажатием `Ctrl+C`.

Первый запуск может занять несколько минут из-за скачивания модели. Когда веса
уже скачаны, можно запускать без сети:

```shell
uv run --locked --all-extras semantic-study-start --offline
```

По умолчанию приложение доступно на [порту 8501](http://127.0.0.1:8501),
API — на [порту 8000](http://127.0.0.1:8000/docs). Другие порты можно задать так:

```shell
uv run --locked --all-extras semantic-study-start --api-port 8001 --ui-port 8502
```

<details>
<summary>Раздельный запуск API и интерфейса</summary>

При необходимости сервисы можно запускать отдельно в двух терминалах.
В первом запустите API. Команды для PowerShell:

```powershell
$env:SSS_MODEL_CACHE = "$PWD/.cache/huggingface/hub"
$env:SSS_OFFLINE = "0"
uv run --locked --all-extras uvicorn semantic_study_search.api.app:app --host 127.0.0.1 --port 8000
```

Первый запуск скачивает закреплённую ревизию E5
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. После скачивания можно установить
`SSS_OFFLINE=1` для работы без сети. Дождитесь `Application startup complete`
и статуса `ready` на [странице состояния](http://127.0.0.1:8000/health).
Отсутствие индекса при первой установке допустимо: API создаёт пустую библиотеку.

Во втором терминале запустите интерфейс:

```powershell
$env:SSS_API_URL = "http://127.0.0.1:8000"
uv run --locked --all-extras streamlit run src/semantic_study_search/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

</details>

## Руководство по эксплуатации
Откройте [приложение](http://127.0.0.1:8501):

1. Раскройте «Добавить материал».
2. Выберите файл; при желании задайте название. Нажмите «Загрузить материал».
3. Дождитесь сообщения «готов к поиску»; этапы обработки обновляются автоматически.
4. Введите вопрос и нажмите «Найти». Оригинал доступен через «Скачать материал».

Повторная загрузка одного и того же файла не создаёт дубликат.

Для удаления откройте «Добавить материал» → «Загрузки», нажмите «Удалить материал» у нужного файла.
После удаления последнего файла поиск отключается до новой загрузки.

## API
API описан в [Swagger](http://127.0.0.1:8000/docs):

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/health` | Готовность и размер библиотеки |
| `POST` | `/search` | Поиск по вопросу |
| `POST` | `/materials` | Multipart: `file` и необязательный `title`; возвращает HTTP 202 и идентификатор |
| `GET` | `/materials` | История загрузок общей библиотеки |
| `GET` | `/materials/{id}` | Состояние обработки |
| `DELETE` | `/materials/{id}` | Удаление файла и его фрагментов; успешный ответ HTTP 204 |
| `GET` | `/materials/{id}/file` | Оригинал опубликованного материала |

## Зависимости

| Группа | Состав | Назначение |
| --- | --- | --- |
| `ml` | NumPy, Sentence Transformers, PyTorch, pypdf, markdown-it-py | Модель, поиск и извлечение текста |
| `api` | FastAPI, Pydantic, Uvicorn, python-multipart, filelock | HTTP, загрузка файлов и блокировка записи |
| `ui` | Streamlit, HTTPX, Pydantic | Интерфейс, HTTP-клиент и проверка ответов |
| `dev` | pytest, pytest-cov, HTTPX, Ruff | Тесты и проверка кода |

## Структура репозитория

```text
semantic-study-search/
├── .github/workflows/ci.yml # Проверки при push и pull request
├── .streamlit/config.toml   # Настройки интерфейса
├── Dockerfile               # Образ API и UI в одном контейнере
├── .dockerignore            # Исключения из контекста сборки
├── notebooks/
│   └── semantic-study-search-colab.ipynb  # Colab и архивы библиотеки в Drive
├── scripts/
│   └── evaluate.py          # Оценка поиска на своём контрольном наборе
├── src/semantic_study_search/
│   ├── __init__.py
│   ├── cli.py               # Поиск из терминала
│   ├── index_cli.py         # Подготовка библиотеки из терминала
│   ├── launcher.py          # Запуск API и UI одной командой
│   ├── domain.py            # Документ, фрагмент и результат
│   ├── evaluation.py        # Контрольные наборы и метрики
│   ├── limits.py            # Ограничения поискового запроса
│   ├── sources.py           # Проверка ссылок на загруженные оригиналы
│   ├── ingestion/           # Импорт, очистка и разбиение текста
│   ├── search/              # E5, ранжирование и фильтрация
│   ├── storage/             # SQLite, NumPy и версии индекса
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py           # Маршруты FastAPI
│   │   ├── schemas.py       # Контракты запросов и ответов
│   │   ├── service.py       # Модель и обновление библиотеки в памяти
│   │   ├── settings.py      # Переменные окружения
│   │   └── uploads.py       # Фоновые задачи и оригиналы файлов
│   └── ui/
│       ├── __init__.py
│       ├── app.py           # Поиск, загрузка и прогресс Streamlit
│       └── client.py        # HTTP-клиент API
├── tests/
│   ├── unit/                # Отдельные компоненты
│   └── integration/         # API, UI, загрузки и настоящая модель
├── data/
│   ├── raw/                 # Локальные исходные материалы
│   └── processed/           # Индекс, оригиналы и состояния загрузок
├── .editorconfig
├── .gitattributes
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```
