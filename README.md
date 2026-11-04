# Semantic Study Search

Семантический поиск по загруженным PDF, TXT и Markdown. Находит фрагменты документов по смыслу запроса через веб-интерфейс или API.

В приложении используется модель [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)

## Требования

- Git и [uv](https://docs.astral.sh/uv/getting-started/installation/) версии `0.11.29` или новее.
- Python **3.13**; uv устанавливает его по файлу `.python-version`.
- Windows x86-64 или Linux x86-64; модель работает на CPU.
- Доступ к GitHub, PyPI и `download.pytorch.org` для установки зависимостей.
- Доступ к Hugging Face при первом скачивании модели; веса занимают около 471 МБ.
- Постоянный диск для библиотеки и оригиналов, в том числе при размещении в облаке.

Поддерживаются TXT и Markdown в UTF-8, PDF с текстовым слоем: один файл до **20 МБ**,
до 300 страниц PDF, до 1 000 000 символов текста и до 1000 поисковых фрагментов.

## Инструкция установки

```shell
git clone https://github.com/AynurAkhmadullin/semantic-study-search.git
cd semantic-study-search
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
├── .github/workflows/ci.yml   # Проверки при push и pull request
├── .streamlit/config.toml    # Настройки интерфейса
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
