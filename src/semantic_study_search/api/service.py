"""Поиск с единственным экземпляром модели в процессе сервера."""

from threading import Lock

from semantic_study_search.api.settings import APISettings
from semantic_study_search.domain import SearchResult
from semantic_study_search.ingestion.chunking import token_count
from semantic_study_search.search.encoder import MAX_TOKENS, QUERY_PREFIX, E5Encoder
from semantic_study_search.search.engine import SearchEngine
from semantic_study_search.search.relevance import default_min_score, validate_min_score
from semantic_study_search.storage.index import IndexConfig, load_index


class SearchBusyError(RuntimeError):
    """Сообщает, что модель уже обрабатывает другой запрос."""


class QueryLimitError(ValueError):
    """Сообщает о превышении числа токенов в вопросе."""


class SearchService:
    """Обслуживает запросы к загруженной библиотеке с ограничением параллельных вычислений."""

    def __init__(
        self, engine: SearchEngine, document_count: int, *, min_score: float | None = None
    ):
        """Сохраняет поисковый движок и сведения о библиотеке.

        Args:
            engine: Поиск по уже загруженным векторам.
            document_count: Количество документов в библиотеке.
            min_score: Минимальное сходство; None использует калиброванный порог.
        """
        self.engine = engine
        self.document_count = document_count
        self.fragment_count = len(engine.chunks)
        self.min_score = default_min_score() if min_score is None else validate_min_score(min_score)
        self._lock = Lock()

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Выполняет один поисковый запрос без одновременного запуска модели.

        Args:
            query: Проверенный вопрос пользователя.
            top_k: Максимальное количество результатов.

        Returns:
            Фрагменты по убыванию сходства.
        """
        if not self._lock.acquire(blocking=False):
            raise SearchBusyError("Сервис занят другим запросом. Попробуйте ещё раз.")
        try:
            if token_count(self.engine.encoder.tokenizer, QUERY_PREFIX + query) > MAX_TOKENS:
                raise QueryLimitError("Вопрос слишком длинный для модели. Сократите формулировку.")
            return self.engine.search(query, top_k=top_k, min_score=self.min_score)
        finally:
            self._lock.release()


def load_service(settings: APISettings) -> SearchService:
    """Загружает совместимый индекс и модель для работы API.

    Args:
        settings: Настройки индекса и кэша модели.

    Returns:
        Готовый поисковый сервис.
    """
    index = load_index(settings.index_dir)
    expected = IndexConfig()
    if (index.config.model_id, index.config.model_revision, index.config.dimension) != (
        expected.model_id,
        expected.model_revision,
        expected.dimension,
    ):
        raise ValueError("Индекс создан другой моделью; выполните semantic-study-index.")
    encoder = E5Encoder(cache_folder=settings.model_cache, offline=settings.offline)
    engine = SearchEngine.from_index(index, encoder)
    return SearchService(engine, len(index.documents), min_score=settings.min_score)
