"""Поиск по учебным материалам из терминала."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from semantic_study_search.search.relevance import default_min_score, validate_min_score


def main(argv: list[str] | None = None) -> int:
    """Выполняет поисковый запрос из командной строки.

    Args:
        argv: Аргументы запуска; None использует аргументы командной строки.

    Returns:
        Код завершения: 0 при успехе, 1 при обработанной ошибке.
    """
    parser = argparse.ArgumentParser(description="Семантический поиск по учебным материалам.")
    parser.add_argument("query", help="Вопрос на русском языке")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Количество фрагментов (по умолчанию 5)"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--corpus", type=Path, help="Разовый поиск по JSON-корпусу без сохранения")
    source.add_argument("--index-dir", type=Path, default=Path("data/processed/library"))
    parser.add_argument("--cache-dir", type=Path, help="Каталог кэша модели Hugging Face")
    parser.add_argument("--offline", action="store_true", help="Загрузить модель только из кэша")
    parser.add_argument("--json", action="store_true", help="Вывести результаты в JSON")
    parser.add_argument(
        "--min-score",
        type=float,
        default=default_min_score(),
        help="Минимальное сходство; -1 отключает фильтрацию",
    )
    args = parser.parse_args(argv)
    try:
        from semantic_study_search.ingestion.chunking import chunk_documents
        from semantic_study_search.ingestion.corpus import load_corpus
        from semantic_study_search.search.encoder import E5Encoder, validate_query
        from semantic_study_search.search.engine import SearchEngine
        from semantic_study_search.storage.index import IndexConfig, load_index

        query = validate_query(args.query)
        if args.top_k < 1:
            raise ValueError("Количество результатов должно быть положительным целым числом.")
        min_score = validate_min_score(args.min_score)
        if args.corpus is not None:
            documents = load_corpus(args.corpus)
            index = None
        else:
            index = load_index(args.index_dir)
            expected = IndexConfig()
            if (index.config.model_id, index.config.model_revision, index.config.dimension) != (
                expected.model_id,
                expected.model_revision,
                expected.dimension,
            ):
                raise ValueError("Индекс создан другой моделью; выполните semantic-study-index.")
        print("Загрузка E5 на CPU…", file=sys.stderr)
        encoder = E5Encoder(cache_folder=args.cache_dir, offline=args.offline)
        if index is None:
            chunks = chunk_documents(documents, encoder.tokenizer)
            print(
                f"Индексация: {len(documents)} документов, {len(chunks)} фрагментов…",
                file=sys.stderr,
            )
            engine = SearchEngine(chunks, encoder)
        else:
            engine = SearchEngine.from_index(index, encoder)
            print(f"Загружен индекс: {len(index.chunks)} фрагментов.", file=sys.stderr)
        results = engine.search(query, top_k=args.top_k, min_score=min_score)
    except ImportError:
        print("Установите зависимости: uv sync --locked --all-extras", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        if not results:
            print("Достаточно близкие фрагменты не найдены. Попробуйте уточнить вопрос.")
        for number, result in enumerate(results, 1):
            print(f"\n{number}. {result.chunk.title} (сходство: {result.score:.4f})")
            print(result.chunk.source_url)
            print(result.chunk.text)
        print("\nСходство — косинусная мера, а не вероятность правильного ответа.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
