"""Подготовка сохранённой библиотеки из терминала."""

import argparse
import sqlite3
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Создаёт или обновляет индекс учебной библиотеки.

    Args:
        argv: Аргументы запуска; None использует командную строку.

    Returns:
        Код завершения: 0 при успехе, 1 при обработанной ошибке.
    """
    parser = argparse.ArgumentParser(description="Подготовка индекса учебной библиотеки.")
    parser.add_argument(
        "--corpus", type=Path, action="append", default=[], help="Готовый JSON-корпус"
    )
    parser.add_argument(
        "--manifest", type=Path, action="append", default=[], help="Список TXT, MD, PDF"
    )
    parser.add_argument("--include-demo", action="store_true", help="Добавить встроенные материалы")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/library"))
    parser.add_argument("--chunk-tokens", type=int, default=320, help="Лимит токенов фрагмента")
    parser.add_argument("--overlap", type=int, default=48, help="Перекрытие в токенах")
    parser.add_argument("--cache-dir", type=Path, help="Каталог кэша модели")
    parser.add_argument("--offline", action="store_true", help="Загрузить модель только из кэша")
    parser.add_argument("--force", action="store_true", help="Принудительно пересоздать индекс")
    args = parser.parse_args(argv)
    try:
        from semantic_study_search.ingestion.corpus import load_corpus
        from semantic_study_search.ingestion.files import load_manifest
        from semantic_study_search.ingestion.pipeline import prepare_index
        from semantic_study_search.search.encoder import E5Encoder
        from semantic_study_search.storage.index import IndexConfig

        config = IndexConfig(max_tokens=args.chunk_tokens, overlap=args.overlap)
        config.validate()
        documents = load_corpus() if args.include_demo or not (args.corpus or args.manifest) else []
        for path in args.corpus:
            documents.extend(load_corpus(path))
        for path in args.manifest:
            documents.extend(load_manifest(path))

        def create_encoder() -> E5Encoder:
            """Загружает E5 для пересчёта векторов библиотеки.

            Returns:
                Модель E5 на CPU.
            """
            print("Загрузка E5 и вычисление векторов документов…", file=sys.stderr)
            return E5Encoder(cache_folder=args.cache_dir, offline=args.offline)

        index, created = prepare_index(
            args.index_dir, documents, create_encoder, config=config, force=args.force
        )
    except ImportError:
        print("Установите зависимости: uv sync --locked --all-extras", file=sys.stderr)
        return 1
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    status = "Индекс сохранён" if created else "Индекс актуален; пересчёт не требуется"
    print(f"{status}: {len(index.documents)} документов, {len(index.chunks)} фрагментов.")
    print(args.index_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
