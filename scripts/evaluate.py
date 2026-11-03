"""Калибровка фильтра и отдельная оценка поиска на контрольных вопросах."""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from semantic_study_search.evaluation import (
    load_benchmark,
    load_policy,
    score_cases,
    select_threshold,
    summarize,
)
from semantic_study_search.search.encoder import E5Encoder
from semantic_study_search.search.engine import SearchEngine
from semantic_study_search.storage.index import corpus_fingerprint, file_digest, load_index

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    """Сохраняет JSON-отчёт в UTF-8.

    Args:
        path: Файл результата.
        payload: Метаданные и измерения.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Выполняет калибровку или проверку заранее выбранного порога.

    Args:
        argv: Аргументы запуска; None читает командную строку.

    Returns:
        Код завершения: 0 при успехе, 1 при ошибке данных.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["calibration", "test"], default="test")
    parser.add_argument("--cases", type=Path, help="Контрольный набор для выбранной части")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/library"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "src/semantic_study_search/resources/relevance_policy.json",
    )
    parser.add_argument("--policy-output", type=Path, default=Path(".cache/relevance_policy.json"))
    parser.add_argument("--output", type=Path, default=Path(".cache/evaluation.json"))
    args = parser.parse_args(argv)
    try:
        if args.repeats < 1:
            raise ValueError("Число повторений должно быть положительным.")
        start = perf_counter()
        index = load_index(args.index_dir)
        index_seconds = perf_counter() - start
        case_path = args.cases or ROOT / f"tests/fixtures/relevance_{args.split}.json"
        benchmark = load_benchmark(case_path, index, args.split)
        policy = load_policy(args.policy, index) if args.split == "test" else None
        print("Загрузка E5 и оценка контрольных вопросов…", file=sys.stderr)
        start = perf_counter()
        encoder = E5Encoder(cache_folder=args.cache_dir, offline=args.offline)
        engine = SearchEngine.from_index(index, encoder)
        load_seconds = perf_counter() - start
        rows = score_cases(engine, benchmark["cases"], args.repeats)
        trials = None
        if args.split == "calibration":
            threshold, trials = select_threshold(rows)
            policy = {
                "schema_version": 1,
                "min_score": threshold,
                "calibration_split": "calibration",
                "calibration_sha256": file_digest(case_path),
                "calibration_questions": len(rows),
                "corpus_sha256": corpus_fingerprint(index.documents),
                "index_config": asdict(index.config),
                "selection": "Maximize balanced accuracy while preserving raw calibration Hit@5 and accepting >=95% of answerable questions; ties prefer the lower threshold.",
                "limitation": "A heuristic for this fixed library, not an answerability guarantee; recalibrate after corpus/model/chunking changes.",
            }
            write_json(args.policy_output, policy)
        samples = [seconds for row in rows for seconds in row["seconds"]]
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "split": args.split,
            "dataset_sha256": file_digest(case_path),
            "policy": policy,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    name: version(name)
                    for name in ("numpy", "torch", "transformers", "sentence-transformers")
                },
            },
            "documents": len(index.documents),
            "chunks": len(index.chunks),
            "timing": {
                "device": "cpu",
                "index_load_seconds": index_seconds,
                "model_load_seconds": load_seconds,
                "query_median_ms": float(np.median(samples) * 1000),
                "query_p95_ms": float(np.percentile(samples, 95) * 1000),
                "repeats": args.repeats,
                "warmup_queries": 1,
            },
            "definitions": "Hit@k and MRR@5 use answerable queries and document labels; empty results count as misses. Rejection metrics use questions without an answer. Timings exclude HTTP/UI and initial warmup.",
            "limitation": "Small author-created Russian dataset with frozen calibration/test splits, not an independent external benchmark; document presence does not prove the returned passage answers the question.",
            "baseline": summarize(rows),
            "filtered": summarize(rows, policy["min_score"]),
            "calibration_trials": trials,
            "queries": rows,
        }
        write_json(args.output, report)
        keys = ("hit_at_5", "unanswerable_rejection_rate")
        print(
            json.dumps(
                {
                    "report": str(args.output),
                    "split": args.split,
                    "min_score": policy["min_score"],
                    "baseline": {k: report["baseline"][k] for k in keys},
                    "filtered": {k: report["filtered"][k] for k in keys},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
