"""Оценка поиска и калибровка отказа от слабых результатов."""

import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from time import perf_counter

from semantic_study_search.search.encoder import validate_query
from semantic_study_search.search.engine import SearchEngine
from semantic_study_search.search.relevance import validate_min_score
from semantic_study_search.storage.index import StoredIndex, corpus_fingerprint

CATEGORIES = {"answerable", "out_of_scope", "related_unanswerable"}


def load_benchmark(path: Path, index: StoredIndex, split: str) -> dict:
    """Загружает разметку для указанной библиотеки и части набора.

    Args:
        path: Файл с контрольными вопросами.
        index: Проверяемая библиотека.
        split: Ожидаемая часть: calibration или test.

    Returns:
        Проверенный набор вопросов с метаданными.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Неподдерживаемая схема контрольного набора.")
    if split not in {"calibration", "test"} or data.get("split") != split:
        raise ValueError("Калибровочная и тестовая части должны использоваться отдельно.")
    if data.get("corpus_sha256") != corpus_fingerprint(index.documents):
        raise ValueError("Разметка относится к другой библиотеке; проверьте состав материалов.")
    if data.get("index_config") != asdict(index.config):
        raise ValueError("Настройки модели или разбиения отличаются от контрольного набора.")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Контрольный набор должен содержать вопросы.")
    ids, queries = set(), set()
    documents = {doc.id for doc in index.documents}
    for case in cases:
        if not isinstance(case, dict) or not all(
            isinstance(case.get(key), str) and case[key].strip()
            for key in ("id", "query", "category", "rationale")
        ):
            raise ValueError("У вопроса отсутствуют обязательные текстовые поля.")
        query = validate_query(case["query"]).casefold()
        if case["id"] in ids or query in queries:
            raise ValueError("В контрольном наборе повторяется вопрос или идентификатор.")
        ids.add(case["id"])
        queries.add(query)
        relevant = case.get("relevant_document_ids")
        if (
            case["category"] not in CATEGORIES
            or not isinstance(relevant, list)
            or any(not isinstance(item, str) or item not in documents for item in relevant)
            or len(set(relevant)) != len(relevant)
            or bool(relevant) != (case["category"] == "answerable")
        ):
            raise ValueError("Категория вопроса и ожидаемые документы не согласованы.")
    if not any(case["category"] == "answerable" for case in cases) or not any(
        case["category"] != "answerable" for case in cases
    ):
        raise ValueError("Нужны вопросы с ответами и без ответов.")
    return data


def score_cases(engine: SearchEngine, cases: list[dict], repeats: int = 3) -> list[dict]:
    """Измеряет исходную выдачу и время запросов к готовой модели.

    Args:
        engine: Поиск по загруженному индексу.
        cases: Проверенные вопросы с ожидаемыми источниками.
        repeats: Число измерений на вопрос.

    Returns:
        Вопросы с первой пятёркой результатов и замерами времени.
    """
    if type(repeats) is not int or repeats < 1 or not cases:
        raise ValueError("Нужны вопросы и положительное число повторений.")
    engine.search(cases[0]["query"], top_k=5)
    rows = []
    for case in cases:
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            results = engine.search(case["query"], top_k=5)
            samples.append(perf_counter() - start)
        rows.append(
            case
            | {
                "seconds": samples,
                "retrieved": [
                    {"chunk_id": r.chunk.id, "document_id": r.chunk.document_id, "score": r.score}
                    for r in results
                ],
            }
        )
    return rows


def summarize(rows: list[dict], min_score: float | None = None) -> dict:
    """Вычисляет качество выдачи и отказов с учётом пустых результатов.

    Args:
        rows: Измеренные результаты контрольных вопросов.
        min_score: Порог фрагментов; None оставляет исходную выдачу.

    Returns:
        Метрики, числа ошибок и список неудачных запросов.
    """
    if min_score is not None:
        min_score = validate_min_score(min_score)
    ranks, failures = [], []
    tp = fn = fp = tn = 0
    groups = {category: {"total": 0, "rejected": 0} for category in CATEGORIES}
    for row in rows:
        results = [r for r in row["retrieved"] if min_score is None or r["score"] >= min_score]
        answerable, accepted = row["category"] == "answerable", bool(results)
        tp += answerable and accepted
        fn += answerable and not accepted
        fp += not answerable and accepted
        tn += not answerable and not accepted
        group = groups[row["category"]]
        group["total"] += 1
        group["rejected"] += not accepted
        rank = next(
            (
                i
                for i, r in enumerate(results, 1)
                if r["document_id"] in row["relevant_document_ids"]
            ),
            None,
        )
        if answerable:
            ranks.append(rank)
        if (answerable and rank is None) or (not answerable and accepted):
            failures.append(
                {
                    "id": row["id"],
                    "query": row["query"],
                    "category": row["category"],
                    "reason": "expected_source_missing" if answerable else "false_accept",
                    "top_score": row["retrieved"][0]["score"] if row["retrieved"] else None,
                }
            )
    if not ranks or not fp + tn:
        raise ValueError("Для оценки нужны вопросы с ответами и без ответов.")
    return {
        "questions": len(rows),
        "answerable": len(ranks),
        "unanswerable": fp + tn,
        **{
            f"hit_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / len(ranks)
            for k in (1, 3, 5)
        },
        "mrr_at_5": sum(1 / rank for rank in ranks if rank is not None and rank <= 5) / len(ranks),
        "answerable_accept_rate": tp / (tp + fn),
        "unanswerable_rejection_rate": tn / (tn + fp),
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "by_category": dict(sorted(groups.items())),
        "failures": failures,
    }


def select_threshold(rows: list[dict]) -> tuple[float, list[dict]]:
    """Подбирает порог без снижения исходного Hit@5 на калибровочных вопросах.

    Args:
        rows: Выдача только калибровочной части.

    Returns:
        Порог и сравнение допустимых кандидатов.
    """
    baseline = summarize(rows)
    scores = sorted({-1.0, 1.0, *(r["score"] for row in rows for r in row["retrieved"])})
    candidates = sorted({-1.0, 1.0, *((a + b) / 2 for a, b in pairwise(scores))})
    trials = []
    for threshold in candidates:
        metrics = summarize(rows, threshold)
        if metrics["hit_at_5"] < baseline["hit_at_5"] or metrics["answerable_accept_rate"] < 0.95:
            continue
        trials.append(
            {
                "min_score": threshold,
                **{
                    key: metrics[key]
                    for key in ("hit_at_5", "balanced_accuracy", "unanswerable_rejection_rate")
                },
            }
        )
    best = max(trials, key=lambda item: (item["balanced_accuracy"], -item["min_score"]))
    return best["min_score"], trials


def load_policy(path: Path, index: StoredIndex) -> dict:
    """Проверяет происхождение порога и соответствие оцениваемой библиотеке.

    Args:
        path: JSON-файл калиброванного порога.
        index: Библиотека для контрольного прогона.

    Returns:
        Проверенная конфигурация фильтра.
    """
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("Неподдерживаемая схема порога.")
    if policy.get("calibration_split") != "calibration":
        raise ValueError("Порог должен быть выбран по калибровочной части.")
    if policy.get("corpus_sha256") != corpus_fingerprint(index.documents) or policy.get(
        "index_config"
    ) != asdict(index.config):
        raise ValueError("Порог откалиброван для другой библиотеки или настроек.")
    validate_min_score(policy.get("min_score"))
    return policy
