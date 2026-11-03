"""Проверки метрик, разделения данных и воспроизводимости порога."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from semantic_study_search.evaluation import (
    load_benchmark,
    load_policy,
    select_threshold,
    summarize,
)
from semantic_study_search.search.relevance import default_min_score
from semantic_study_search.storage.index import corpus_fingerprint, file_digest

ROOT = Path(__file__).parents[2]


@pytest.fixture
def measured_rows():
    """Создаёт выдачу с известными рангами и вопросами без ответа.

    Returns:
        Четыре вопроса с заданными оценками сходства.
    """
    return [
        {
            "id": "a",
            "query": "первый",
            "category": "answerable",
            "relevant_document_ids": ["right"],
            "retrieved": [
                {"document_id": "wrong", "score": 0.9},
                {"document_id": "right", "score": 0.8},
            ],
        },
        {
            "id": "b",
            "query": "второй",
            "category": "answerable",
            "relevant_document_ids": ["right"],
            "retrieved": [{"document_id": "right", "score": 0.75}],
        },
        {
            "id": "c",
            "query": "посторонний",
            "category": "out_of_scope",
            "relevant_document_ids": [],
            "retrieved": [{"document_id": "wrong", "score": 0.7}],
        },
        {
            "id": "d",
            "query": "близкий",
            "category": "related_unanswerable",
            "relevant_document_ids": [],
            "retrieved": [{"document_id": "wrong", "score": 0.85}],
        },
    ]


def test_metrics_count_rejected_positives_and_false_accepts(measured_rows):
    """Пустые ответы снижают Hit@k, а нерелевантная выдача считается ложным допуском.

    Args:
        measured_rows: Выдача с известными оценками и рангами.
    """
    baseline = summarize(measured_rows)
    assert baseline["hit_at_1"] == 0.5
    assert baseline["hit_at_5"] == 1
    assert baseline["mrr_at_5"] == 0.75
    filtered = summarize(measured_rows, 0.8)
    assert filtered["hit_at_1"] == 0
    assert filtered["hit_at_5"] == 0.5
    assert filtered["mrr_at_5"] == 0.25
    assert filtered["confusion"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    assert filtered["balanced_accuracy"] == 0.5
    assert {row["id"] for row in filtered["failures"]} == {"b", "d"}


def test_selection_preserves_calibration_hits(measured_rows):
    """Подбор порога сохраняет Hit@5 и отсеивает доступные отрицательные примеры.

    Args:
        measured_rows: Выдача с известными оценками.
    """
    threshold, trials = select_threshold(measured_rows)
    assert 0.7 < threshold <= 0.75
    assert summarize(measured_rows, threshold)["hit_at_5"] == 1
    assert summarize(measured_rows, threshold)["unanswerable_rejection_rate"] == 0.5
    assert all(trial["hit_at_5"] == 1 for trial in trials)


@pytest.fixture
def benchmark_payload(sample_index):
    """Создаёт корректную разметку для тестового индекса.

    Args:
        sample_index: Индекс трёх учебных документов.

    Returns:
        Калибровочный набор с двумя вопросами.
    """
    return {
        "schema_version": 1,
        "split": "calibration",
        "corpus_sha256": corpus_fingerprint(sample_index.documents),
        "index_config": asdict(sample_index.config),
        "cases": [
            {
                "id": "a",
                "query": "вопрос",
                "category": "answerable",
                "relevant_document_ids": ["0"],
                "rationale": "Есть в первом документе.",
            },
            {
                "id": "b",
                "query": "другой",
                "category": "out_of_scope",
                "relevant_document_ids": [],
                "rationale": "Материал отсутствует.",
            },
        ],
    }


@pytest.mark.parametrize(
    "change",
    [
        "split",
        "corpus",
        "config",
        "duplicate",
        "unknown_doc",
        "category",
        "only_positive",
        "empty_query",
    ],
)
def test_invalid_benchmark_is_rejected(tmp_path, sample_index, benchmark_payload, change):
    """Разметка другого корпуса, другой части или с ошибками отклоняется.

    Args:
        tmp_path: Временный каталог.
        sample_index: Проверяемый индекс.
        benchmark_payload: Корректная разметка.
        change: Нарушение контракта контрольного набора.
    """
    if change == "split":
        benchmark_payload["split"] = "test"
    elif change == "corpus":
        benchmark_payload["corpus_sha256"] = "other"
    elif change == "config":
        benchmark_payload["index_config"]["overlap"] = 0
    elif change == "duplicate":
        benchmark_payload["cases"].append(benchmark_payload["cases"][0])
    elif change == "unknown_doc":
        benchmark_payload["cases"][0]["relevant_document_ids"] = ["absent"]
    elif change == "category":
        benchmark_payload["cases"][0]["category"] = "out_of_scope"
    elif change == "only_positive":
        benchmark_payload["cases"].pop()
    else:
        benchmark_payload["cases"][0]["query"] = " "
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(benchmark_payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark(path, sample_index, "calibration")


def test_policy_rejects_another_corpus_and_test_tuning(tmp_path, sample_index):
    """Проверка не принимает порог другой библиотеки или тестовой части.

    Args:
        tmp_path: Временный каталог.
        sample_index: Тестовый индекс.
    """
    policy = {
        "schema_version": 1,
        "min_score": 0.8,
        "calibration_split": "calibration",
        "corpus_sha256": corpus_fingerprint(sample_index.documents),
        "index_config": asdict(sample_index.config),
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert load_policy(path, sample_index)["min_score"] == 0.8
    for changed in (policy | {"calibration_split": "test"}, policy | {"corpus_sha256": "other"}):
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            load_policy(path, sample_index)


def test_frozen_splits_are_disjoint_and_policy_matches_calibration():
    """Контрольные вопросы разделены, а порог связан с калибровочным файлом."""
    folder = ROOT / "tests/fixtures"
    calibration_path = folder / "relevance_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    test = json.loads((folder / "relevance_test.json").read_text(encoding="utf-8"))
    old = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    sets = [
        {row["query"].strip().casefold() for row in cases}
        for cases in (calibration["cases"], test["cases"], old)
    ]
    assert len(sets[0]) == len(sets[1]) == 40
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert "что такое гладиолус?" in sets[0]
    policy = json.loads(
        (ROOT / "src/semantic_study_search/resources/relevance_policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["calibration_sha256"] == file_digest(calibration_path)
    assert default_min_score() == policy["min_score"]
    assert policy["corpus_sha256"] == calibration["corpus_sha256"] == test["corpus_sha256"]
