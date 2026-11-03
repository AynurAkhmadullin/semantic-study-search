"""Проверки отдельных запусков калибровки и контрольной оценки."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest

from semantic_study_search.storage.index import corpus_fingerprint, save_index

pytestmark = pytest.mark.integration


def test_evaluation_cli_separates_calibration_and_test(
    tmp_path, sample_index, encoder, monkeypatch
):
    """CLI сохраняет порог при калибровке и только читает его при проверке.

    Args:
        tmp_path: Временный каталог.
        sample_index: Тестовый индекс.
        encoder: Кодировщик с фиксированными векторами.
        monkeypatch: Фикстура подмены модели.
    """
    spec = importlib.util.spec_from_file_location(
        "evaluation_script", Path(__file__).parents[2] / "scripts/evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = Mock(return_value=encoder)
    monkeypatch.setattr(module, "E5Encoder", model)
    index_path = tmp_path / "index"
    save_index(index_path, sample_index)
    data = {
        "schema_version": 1,
        "split": "calibration",
        "corpus_sha256": corpus_fingerprint(sample_index.documents),
        "index_config": asdict(sample_index.config),
        "cases": [
            {
                "id": "yes",
                "query": "учебный вопрос",
                "category": "answerable",
                "relevant_document_ids": ["0"],
                "rationale": "Есть в источнике.",
            },
            {
                "id": "no",
                "query": "посторонний вопрос",
                "category": "out_of_scope",
                "relevant_document_ids": [],
                "rationale": "Нет в библиотеке.",
            },
        ],
    }
    cases, policy, output = (
        tmp_path / name for name in ("cases.json", "policy.json", "report.json")
    )
    cases.write_text(json.dumps(data), encoding="utf-8")
    common = [
        "--index-dir",
        str(index_path),
        "--cases",
        str(cases),
        "--repeats",
        "1",
        "--output",
        str(output),
    ]
    assert module.main([*common, "--split", "calibration", "--policy-output", str(policy)]) == 0
    before = policy.read_bytes()
    data["split"] = "test"
    cases.write_text(json.dumps(data), encoding="utf-8")
    assert module.main([*common, "--policy", str(policy)]) == 0
    assert policy.read_bytes() == before
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["split"] == "test"
    assert report["calibration_trials"] is None
    assert len(report["queries"]) == 2
    assert report["baseline"]["hit_at_5"] == 1
    model.reset_mock()
    assert module.main([*common, "--split", "calibration"]) == 1
    model.assert_not_called()
