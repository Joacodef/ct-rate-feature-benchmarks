"""Unit tests for scripts/evaluate_and_aggregate_runs.py helpers."""

from importlib import util
from pathlib import Path

import pandas as pd


def _load_module(module_name: str, script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / script_name
    spec = util.spec_from_file_location(module_name, script_path)
    module = util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_budget_seed_parses_fold_budget_seed() -> None:
    mod = _load_module(
        "evaluate_and_aggregate_runs_parse_fold", "evaluate_and_aggregate_runs.py"
    )

    parsed = mod._extract_budget_seed("manual_kfold_f3_n250_s52.csv")

    assert parsed["cv_fold"] == 3
    assert parsed["budget_n"] == 250
    assert parsed["split_seed"] == 52


def test_add_primary_test_metrics_merges_fold_specific_columns() -> None:
    mod = _load_module(
        "evaluate_and_aggregate_runs_primary_metrics", "evaluate_and_aggregate_runs.py"
    )

    frame = pd.DataFrame(
        [
            {
                "test_manual_kfold_f1_test_auprc": 0.40,
                "test_manual_kfold_f2_test_auprc": None,
                "test_manual_kfold_f1_test_auroc": 0.70,
                "test_manual_kfold_f2_test_auroc": None,
                "test_manual_kfold_f1_test_f1_macro": 0.50,
                "test_manual_kfold_f2_test_f1_macro": None,
            },
            {
                "test_manual_kfold_f1_test_auprc": None,
                "test_manual_kfold_f2_test_auprc": 0.60,
                "test_manual_kfold_f1_test_auroc": None,
                "test_manual_kfold_f2_test_auroc": 0.80,
                "test_manual_kfold_f1_test_f1_macro": None,
                "test_manual_kfold_f2_test_f1_macro": 0.55,
            },
        ]
    )

    out = mod._add_primary_test_metrics(frame)

    assert list(out["test_primary_auprc"]) == [0.40, 0.60]
    assert list(out["test_primary_auroc"]) == [0.70, 0.80]
    assert list(out["test_primary_f1_macro"]) == [0.50, 0.55]
