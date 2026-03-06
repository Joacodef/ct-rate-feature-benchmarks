"""Unit tests for scripts/generate_kfold_budget_manifests.py."""

from importlib import util
from pathlib import Path
import sys

import pandas as pd
import pytest


def _load_module(module_name: str, script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / script_name
    spec = util.spec_from_file_location(module_name, script_path)
    module = util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    # Dataclass decoration expects the importing module to be present in sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_multilabel_grouped_kfold_requires_enough_groups() -> None:
    mod = _load_module(
        "generate_kfold_budget_manifests_group_guard",
        "generate_kfold_budget_manifests.py",
    )

    grouped = pd.DataFrame({"label_a": [1]}, index=pd.Index(["g1"], name="group_key"))

    with pytest.raises(ValueError, match="Cannot create 2 folds"):
        mod._multilabel_grouped_kfold(grouped, k_folds=2, seed=52)


def test_main_generates_fold_budget_manifests_and_index(tmp_path, monkeypatch) -> None:
    mod = _load_module(
        "generate_kfold_budget_manifests_main",
        "generate_kfold_budget_manifests.py",
    )

    full_manifest = tmp_path / "all.csv"
    frame = pd.DataFrame(
        [
            {"volumename": "case_1_a", "visual_feature_path": "image/1.pt", "label_a": 1, "label_b": 0},
            {"volumename": "case_2_a", "visual_feature_path": "image/2.pt", "label_a": 0, "label_b": 1},
            {"volumename": "case_3_a", "visual_feature_path": "image/3.pt", "label_a": 1, "label_b": 1},
            {"volumename": "case_4_a", "visual_feature_path": "image/4.pt", "label_a": 0, "label_b": 0},
            {"volumename": "case_5_a", "visual_feature_path": "image/5.pt", "label_a": 1, "label_b": 0},
            {"volumename": "case_6_a", "visual_feature_path": "image/6.pt", "label_a": 0, "label_b": 1},
            {"volumename": "case_7_a", "visual_feature_path": "image/7.pt", "label_a": 1, "label_b": 1},
            {"volumename": "case_8_a", "visual_feature_path": "image/8.pt", "label_a": 0, "label_b": 0},
        ]
    )
    frame.to_csv(full_manifest, index=False)

    # Deterministic fold assignment for a stable test.
    def fake_kfold(_grouped_labels, k_folds, seed):
        assert k_folds == 2
        assert seed == 52
        return [
            (["case_1", "case_2", "case_3", "case_4"], ["case_5", "case_6", "case_7", "case_8"]),
            (["case_5", "case_6", "case_7", "case_8"], ["case_1", "case_2", "case_3", "case_4"]),
        ]

    monkeypatch.setattr(mod, "_multilabel_grouped_kfold", fake_kfold)

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_kfold_budget_manifests.py",
            "--full-manifest-path",
            str(full_manifest),
            "--target-labels",
            "label_a,label_b",
            "--k-folds",
            "2",
            "--budgets",
            "2,10",
            "--seed",
            "52",
            "--val-fraction",
            "0.25",
            "--output-subdir",
            "manual_kfold_budget_splits",
            "--prefix",
            "manual_kfold",
        ],
    )

    mod.main()

    out_dir = tmp_path / "manual_kfold_budget_splits"
    assert out_dir.exists()

    expected_files = [
        "manual_kfold_f1_test.csv",
        "manual_kfold_f2_test.csv",
        "manual_kfold_f1_n2_s52.csv",
        "manual_kfold_f1_n2_s52_val.csv",
        "manual_kfold_f1_n10_s52.csv",
        "manual_kfold_f1_n10_s52_val.csv",
        "manual_kfold_f2_n2_s52.csv",
        "manual_kfold_f2_n2_s52_val.csv",
        "manual_kfold_f2_n10_s52.csv",
        "manual_kfold_f2_n10_s52_val.csv",
        "manifest_index.csv",
    ]

    for filename in expected_files:
        assert (out_dir / filename).exists(), filename

    index_df = pd.read_csv(out_dir / "manifest_index.csv")
    assert len(index_df) == 4
    assert sorted(index_df["fold"].unique().tolist()) == [1, 2]
    assert sorted(index_df["requested_budget"].unique().tolist()) == [2, 10]

    # Budget 10 should be capped to fold train-pool size (4 rows).
    capped_rows = index_df[index_df["requested_budget"] == 10]
    assert (capped_rows["effective_budget"] == 4).all()

    # Every row should point to existing manifests.
    for _, row in index_df.iterrows():
        assert (out_dir / row["train_manifest"]).exists()
        assert (out_dir / row["val_manifest"]).exists()
        assert (out_dir / row["test_manifest"]).exists()
