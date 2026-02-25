"""Unit tests for Optuna helper scripts under scripts/."""

from importlib import util
from pathlib import Path

import optuna


def _load_module(module_name: str, script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / script_name
    spec = util.spec_from_file_location(module_name, script_path)
    module = util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_optuna_search_reads_auprc_first(tmp_path):
    mod = _load_module("optuna_mlp_search_test", "optuna_mlp_search.py")
    outputs_root = tmp_path / "outputs"
    job_name = "trial_a"
    job_dir = outputs_root / job_name
    job_dir.mkdir(parents=True)
    (job_dir / "latest_run.json").write_text(
        '{"best_val_auprc": 0.42, "best_val_auroc": 0.99}'
    )

    value = mod._read_best_trial_value(outputs_root, job_name)
    assert value == 0.42


def test_optuna_search_falls_back_to_auroc(tmp_path):
    mod = _load_module("optuna_mlp_search_test_fallback", "optuna_mlp_search.py")
    outputs_root = tmp_path / "outputs"
    job_name = "trial_b"
    job_dir = outputs_root / job_name
    job_dir.mkdir(parents=True)
    (job_dir / "latest_run.json").write_text('{"best_val_auroc": 0.73}')

    value = mod._read_best_trial_value(outputs_root, job_name)
    assert value == 0.73


def test_optuna_search_study_outputs_root_appends_study_name(tmp_path):
    mod = _load_module("optuna_mlp_search_test_study_root", "optuna_mlp_search.py")
    outputs_root = tmp_path / "outputs"

    study_root = mod._study_outputs_root(outputs_root, "manual_labels")

    assert study_root == outputs_root / "manual_labels"


def test_optuna_search_study_outputs_root_sanitizes_study_name(tmp_path):
    mod = _load_module("optuna_mlp_search_test_study_root_sanitized", "optuna_mlp_search.py")
    outputs_root = tmp_path / "outputs"

    study_root = mod._study_outputs_root(outputs_root, "manual labels/exp:01")

    assert study_root == outputs_root / "manual_labels_exp_01"


def test_optuna_search_build_pruner_default_is_median():
    mod = _load_module("optuna_mlp_search_test_pruner_default", "optuna_mlp_search.py")

    pruner = mod._build_pruner(
        disable_pruning=False,
        startup_trials=12,
        warmup_steps=7,
        interval_steps=2,
    )

    assert isinstance(pruner, optuna.pruners.MedianPruner)


def test_optuna_search_build_pruner_disable_uses_nop():
    mod = _load_module("optuna_mlp_search_test_pruner_disabled", "optuna_mlp_search.py")

    pruner = mod._build_pruner(
        disable_pruning=True,
        startup_trials=12,
        warmup_steps=7,
        interval_steps=2,
    )

    assert isinstance(pruner, optuna.pruners.NopPruner)


def test_optuna_search_extract_report_metric_prefers_auprc():
    mod = _load_module("optuna_mlp_search_test_extract_metric", "optuna_mlp_search.py")

    value = mod._extract_report_metric(
        {
            "best_val_auprc": 0.44,
            "best_val_auroc": 0.80,
            "val_metrics": {"auprc": 0.39, "auroc": 0.78},
        }
    )

    assert value == 0.44


def test_optuna_search_extract_report_metric_falls_back_to_auroc():
    mod = _load_module("optuna_mlp_search_test_extract_metric_fallback", "optuna_mlp_search.py")

    value = mod._extract_report_metric(
        {
            "val_metrics": {"auroc": 0.71},
        }
    )

    assert value == 0.71


def test_optuna_search_resolve_pruning_settings_from_config(tmp_path):
    mod = _load_module("optuna_mlp_search_test_pruning_resolve_cfg", "optuna_mlp_search.py")
    cfg = tmp_path / "optuna_cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "optuna:",
                "  disable_pruning: true",
                "  pruner_startup_trials: 15",
                "  pruner_warmup_steps: 6",
                "  pruner_interval_steps: 2",
            ]
        )
    )

    resolved = mod._resolve_optuna_pruning_settings(
        cli_disable_pruning=None,
        cli_startup_trials=None,
        cli_warmup_steps=None,
        cli_interval_steps=None,
        config_name=str(cfg),
        overrides=[],
    )

    assert resolved == (True, 15, 6, 2)


def test_optuna_search_resolve_pruning_settings_cli_precedence(tmp_path):
    mod = _load_module("optuna_mlp_search_test_pruning_resolve_cli", "optuna_mlp_search.py")
    cfg = tmp_path / "optuna_cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "optuna:",
                "  disable_pruning: false",
                "  pruner_startup_trials: 10",
                "  pruner_warmup_steps: 5",
                "  pruner_interval_steps: 1",
            ]
        )
    )

    resolved = mod._resolve_optuna_pruning_settings(
        cli_disable_pruning=True,
        cli_startup_trials=30,
        cli_warmup_steps=9,
        cli_interval_steps=3,
        config_name=str(cfg),
        overrides=[
            "optuna.disable_pruning=false",
            "optuna.pruner_startup_trials=20",
            "optuna.pruner_warmup_steps=8",
            "optuna.pruner_interval_steps=2",
        ],
    )

    assert resolved == (True, 30, 9, 3)


def test_optuna_best_summary_prefers_auprc(capsys, tmp_path):
    mod = _load_module("optuna_best_summary_test", "optuna_best_summary.py")
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    t1 = study_dir / "optuna_mlp_t0001"
    t2 = study_dir / "optuna_mlp_t0002"
    t1.mkdir()
    t2.mkdir()

    (t1 / "latest_run.json").write_text(
        '{"best_val_auprc": 0.40, "best_val_auroc": 0.95, "best_val_f1_macro": 0.35}'
    )
    (t2 / "latest_run.json").write_text(
        '{"best_val_auprc": 0.50, "best_val_auroc": 0.70, "best_val_f1_macro": 0.45}'
    )

    mod.summarize(str(study_dir))
    out = capsys.readouterr().out

    assert "Best trial: optuna_mlp_t0002" in out
    assert "Best val AUPRC: 0.500000" in out
    assert "Best val AUROC: 0.700000" in out
    assert "Best val F1-macro: 0.450000" in out


def test_optuna_best_summary_fallback_to_auroc(capsys, tmp_path):
    mod = _load_module("optuna_best_summary_test_fallback", "optuna_best_summary.py")
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    t1 = study_dir / "optuna_mlp_t0001"
    t2 = study_dir / "optuna_mlp_t0002"
    t1.mkdir()
    t2.mkdir()

    (t1 / "latest_run.json").write_text('{"best_val_auroc": 0.60}')
    (t2 / "latest_run.json").write_text('{"best_val_auroc": 0.80}')

    mod.summarize(str(study_dir))
    out = capsys.readouterr().out

    assert "Best trial: optuna_mlp_t0002" in out
    assert "Best val AUROC: 0.800000" in out