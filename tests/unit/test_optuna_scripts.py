"""Unit tests for Optuna helper scripts under scripts/."""

from importlib import util
from pathlib import Path


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