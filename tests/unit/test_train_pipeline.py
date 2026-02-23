"""Integration-style test for the train_model helper."""

import json
from pathlib import Path

import pytest
from common.train.resume import resolve_path
from omegaconf import DictConfig

from classification.train import train_model


def test_train_model_runs(mock_cfg: DictConfig) -> None:
    """
    An integration test to verify that the entire train_model
    pipeline runs to completion without exceptions.
    """
    
    # Set a fixed seed for the test
    mock_cfg.utils.seed = 42
    
    # Run the main training function
    result_metric = train_model(mock_cfg)

    # --- Assertions ---
    # 1. Check that the function returned a float (the metric)
    assert isinstance(result_metric, float)

    # 2. Check that artifacts were created
    output_dir = Path(mock_cfg.paths.output_dir)
    checkpoint_dir = Path(mock_cfg.paths.checkpoint_dir)

    assert output_dir.exists()
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "final_model.pt").exists()
    assert (checkpoint_dir / "last_state.pt").exists()


def test_train_model_writes_latest_run_primary_metric(mock_cfg: DictConfig) -> None:
    result_metric = train_model(mock_cfg)

    pointer_path = resolve_path(mock_cfg, mock_cfg.paths.latest_run_pointer)
    assert pointer_path.exists()

    payload = json.loads(pointer_path.read_text())
    assert payload["primary_metric"] == "auprc"
    assert "best_val_auprc" in payload
    assert "best_val_auroc" in payload
    assert "best_val_f1_macro" in payload
    assert "best_val_metrics" in payload
    assert payload["best_val_metrics"]["auprc"] == pytest.approx(payload["best_val_auprc"])
    assert payload["best_val_metrics"]["auroc"] == pytest.approx(payload["best_val_auroc"])
    assert payload["best_val_metrics"]["f1_macro"] == pytest.approx(payload["best_val_f1_macro"])
    assert isinstance(result_metric, float)