"""Integration-style test for the train_model helper."""

from pathlib import Path

import pytest
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