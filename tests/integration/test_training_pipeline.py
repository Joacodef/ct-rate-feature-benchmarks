"""Integration tests that exercise the full training loop."""

from pathlib import Path

import torch

from ct_rate_benchmarks.train import train_model


def test_training_pipeline_produces_checkpoint(mock_cfg) -> None:
    """Runs the training pipeline end-to-end and inspects outputs."""
    mock_cfg.utils.seed = 7

    metric = train_model(mock_cfg)

    assert 0.0 <= metric <= 1.0

    checkpoint_dir = Path(mock_cfg.paths.checkpoint_dir)
    checkpoint_path = checkpoint_dir / "final_model.pt"

    assert checkpoint_dir.exists()
    assert checkpoint_path.exists()

    # Load checkpoint to ensure it is readable and non-empty
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    assert isinstance(state_dict, dict)
    assert state_dict  # Ensure the state dict contains parameters
