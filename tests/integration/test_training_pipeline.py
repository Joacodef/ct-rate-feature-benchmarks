"""Integration tests that exercise the full training loop."""

from pathlib import Path

import torch

from classification.train import train_model


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


def test_training_can_resume_from_saved_state(mock_cfg) -> None:
    """Ensure last_state checkpoints enable a resumed training run."""

    # Run a short initial training session.
    mock_cfg.training.max_epochs = 2
    mock_cfg.training.early_stopping_patience = 10
    train_model(mock_cfg)

    checkpoint_dir = Path(mock_cfg.paths.checkpoint_dir)
    state_path = checkpoint_dir / "last_state.pt"
    assert state_path.exists()

    try:
        saved_state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        saved_state = torch.load(state_path, map_location="cpu")
    assert saved_state.get("epoch") == 2

    # Resume training for additional epochs using the saved state.
    mock_cfg.training.max_epochs = 4
    mock_cfg.training.resume.enabled = True
    mock_cfg.training.resume.state_path = str(state_path)

    resumed_metric = train_model(mock_cfg)
    assert isinstance(resumed_metric, float)

    try:
        resumed_state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        resumed_state = torch.load(state_path, map_location="cpu")
    assert resumed_state.get("epoch") >= 4
    assert "optimizer" in resumed_state
    assert "best_val_auprc" in resumed_state
    assert "best_val_auroc" in resumed_state
    assert "best_primary_metric" in resumed_state
