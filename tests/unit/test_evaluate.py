import os
import torch
import types
from omegaconf import OmegaConf, DictConfig

import pytest

from ct_rate_benchmarks import evaluate


class SimpleModel(torch.nn.Module):
    def __init__(self, out_features: int = 1):
        super().__init__()
        # fixed input dim of 10 for the test
        self.linear = torch.nn.Linear(10, out_features)

    def forward(self, x):
        return self.linear(x)


class DummyDataset:
    def __init__(self, *args, **kwargs):
        # provide one sample
        self.sample = (torch.zeros(10), torch.tensor([0.0]))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample


def test_evaluate_model_returns_metrics(tmp_path, monkeypatch):
    # Create a checkpoint from a SimpleModel and save
    ckpt_path = tmp_path / "model.pt"
    state = SimpleModel(out_features=2).state_dict()
    torch.save(state, ckpt_path)

    # Monkeypatch hydra.utils.instantiate to return model or loss depending on args
    def fake_instantiate(*args, **kwargs):
        # If instantiate is asked to create a model it will receive out_features kwarg
        if "out_features" in kwargs:
            out_features = int(kwargs.get("out_features", 1))
            return SimpleModel(out_features=out_features)
        # Called for loss/criterion
        return torch.nn.BCEWithLogitsLoss()

    monkeypatch.setattr(evaluate.hydra.utils, "instantiate", fake_instantiate)

    # Monkeypatch FeatureDataset to the DummyDataset
    monkeypatch.setattr(evaluate, "FeatureDataset", DummyDataset)

    # Monkeypatch set_seed to no-op
    monkeypatch.setattr(evaluate, "set_seed", lambda *_: None)

    # Monkeypatch evaluate_epoch to return deterministic metrics
    def fake_evaluate_epoch(model, dataloader, criterion, device):
        return 0.1234, {"auroc": 0.7777}

    monkeypatch.setattr(evaluate, "evaluate_epoch", fake_evaluate_epoch)

    # Build a minimal config used by evaluate_model
    cfg = OmegaConf.create({
        "utils": {"seed": 42},
        "training": {
            "target_labels": ["a", "b"],
            "batch_size": 1,
            "num_workers": 0,
            "loss": {"_target_": "torch.nn.BCEWithLogitsLoss"},
        },
        "data": {
            "columns": {"visual_feature": "visual", "text_feature": "text"},
            "test_manifests": ["test_manifest.csv"],
        },
        "paths": {
            "data_root": "./data",
            "manifest_dir": str(tmp_path),
            # explicit checkpoint path -> should be resolved and used
            "checkpoint_path": str(ckpt_path),
        },
        "model": {"params": {}, "_target_": "ct_rate_benchmarks.tests.dummy"},
    })

    # Call evaluate_model and assert expected metric is returned
    results = evaluate.evaluate_model(DictConfig(cfg))

    assert isinstance(results, dict)
    metric_key = f"test_test_manifest.csv_auroc"
    assert metric_key in results
    assert abs(results[metric_key] - 0.7777) < 1e-6
