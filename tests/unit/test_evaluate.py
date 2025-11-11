import json
import os
import torch
from omegaconf import OmegaConf, DictConfig

from classification import evaluate


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
        "model": {"params": {}, "_target_": "classification.tests.dummy"},
    })

    # Call evaluate_model and assert expected metric is returned
    results = evaluate.evaluate_model(DictConfig(cfg))

    assert isinstance(results, dict)
    metric_key = f"test_test_manifest.csv_auroc"
    assert metric_key in results
    assert abs(results[metric_key] - 0.7777) < 1e-6


def test_evaluate_model_writes_detailed_metrics(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "model.pt"
    model_state = SimpleModel(out_features=1).state_dict()
    torch.save(model_state, ckpt_path)

    def fake_instantiate(*args, **kwargs):
        if "out_features" in kwargs:
            return SimpleModel(out_features=int(kwargs["out_features"]))
        return torch.nn.BCEWithLogitsLoss()

    monkeypatch.setattr(evaluate.hydra.utils, "instantiate", fake_instantiate)
    monkeypatch.setattr(evaluate, "FeatureDataset", DummyDataset)
    monkeypatch.setattr(evaluate, "set_seed", lambda *_: None)

    captured_calls = {}

    def fake_evaluate_epoch(model, dataloader, criterion, device, labels, negative_class):
        captured_calls["labels"] = labels
        captured_calls["negative"] = negative_class
        return 0.4321, {
            "auroc": 0.8888,
            "per_class": {
                negative_class: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
                labels[0]: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1},
            },
        }

    monkeypatch.setattr(evaluate, "evaluate_epoch", fake_evaluate_epoch)

    detailed_root = tmp_path / "runs" / "eval_run"
    cfg = OmegaConf.create({
        "utils": {"seed": 0},
        "hydra": {"runtime": {"cwd": str(tmp_path)}},
        "training": {
            "target_labels": ["Lesion"],
            "batch_size": 1,
            "num_workers": 0,
            "loss": {"_target_": "torch.nn.BCEWithLogitsLoss"},
        },
        "data": {
            "columns": {"visual_feature": "visual", "text_feature": "text"},
            "test_manifests": ["eval_manifest.csv"],
        },
        "paths": {
            "data_root": str(tmp_path / "data"),
            "manifest_dir": str(tmp_path),
            "run_dir": str(detailed_root),
            "checkpoint_path": str(ckpt_path),
        },
        "evaluation": {"negative_class_name": "Healthy"},
        "model": {"params": {}, "_target_": "classification.tests.dummy"},
    })

    results = evaluate.evaluate_model(DictConfig(cfg))

    assert captured_calls == {"labels": ["Lesion"], "negative": "Healthy"}
    metric_key = "test_eval_manifest.csv_auroc"
    assert metric_key in results and abs(results[metric_key] - 0.8888) < 1e-6

    metrics_dir = detailed_root / "detailed_metrics"
    expected_report = metrics_dir / "eval_manifest.csv_detailed_metrics.json"
    assert expected_report.exists()
    payload = json.loads(expected_report.read_text())
    assert payload["manifest"] == "eval_manifest.csv"
    assert "per_class" in payload and len(payload["per_class"]) == 2


def test_resolve_detailed_metrics_dir_prefers_run_dir(tmp_path):
    cfg = DictConfig(OmegaConf.create({
        "hydra": {"runtime": {"cwd": str(tmp_path)}},
        "paths": {"run_dir": "outputs/run"},
    }))

    resolved = evaluate._resolve_detailed_metrics_dir(cfg)
    expected = os.path.normpath(os.path.join(str(tmp_path), "outputs", "run", "detailed_metrics"))
    assert os.path.normpath(resolved) == expected


def test_resolve_detailed_metrics_dir_defaults_to_cwd(tmp_path):
    cfg = DictConfig(OmegaConf.create({
        "hydra": {"runtime": {"cwd": str(tmp_path)}},
    }))

    resolved = evaluate._resolve_detailed_metrics_dir(cfg)
    expected = os.path.normpath(os.path.join(str(tmp_path), "detailed_metrics"))
    assert os.path.normpath(resolved) == expected
