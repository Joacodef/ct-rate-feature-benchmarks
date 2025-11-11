"""Unit tests for classification.loops utilities."""

import torch
import pytest

from classification import loops


def test_compute_metrics_binary_with_label_names():
    logits = torch.tensor([[0.0], [3.0]])
    targets = torch.tensor([[0.0], [1.0]])

    metrics = loops.compute_metrics(
        logits,
        targets,
        label_names=["Lesion"],
        negative_class_name="Healthy",
    )

    assert "auroc" in metrics and 0.0 <= metrics["auroc"] <= 1.0
    per_class = metrics["per_class"]
    assert set(per_class.keys()) == {"Healthy", "Lesion"}
    assert all(set(stats.keys()) == {"precision", "recall", "f1", "support"} for stats in per_class.values())


def test_compute_metrics_handles_missing_labels():
    logits = torch.zeros((3, 1))
    targets = torch.zeros((3, 1))

    metrics = loops.compute_metrics(logits, targets)

    assert metrics["auroc"] == 0.0
    assert "per_class" not in metrics


def test_compute_metrics_handles_per_class_error(monkeypatch):
    logits = torch.tensor([[0.0], [2.0]])
    targets = torch.tensor([[0.0], [1.0]])

    def boom(*_args, **_kwargs):
        raise RuntimeError("precision failure")

    monkeypatch.setattr(loops, "precision_recall_fscore_support", boom)

    metrics = loops.compute_metrics(
        logits,
        targets,
        label_names=["Anomaly"],
        negative_class_name="Normal",
    )

    assert "per_class" not in metrics
