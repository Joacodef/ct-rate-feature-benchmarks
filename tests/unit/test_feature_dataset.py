"""Unit tests for FeatureDataset."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from ct_rate_benchmarks.data.dataset import FeatureDataset


def test_missing_manifest_raises(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()

    with pytest.raises(FileNotFoundError):
        FeatureDataset(
            manifest_path=str(tmp_path / "missing.csv"),
            data_root=str(data_root),
            target_labels=["label"],
            visual_feature_col="visual",
        )


def test_missing_target_column_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"visual": ["features/sample.pt"]}).to_csv(manifest_path, index=False)

    with pytest.raises(ValueError):
        FeatureDataset(
            manifest_path=str(manifest_path),
            data_root=str(tmp_path),
            target_labels=["missing_label"],
            visual_feature_col="visual",
        )


def test_missing_feature_file_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    feature_rel_path = "features/sample.pt"
    pd.DataFrame({
        "visual": [feature_rel_path],
        "label": [1],
    }).to_csv(manifest_path, index=False)

    (tmp_path / "features").mkdir()

    dataset = FeatureDataset(
        manifest_path=str(manifest_path),
        data_root=str(tmp_path),
        target_labels=["label"],
        visual_feature_col="visual",
    )

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_returns_expected_item(tmp_path: Path) -> None:
    data_root = tmp_path / "root"
    feature_dir = data_root / "features"
    feature_dir.mkdir(parents=True)

    visual_tensor = torch.randn(4)
    text_tensor = torch.randn(3)
    torch.save(visual_tensor, feature_dir / "visual.pt")
    torch.save(text_tensor, feature_dir / "text.pt")

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "visual": ["features/visual.pt"],
        "text": ["features/text.pt"],
        "label_a": [0],
        "label_b": [1],
    }).to_csv(manifest_path, index=False)

    dataset = FeatureDataset(
        manifest_path=str(manifest_path),
        data_root=str(data_root),
        target_labels=["label_a", "label_b"],
        visual_feature_col="visual",
        text_feature_col="text",
    )

    item = dataset[0]

    assert torch.equal(item["visual_features"], visual_tensor)
    assert torch.equal(item["text_features"], text_tensor)
    assert item["labels"].tolist() == [0.0, 1.0]
    assert len(dataset) == 1
