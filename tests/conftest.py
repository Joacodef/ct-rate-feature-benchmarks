"""Shared pytest fixtures for the test suite."""

from pathlib import Path

import hydra
import pandas as pd
import pytest
import torch
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

MOCK_IN_FEATURES = 16  # Must match model.in_features override
MOCK_NUM_LABELS = 5  # Must match len(training.target_labels)
MOCK_NUM_SAMPLES = 10
MOCK_BATCH_SIZE = 2
MOCK_EPOCHS = 1


@pytest.fixture(scope="function")
def mock_data_path(tmp_path: Path) -> Path:
    """Creates a temporary directory structure with mock features and manifests."""
    data_root = tmp_path / "data_root"
    manifest_dir = tmp_path / "manifests"
    (data_root / "image").mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    torch.manual_seed(1234)

    sample_names = []
    for i in range(MOCK_NUM_SAMPLES):
        sample_name = f"sample_{i + 1}"
        sample_names.append(sample_name)

        mock_tensor = torch.randn(MOCK_IN_FEATURES)
        torch.save(mock_tensor, data_root / "image" / f"{sample_name}.pt")

    mock_labels = [f"label_{j}" for j in range(MOCK_NUM_LABELS)]

    df = pd.DataFrame({
        "volumename": sample_names,
        "visual_feature_path": [f"image/{name}.pt" for name in sample_names],
    })

    labels_df = pd.DataFrame(
        torch.randint(0, 2, (MOCK_NUM_SAMPLES, MOCK_NUM_LABELS)).numpy(),
        columns=mock_labels,
    )
    manifest_df = pd.concat([df, labels_df], axis=1)

    manifest_df.iloc[: MOCK_BATCH_SIZE * 2].to_csv(
        manifest_dir / "train.csv", index=False
    )
    manifest_df.iloc[MOCK_BATCH_SIZE * 2 : MOCK_BATCH_SIZE * 3].to_csv(
        manifest_dir / "valid.csv", index=False
    )
    manifest_df.iloc[MOCK_BATCH_SIZE * 3 :].to_csv(
        manifest_dir / "test1.csv", index=False
    )

    return tmp_path


@pytest.fixture(scope="function")
def mock_cfg(mock_data_path: Path) -> DictConfig:
    """Loads the Hydra configuration with overrides for fast runtime."""

    config_dir = Path(__file__).resolve().parent.parent / "configs"

    GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(config_dir=str(config_dir), job_name="test", version_base=None):
        cfg = hydra.compose(
            config_name="config.yaml",
            overrides=[
                "hydra/job_logging=default",
                "hydra/hydra_logging=default",
                # Provide concrete paths for keys that would otherwise use
                # Hydra resolvers (e.g. ${hydra:run.dir}). Setting these
                # `paths.*` values directly avoids the need for a running
                # HydraConfig during composition.
                f"paths.run_dir={(mock_data_path / 'outputs').as_posix()}",
                f"paths.job_base_dir={(mock_data_path / 'outputs' / 'job').as_posix()}",
                f"paths.data_root={(mock_data_path / 'data_root').as_posix()}",
                f"paths.manifest_dir={(mock_data_path / 'manifests').as_posix()}",
                f"data.train_manifest={ (mock_data_path / 'manifests' / 'train.csv').name}",
                f"data.val_manifest={ (mock_data_path / 'manifests' / 'valid.csv').name}",
                f"paths.output_dir={(mock_data_path / 'outputs').as_posix()}",
                f"paths.checkpoint_dir={(mock_data_path / 'outputs' / 'checkpoints').as_posix()}",
                "data.test_manifests=[test1.csv]",
                f"training.max_epochs={MOCK_EPOCHS}",
                f"training.batch_size={MOCK_BATCH_SIZE}",
                "training.num_workers=0",
                f"training.target_labels=[{','.join([f'label_{j}' for j in range(MOCK_NUM_LABELS)])}]",
                "training.early_stopping_patience=2",
                "model=mlp_visual",
                f"model.params.in_features={MOCK_IN_FEATURES}",
                "model.params.hidden_dims=[4]",
                "logging.logger_name=None",
            ],
        )

    return cfg
