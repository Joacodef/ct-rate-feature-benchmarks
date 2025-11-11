import logging
import os
from typing import Dict, List, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from common.data.dataset import FeatureDataset
from common.utils import set_seed
from common.eval import ensure_eval_run_dir_override, resolve_evaluation_checkpoint

from .loops import evaluate_epoch

log = logging.getLogger(__name__)


def _build_test_loaders(
    manifests: List[str],
    dataset_args: Dict,
    loader_args: Dict,
    manifest_root: str,
) -> List[Tuple[str, DataLoader]]:
    """Create DataLoaders for each configured test manifest."""
    test_loaders: List[Tuple[str, DataLoader]] = []
    for manifest_name in manifests:
        manifest_path = os.path.normpath(os.path.join(manifest_root, manifest_name))
        dataset = FeatureDataset(manifest_path=manifest_path, **dataset_args)
        dataloader = DataLoader(dataset, shuffle=False, **loader_args)
        test_loaders.append((manifest_name, dataloader))
        log.info("Loaded test data from: %s", manifest_path)
    return test_loaders


log = logging.getLogger(__name__)


def evaluate_model(cfg: DictConfig) -> Dict[str, float]:
    """Evaluate a trained model on all configured test manifests."""
    log.info("Starting evaluation...")
    log.info("Full configuration:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.utils.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    dataset_args = {
        "data_root": cfg.paths.data_root,
        "target_labels": list(cfg.training.target_labels),
        "visual_feature_col": cfg.data.columns.visual_feature,
        "text_feature_col": cfg.data.columns.text_feature,
        "preload": bool(OmegaConf.select(cfg, "data.preload_features", default=False)),
    }

    loader_args = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.training.num_workers,
        "pin_memory": True,
    }

    test_loaders = _build_test_loaders(
        manifests=list(cfg.data.test_manifests),
        dataset_args=dataset_args,
        loader_args=loader_args,
        manifest_root=cfg.paths.manifest_dir,
    )

    model = hydra.utils.instantiate(
        cfg.model.params,
        _target_=cfg.model._target_,
        out_features=len(cfg.training.target_labels),
        _recursive_=False,
    ).to(device)

    criterion = hydra.utils.instantiate(cfg.training.loss)

    checkpoint_path = resolve_evaluation_checkpoint(cfg)
    checkpoint_path = os.path.normpath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Provide paths.checkpoint_path or update evaluation.checkpoint_name."
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    log.info("Loaded checkpoint: %s", checkpoint_path)

    results: Dict[str, float] = {}
    for manifest_name, dataloader in test_loaders:
        test_loss, test_metrics = evaluate_epoch(model, dataloader, criterion, device)
        metric_key = f"test_{manifest_name}_auroc"
        results[metric_key] = test_metrics["auroc"]
        log.info(
            "Test Set: %s | Loss: %.4f | AUROC: %.4f",
            manifest_name,
            test_loss,
            test_metrics["auroc"],
        )

    return results


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    try:
        evaluate_model(cfg)
    except Exception as exc:
        log.exception("An error occurred during evaluation: %s", exc)
        raise


if __name__ == "__main__":
    ensure_eval_run_dir_override()
    main()
