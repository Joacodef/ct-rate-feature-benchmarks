import logging
import os
from typing import Dict, List, Tuple

import hydra
import json
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
    """Build test dataloaders for each manifest name.

    Args:
        manifests: Manifest filenames to load.
        dataset_args: Keyword arguments forwarded to ``FeatureDataset``.
        loader_args: Keyword arguments forwarded to ``DataLoader``.
        manifest_root: Base directory containing manifest files.

    Returns:
        List of ``(manifest_name, dataloader)`` pairs.

    Logic:
        Resolve each manifest path under ``manifest_root``, construct the
        dataset and dataloader, and collect them in input order.
    """
    test_loaders: List[Tuple[str, DataLoader]] = []
    for manifest_name in manifests:
        manifest_path = os.path.normpath(os.path.join(manifest_root, manifest_name))
        dataset = FeatureDataset(manifest_path=manifest_path, **dataset_args)
        dataloader = DataLoader(dataset, shuffle=False, **loader_args)
        test_loaders.append((manifest_name, dataloader))
        log.info("Loaded test data from: %s", manifest_path)
    return test_loaders


def _resolve_detailed_metrics_dir(cfg: DictConfig) -> str:
    """Resolve output directory for detailed per-class metric reports.

    Args:
        cfg: Hydra/OmegaConf config with optional runtime/output path keys.

    Returns:
        Absolute path to a ``detailed_metrics`` directory.

    Logic:
        Check preferred path keys in order (Hydra runtime/output locations
        first), convert relative paths against ``hydra.runtime.cwd``, then
        append ``detailed_metrics``. Fall back to ``<cwd>/detailed_metrics``.
    """

    candidate_keys = [
        "hydra.runtime.output_dir",
        "hydra.run.dir",
        "paths.run_dir",
        "paths.output_dir",
    ]

    base_dir = OmegaConf.select(cfg, "hydra.runtime.cwd", default=os.getcwd())
    base_dir = os.path.normpath(base_dir)

    for key in candidate_keys:
        try:
            candidate = OmegaConf.select(cfg, key)
        except Exception:
            candidate = None

        if candidate:
            candidate_path = os.path.normpath(str(candidate))
            if not os.path.isabs(candidate_path):
                candidate_path = os.path.normpath(os.path.join(base_dir, candidate_path))
            return os.path.join(candidate_path, "detailed_metrics")

    return os.path.join(base_dir, "detailed_metrics")


def evaluate_model(cfg: DictConfig) -> Dict[str, float]:
    """Evaluate a checkpointed model across configured test manifests.

    Args:
        cfg: Hydra/OmegaConf evaluation configuration.

    Returns:
        Flat metric dictionary keyed by manifest (AUPRC, AUROC, and F1-macro
        entries for each test manifest).

    Logic:
        Set seed/device, build datasets and loaders, instantiate model/loss,
        resolve and load checkpoint, run per-manifest evaluation, aggregate
        summary metrics, and optionally write detailed per-class JSON reports.
    """
    log.info("Starting evaluation...")
    log.info("Full configuration:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.utils.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    target_labels = list(cfg.training.target_labels)
    negative_class_name = OmegaConf.select(
        cfg, "evaluation.negative_class_name", default="No pathology"
    )
    detailed_metrics_dir = _resolve_detailed_metrics_dir(cfg)

    dataset_args = {
        "data_root": cfg.paths.data_root,
        "target_labels": target_labels,
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
        out_features=len(target_labels),
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
        # Try to request per-class metrics by passing label names. Some call-sites
        # (and tests) may monkeypatch or expect the older signature, so fall
        # back to the 4-arg call if the function does not accept the extra
        # parameter.
        try:
            test_loss, test_metrics = evaluate_epoch(
                model,
                dataloader,
                criterion,
                device,
                target_labels,
                negative_class_name,
            )
        except TypeError:
            test_loss, test_metrics = evaluate_epoch(model, dataloader, criterion, device)
        metric_key_auprc = f"test_{manifest_name}_auprc"
        metric_key_auroc = f"test_{manifest_name}_auroc"
        metric_key_f1 = f"test_{manifest_name}_f1_macro"
        results[metric_key_auprc] = float(test_metrics.get("auprc", 0.0))
        results[metric_key_auroc] = float(test_metrics.get("auroc", 0.0))
        results[metric_key_f1] = float(test_metrics.get("f1_macro", 0.0))
        log.info(
            "Test Set: %s | Loss: %.4f | AUPRC: %.4f | AUROC: %.4f | F1-macro: %.4f",
            manifest_name,
            test_loss,
            test_metrics.get("auprc", 0.0),
            test_metrics.get("auroc", 0.0),
            test_metrics.get("f1_macro", 0.0),
        )

        # If per-class metrics are available, write a detailed per-manifest
        # report alongside the rest of the Hydra evaluation artifacts.
        if "per_class" in test_metrics:
            try:
                os.makedirs(detailed_metrics_dir, exist_ok=True)
                report_path = os.path.join(
                    detailed_metrics_dir, f"{manifest_name}_detailed_metrics.json"
                )
                payload = {
                    "manifest": manifest_name,
                    "loss": float(test_loss),
                    "auprc": float(test_metrics.get("auprc", 0.0)),
                    "auroc": float(test_metrics.get("auroc", 0.0)),
                    "f1_macro": float(test_metrics.get("f1_macro", 0.0)),
                    "per_class": test_metrics.get("per_class"),
                }
                with open(report_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                log.info("Wrote detailed per-class metrics to: %s", report_path)
            except Exception as exc:
                log.warning("Failed to write per-class metrics report: %s", exc)

    return results


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for evaluation runs.

    Args:
        cfg: Hydra/OmegaConf config injected by ``@hydra.main``.

    Returns:
        ``None``.

    Logic:
        Execute ``evaluate_model`` and log/re-raise any exception so calling
        environments receive a failing exit status.
    """
    try:
        evaluate_model(cfg)
    except Exception as exc:
        log.exception("An error occurred during evaluation: %s", exc)
        raise


if __name__ == "__main__":
    ensure_eval_run_dir_override()
    main()
