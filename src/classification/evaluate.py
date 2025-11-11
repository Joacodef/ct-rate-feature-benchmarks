import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError
from torch.utils.data import DataLoader

from common.data.dataset import FeatureDataset
from common.utils import set_seed

from .loops import evaluate_epoch

log = logging.getLogger(__name__)


def _ensure_eval_run_dir_override() -> None:
    """Ensure evaluation outputs land inside the evaluated run directory."""
    if any(arg.startswith("hydra.run.dir=") for arg in sys.argv[1:]):
        return

    def _value(arg: str) -> str:
        return arg.split("=", 1)[1].strip('"\'') if "=" in arg else ""

    job_name: Optional[str] = None
    outputs_root = "outputs"
    pointer_override: Optional[str] = None
    checkpoint_override: Optional[str] = None

    for arg in sys.argv[1:]:
        if arg.startswith("hydra.job.name="):
            job_name = _value(arg)
        elif arg.startswith("paths.outputs_root="):
            outputs_root = _value(arg)
        elif arg.startswith("paths.latest_run_pointer="):
            pointer_override = _value(arg)
        elif arg.startswith("paths.checkpoint_path="):
            checkpoint_override = _value(arg)

    run_dir: Optional[Path] = None

    if checkpoint_override:
        checkpoint_path = Path(checkpoint_override)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if checkpoint_path.exists():
            checkpoint_parent = checkpoint_path.parent
            if checkpoint_parent.name.lower() == "checkpoints":
                run_dir = checkpoint_parent.parent
            else:
                run_dir = checkpoint_parent

    if run_dir is None:
        pointer_path: Optional[Path] = None
        if pointer_override:
            pointer_path = Path(pointer_override)
            if not pointer_path.is_absolute():
                pointer_path = Path.cwd() / pointer_path
        elif job_name:
            pointer_path = Path(outputs_root) / job_name / "latest_run.json"
            if not pointer_path.is_absolute():
                pointer_path = Path.cwd() / pointer_path

        if pointer_path and pointer_path.exists():
            try:
                payload = json.loads(pointer_path.read_text())
            except Exception:
                payload = {}

            run_dir_str = payload.get("run_dir")
            checkpoint_from_pointer = payload.get("checkpoint")

            if run_dir_str:
                run_dir_candidate = Path(run_dir_str)
                if not run_dir_candidate.is_absolute():
                    run_dir_candidate = Path.cwd() / run_dir_candidate

                if job_name and job_name.lower() not in {part.lower() for part in run_dir_candidate.parts}:
                    run_dir_candidate = None

                if run_dir_candidate is not None:
                    run_dir = run_dir_candidate

            if run_dir is None and checkpoint_from_pointer:
                checkpoint_path = Path(checkpoint_from_pointer)
                run_dir = checkpoint_path.parent.parent

    if run_dir is None and job_name:
        candidate_root = Path(outputs_root) / job_name
        if not candidate_root.is_absolute():
            candidate_root = Path.cwd() / candidate_root

        if candidate_root.exists():
            def _is_run_dir(path: Path) -> bool:
                if not path.is_dir():
                    return False
                name_lower = path.name.lower()
                if name_lower in {"evaluation", ".hydra"}:
                    return False
                return True

            run_dir_candidates = [p for p in candidate_root.iterdir() if _is_run_dir(p)]
            if run_dir_candidates:
                run_dir = sorted(run_dir_candidates, key=lambda p: p.stat().st_mtime)[-1]

    if run_dir is not None:
        eval_root = (Path(run_dir).resolve() / "evaluation").as_posix()
        sys.argv.append(
            f"hydra.run.dir={eval_root}/${{now:%Y-%m-%d_%H-%M-%S}}"
        )
        return

    if job_name:
        fallback_root = (Path(outputs_root) / job_name / "evaluation").as_posix()
    else:
        fallback_root = Path("outputs/evaluation").as_posix()

    sys.argv.append(f"hydra.run.dir={fallback_root}/${{now:%Y-%m-%d_%H-%M-%S}}")


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


def _original_cwd(cfg: DictConfig) -> Path:
    """Return the original working directory (before Hydra chdir)."""
    base_dir = OmegaConf.select(cfg, "hydra.runtime.cwd", default=os.getcwd())
    return Path(base_dir)


def _as_path(cfg: DictConfig, path_str: str) -> Path:
    """Resolve a potentially relative path against the original cwd."""
    path = Path(path_str)
    if not path.is_absolute():
        path = _original_cwd(cfg) / path
    return path


def _checkpoint_from_pointer(cfg: DictConfig) -> Optional[str]:
    """Try loading the latest run pointer to locate a checkpoint."""
    pointer_str = OmegaConf.select(cfg, "paths.latest_run_pointer")
    if not pointer_str:
        return None

    pointer_path = _as_path(cfg, pointer_str)
    if not pointer_path.exists():
        return None

    try:
        payload = json.loads(pointer_path.read_text())
    except Exception as exc:
        log.warning("Failed to decode latest run pointer %s: %s", pointer_path, exc)
        return None

    checkpoint = payload.get("checkpoint")
    if checkpoint:
        checkpoint_path = _as_path(cfg, checkpoint)
        if checkpoint_path.exists():
            log.info("Resolved checkpoint via latest run pointer: %s", checkpoint_path)
            return str(checkpoint_path)

    run_dir = payload.get("run_dir")
    if run_dir:
        checkpoint_name = OmegaConf.select(
            cfg, "evaluation.checkpoint_name", default="final_model.pt"
        )
        candidate = _as_path(cfg, os.path.join(run_dir, "checkpoints", checkpoint_name))
        if candidate.exists():
            log.info("Resolved checkpoint via run_dir pointer: %s", candidate)
            return str(candidate)

    return None


def _resolve_checkpoint_path(cfg: DictConfig) -> str:
    """Determine which checkpoint file to load for evaluation."""
    explicit_path = OmegaConf.select(cfg, "paths.checkpoint_path")
    if explicit_path:
        path = _as_path(cfg, explicit_path)
        return str(path)

    pointer_checkpoint = _checkpoint_from_pointer(cfg)
    if pointer_checkpoint:
        return pointer_checkpoint

    try:
        checkpoint_dir = _as_path(cfg, cfg.paths.checkpoint_dir)
    except (InterpolationKeyError, AttributeError):
        job_name = OmegaConf.select(cfg, "hydra.job.name", default="manual_run")
        fallback_dir = _as_path(cfg, os.path.join("outputs", job_name, "checkpoints"))
        log.warning(
            "hydra.job.name not available; falling back to checkpoint directory: %s",
            fallback_dir,
        )
        checkpoint_dir = fallback_dir

    checkpoint_name = OmegaConf.select(
        cfg, "evaluation.checkpoint_name", default="final_model.pt"
    )
    return str(checkpoint_dir / checkpoint_name)


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

    checkpoint_path = _resolve_checkpoint_path(cfg)
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
    _ensure_eval_run_dir_override()
    main()
