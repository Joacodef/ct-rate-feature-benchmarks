"""Utility helpers for integrating Weights & Biases logging into training loops."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def init_wandb_run(
    cfg: DictConfig,
    run_dir: Path,
    model: nn.Module,
    train_dataset_size: Optional[int],
    val_dataset_size: Optional[int],
) -> Optional[Tuple[Any, Any]]:
    """Initialize Weights & Biases logging from training configuration.

    Args:
        cfg: Hydra/OmegaConf configuration containing logging options under
            ``logging.*``.
        run_dir: Directory to use as W&B local run directory.
        model: Model instance that may be watched for gradients/parameters.
        train_dataset_size: Optional number of train samples for metadata.
        val_dataset_size: Optional number of validation samples for metadata.

    Input:
        Logging configuration, runtime directory, model instance, and optional
        dataset cardinalities.

    Returns:
        ``(wandb_module, wandb_run)`` when initialization succeeds; otherwise
        ``None`` when W&B is disabled/unavailable or initialization fails.

    Logic:
        1. Enable only when ``logging.logger_name`` is ``wandb``.
        2. Import ``wandb`` lazily and fail gracefully if package is missing.
        3. Build init kwargs from config (project/entity/group/job_type/name)
           and tags.
        4. If run name is missing, derive it from Hydra job metadata.
        5. Convert config to serializable payload (resolved first, fallback to
           unresolved).
        6. Initialize run and optionally enable model watching.
        7. Push dataset size metadata and register metric step definitions.
        8. Return module/run handles for subsequent logging/finalization.
    """

    logger_name = OmegaConf.select(cfg, "logging.logger_name", default=None)
    if not logger_name or str(logger_name).lower() != "wandb":
        return None

    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - triggered when wandb missing
        log.warning("W&B logging requested but wandb is not installed: %s", exc)
        return None

    wandb_kwargs: Dict[str, Any] = {}
    for kwarg, conf_key in [
        ("project", "logging.project_name"),
        ("entity", "logging.entity"),
        ("group", "logging.group"),
        ("job_type", "logging.job_type"),
        ("name", "logging.run_name"),
    ]:
        value = OmegaConf.select(cfg, conf_key, default=None)
        if value:
            wandb_kwargs[kwarg] = value

    tags = OmegaConf.select(cfg, "logging.tags", default=None)
    if tags:
        wandb_kwargs["tags"] = [str(tag) for tag in tags]

    if "name" not in wandb_kwargs:
        try:
            hydra_cfg = HydraConfig.get()
            job_id = hydra_cfg.job.get("id") if hydra_cfg.job else None
            job_name = hydra_cfg.job.get("name") if hydra_cfg.job else None
            resolved_name = job_id or job_name
            if resolved_name:
                wandb_kwargs["name"] = resolved_name
            if job_name and "group" not in wandb_kwargs:
                wandb_kwargs["group"] = job_name
        except Exception:
            pass

    try:
        config_payload = OmegaConf.to_container(cfg, resolve=True)
    except Exception as exc:
        log.debug("Falling back to non-resolved config for W&B init: %s", exc)
        config_payload = OmegaConf.to_container(cfg, resolve=False)

    try:
        wandb_run = wandb.init(dir=str(run_dir), config=config_payload, **wandb_kwargs)
    except Exception as exc:
        log.warning("Failed to initialize W&B run: %s", exc)
        return None

    if wandb_run is None:
        log.warning("wandb.init returned None; proceeding without W&B logging.")
        return None

    log_model_cfg = OmegaConf.select(cfg, "logging.log_model", default=False)
    watch_log = OmegaConf.select(cfg, "logging.watch.log", default=None)
    watch_log_freq = OmegaConf.select(cfg, "logging.watch.log_freq", default=100)

    watch_mode = None
    if log_model_cfg:
        watch_mode = "all" if str(log_model_cfg).lower() == "all" else "gradients"
    elif watch_log:
        watch_mode = str(watch_log)

    if watch_mode:
        try:
            wandb.watch(model, log=watch_mode, log_freq=int(watch_log_freq or 100))
        except Exception as exc:
            log.warning("Unable to enable W&B model watching: %s", exc)

    dataset_info: Dict[str, Any] = {}
    if train_dataset_size is not None:
        dataset_info["dataset/train_samples"] = int(train_dataset_size)
    if val_dataset_size is not None:
        dataset_info["dataset/val_samples"] = int(val_dataset_size)
    if dataset_info:
        try:
            wandb.config.update(dataset_info, allow_val_change=True)
        except Exception as exc:
            log.debug("Failed to push dataset stats to W&B config: %s", exc)

    try:
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("best/*", step_metric="epoch")
        wandb.define_metric("training/*", step_metric="epoch")
    except Exception as exc:
        log.debug("Unable to register W&B metric definitions: %s", exc)

    run_identifier = getattr(wandb_run, "name", None) or getattr(wandb_run, "id", "unknown")
    log.info("W&B logging enabled for run: %s", run_identifier)

    return wandb, wandb_run


def build_epoch_payload(
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: Dict[str, Any],
    best_val_auprc: float,
    best_val_auroc: float,
    best_val_f1_macro: float,
    epochs_no_improve: int,
    improved: bool,
) -> Dict[str, Any]:
    """Build a flattened metrics payload for one training epoch.

    Args:
        epoch: Epoch index to associate with all logged metrics.
        train_loss: Training loss scalar for the epoch.
        val_loss: Validation loss scalar for the epoch.
        val_metrics: Validation metrics dictionary that may include global
            metrics (``auprc``, ``auroc``, ``f1_macro``) and optional per-class
            nested metrics under ``per_class``.
        best_val_auprc: Best validation AUPRC seen so far.
        best_val_auroc: Best validation AUROC seen so far.
        best_val_f1_macro: Best validation macro-F1 seen so far.
        epochs_no_improve: Count of consecutive epochs without improvement.
        improved: Whether the primary metric improved this epoch.

    Input:
        Scalar epoch statistics plus optional nested per-class validation
        metric dictionaries.

    Returns:
        Flat dictionary keyed with W&B-friendly names (e.g., ``train/*``,
        ``val/*``, ``best/*``, ``training/*``).

    Logic:
        1. Seed payload with core train/validation/best/training fields.
        2. Read optional ``per_class`` metrics.
        3. Flatten numeric per-class values into keys of the form
           ``val/<label>/<metric>``.
        4. Return payload ready for ``wandb.log``.
    """

    payload: Dict[str, Any] = {
        "epoch": epoch,
        "train/loss": float(train_loss),
        "val/loss": float(val_loss),
        "val/auprc": float(val_metrics.get("auprc", 0.0)),
        "val/auroc": float(val_metrics.get("auroc", 0.0)),
        "val/f1_macro": float(val_metrics.get("f1_macro", 0.0)),
        "best/val_auprc": float(best_val_auprc),
        "best/val_auroc": float(best_val_auroc),
        "best/val_f1_macro": float(best_val_f1_macro),
        "training/epochs_no_improve": int(epochs_no_improve),
        "training/best_improved": int(improved),
    }

    per_class = val_metrics.get("per_class")
    if isinstance(per_class, dict):
        for label_name, metric_dict in per_class.items():
            if not isinstance(metric_dict, dict):
                continue
            for metric_name, value in metric_dict.items():
                if isinstance(value, (int, float)):
                    payload[f"val/{label_name}/{metric_name}"] = float(value)

    return payload


def log_wandb_metrics(wandb_module: Any, payload: Dict[str, Any], step: int) -> None:
    """Log metrics to W&B with error shielding.

    Args:
        wandb_module: Imported ``wandb`` module instance.
        payload: Flattened metrics dictionary to log.
        step: Step value passed to ``wandb.log``.

    Input:
        W&B module handle and a potentially empty payload.

    Returns:
        ``None``.

    Logic:
        1. Exit early when payload is empty.
        2. Attempt ``wandb.log(payload, step=step)``.
        3. Log warning instead of raising to avoid interrupting training.
    """

    if not payload:
        return

    try:
        wandb_module.log(payload, step=step)
    except Exception as exc:
        log.warning("Failed to log metrics to W&B: %s", exc)


def finalize_wandb_run(
    wandb_module: Optional[Any],
    wandb_run: Optional[Any],
    best_val_auprc: float,
    best_val_auroc: float,
    best_val_f1_macro: float,
    interrupted: bool,
) -> None:
    """Finalize W&B logging and persist end-of-run summary metrics.

    Args:
        wandb_module: Optional imported ``wandb`` module handle.
        wandb_run: Optional active run object returned by ``wandb.init``.
        best_val_auprc: Final best validation AUPRC.
        best_val_auroc: Final best validation AUROC.
        best_val_f1_macro: Final best validation macro-F1.
        interrupted: Whether training terminated early/interrupted.

    Input:
        Active run handles and final summary values from training loop.

    Returns:
        ``None``.

    Logic:
        1. No-op if module/run handles are unavailable.
        2. Best-effort update run summary with final best metrics, primary
           metric name, and interrupted flag.
        3. Best-effort call ``wandb.finish()`` and warn only on failure.
    """

    if not wandb_module or not wandb_run:
        return

    try:
        wandb_run.summary["best_val_auprc"] = float(best_val_auprc)
        wandb_run.summary["best_val_auroc"] = float(best_val_auroc)
        wandb_run.summary["best_val_f1_macro"] = float(best_val_f1_macro)
        wandb_run.summary["primary_metric"] = "auprc"
        wandb_run.summary["interrupted"] = bool(interrupted)
    except Exception as exc:
        log.debug("Failed to update W&B summary: %s", exc)

    try:
        wandb_module.finish()
    except Exception as exc:
        log.warning("Failed to finalize W&B run: %s", exc)
