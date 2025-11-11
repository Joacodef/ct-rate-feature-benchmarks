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
    """Initialize a W&B run if requested by the configuration."""

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
    best_val_auroc: float,
    epochs_no_improve: int,
    improved: bool,
) -> Dict[str, Any]:
    """Flatten epoch metrics into a W&B-friendly dictionary."""

    payload: Dict[str, Any] = {
        "epoch": epoch,
        "train/loss": float(train_loss),
        "val/loss": float(val_loss),
        "val/auroc": float(val_metrics.get("auroc", 0.0)),
        "best/val_auroc": float(best_val_auroc),
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
    """Safely log metrics to W&B, shielding the caller from logging errors."""

    if not payload:
        return

    try:
        wandb_module.log(payload, step=step)
    except Exception as exc:
        log.warning("Failed to log metrics to W&B: %s", exc)


def finalize_wandb_run(
    wandb_module: Optional[Any],
    wandb_run: Optional[Any],
    best_val_auroc: float,
    interrupted: bool,
) -> None:
    """Finalize the W&B run, updating summary metrics if available."""

    if not wandb_module or not wandb_run:
        return

    try:
        wandb_run.summary["best_val_auroc"] = float(best_val_auroc)
        wandb_run.summary["interrupted"] = bool(interrupted)
    except Exception as exc:
        log.debug("Failed to update W&B summary: %s", exc)

    try:
        wandb_module.finish()
    except Exception as exc:
        log.warning("Failed to finalize W&B run: %s", exc)
