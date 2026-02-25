import json
import logging
import os
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from common.data.dataset import FeatureDataset
from common.eval import resolve_checkpoint_dir
from common.utils import set_seed
from common.train import (
    build_epoch_payload,
    finalize_wandb_run,
    init_wandb_run,
    load_resume_state,
    log_wandb_metrics,
    resolve_path,
    restore_rng_state,
    save_training_state,
    torch_load_full,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .loops import evaluate_epoch, train_epoch


log = logging.getLogger(__name__)


def _find_latest_state(cfg: DictConfig, exclude_run_dir: Optional[Path]) -> Optional[Path]:
    """Find the newest resumable state checkpoint under the job base directory.

    Args:
        cfg: Hydra/OmegaConf config with optional ``paths.job_base_dir``.
        exclude_run_dir: Optional run directory to skip when searching.

    Returns:
        Path to the most recent existing ``checkpoints/last_state.pt`` file, or
        ``None`` when no candidate is found.

    Logic:
        Resolve the configured job base directory, sort run folders by mtime
        (newest first), skip ``exclude_run_dir``, and return the first run that
        contains ``last_state.pt``.
    """
    job_base_dir_str = OmegaConf.select(cfg, "paths.job_base_dir", default=None)
    if not job_base_dir_str:
        return None

    try:
        job_base_dir = resolve_path(cfg, job_base_dir_str)
    except Exception:
        return None

    if not job_base_dir.exists():
        return None

    exclude_resolved = exclude_run_dir.resolve() if exclude_run_dir else None

    try:
        candidates = sorted(
            [p for p in job_base_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return None

    for run_dir in candidates:
        try:
            run_dir_resolved = run_dir.resolve()
        except Exception:
            run_dir_resolved = run_dir

        if exclude_resolved and run_dir_resolved == exclude_resolved:
            continue

        candidate_state = run_dir_resolved / "checkpoints" / "last_state.pt"
        if candidate_state.exists():
            return candidate_state

    return None


def _update_latest_run_pointer(
    cfg: DictConfig,
    run_dir: Path,
    checkpoint_path: Optional[Path],
    state_path: Optional[Path],
    best_val_auprc: float,
    best_val_auroc: float,
    best_val_f1_macro: float,
) -> None:
    """Write/update the latest-run pointer JSON with current artifact metadata.

    Args:
        cfg: Hydra/OmegaConf config with optional ``paths.latest_run_pointer``.
        run_dir: Directory of the active training run.
        checkpoint_path: Optional final model checkpoint path.
        state_path: Optional resumable state checkpoint path.
        best_val_auprc: Best validation AUPRC.
        best_val_auroc: Best validation AUROC.
        best_val_f1_macro: Best validation macro F1.

    Returns:
        ``None``.

    Logic:
        If pointer path is configured, create parent directories, write a JSON
        payload with run/checkpoint/state paths and best metrics, and log
        warnings on failure instead of raising.
    """
    pointer_path_str = OmegaConf.select(cfg, "paths.latest_run_pointer", default=None)
    if not pointer_path_str:
        return

    try:
        pointer_path = resolve_path(cfg, pointer_path_str)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "run_dir": str(run_dir.resolve()),
            "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path and checkpoint_path.exists() else None,
            "state_checkpoint": str(state_path.resolve()) if state_path and state_path.exists() else None,
            "primary_metric": "auprc",
            "best_val_auprc": float(best_val_auprc),
            "best_val_auroc": float(best_val_auroc),
            "best_val_f1_macro": float(best_val_f1_macro),
            "best_val_metrics": {
                "auprc": float(best_val_auprc),
                "auroc": float(best_val_auroc),
                "f1_macro": float(best_val_f1_macro),
            },
        }

        pointer_path.write_text(json.dumps(payload, indent=2))
        log.debug("Latest run pointer updated: %s", pointer_path)
    except Exception as exc:
        log.warning("Unable to update latest run pointer at %s: %s", pointer_path_str, exc)

def train_model(cfg: DictConfig) -> float:
    """Run end-to-end training, validation, checkpointing, and logging.

    Args:
        cfg: Hydra/OmegaConf training configuration.

    Returns:
        Best validation AUPRC achieved during the run.

    Logic:
        Initializes reproducibility/device, builds datasets and loaders,
        instantiates model/optimizer/loss, optionally restores resume state,
        executes epoch loop with early stopping and metric logging, persists
        state and final checkpoint artifacts, updates latest-run pointers, and
        finalizes W&B logging.
    """
    log.info("Starting training pipeline...")
    # Rendering the full OmegaConf to YAML may trigger interpolations that
    # rely on Hydra runtime state (e.g. ${hydra:...}). During unit tests we do
    # not run inside the Hydra application context so those interpolations can
    # raise. Attempt a safe render and fall back to a non-resolving str() if it
    # fails.
    is_optuna_run = os.environ.get("CT_RATE_OPTUNA_RUN") == "1"
    if is_optuna_run:
        log.info("Full configuration dump skipped for Optuna trial run.")
    else:
        try:
            cfg_repr = OmegaConf.to_yaml(cfg)
        except Exception as exc:
            log.warning("Could not render full config to YAML (skipping detailed dump): %s", exc)
            # str(cfg) is safe and will not attempt to resolve interpolations
            cfg_repr = str(cfg)
        log.info(f"Full configuration:\n{cfg_repr}")

    # --- 1. Set Up Reproducibility & Device ---
    set_seed(cfg.utils.seed)
    log.info(f"Using seed: {cfg.utils.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    checkpoint_dir = resolve_checkpoint_dir(cfg)
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir_path / "last_state.pt"
    run_dir = checkpoint_dir_path.parent

    wandb_module: Optional[Any] = None
    wandb_run: Optional[Any] = None
    interrupted = False
    start_epoch = 1
    best_val_auprc = 0.0
    best_val_auroc = 0.0
    best_val_f1_macro = 0.0
    epochs_no_improve = 0
    best_model_state: Optional[Any] = None
    best_metrics: Optional[Dict[str, Any]] = None

    # --- 2. Calculate Dynamic Parameters ---
    # Dynamically determine the number of labels from the config list
    num_labels = len(cfg.training.target_labels)
    log.info(f"Resolved {num_labels} target labels from config.")

    # --- 3. Load Data ---
    log.info(f"Loading data from data_root: {cfg.paths.data_root}")
    
    preload_features = OmegaConf.select(cfg, "data.preload_features", default=False)
    if preload_features:
        log.info("Dataset preloading enabled; features will be loaded into RAM prior to training.")
        if cfg.training.num_workers > 0:
            log.warning(
                "Preloading with num_workers=%d will replicate the dataset per worker. "
                "Consider setting num_workers=0 if memory becomes an issue.",
                cfg.training.num_workers,
            )

    # Common dataset args
    text_feature_col = OmegaConf.select(cfg, "data.columns.text_feature", default=None)

    dataset_args = {
        "data_root": cfg.paths.data_root,
        "target_labels": list(cfg.training.target_labels),
        "visual_feature_col": cfg.data.columns.visual_feature,
        "text_feature_col": text_feature_col if text_feature_col else None,
        "preload": bool(preload_features),
    }
    
    # Common dataloader args
    loader_args = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.training.num_workers,
        "pin_memory": True,
    }

    # Instantiate Train Loader
    train_manifest_path = os.path.normpath(
        os.path.join(cfg.paths.manifest_dir, cfg.data.train_manifest)
    )
    def _load_dataset(manifest_path: str) -> FeatureDataset:
        """Load one manifest into ``FeatureDataset`` with text-column fallback.

        Args:
            manifest_path: Path to the manifest CSV to load.

        Returns:
            Initialized ``FeatureDataset``.

        Logic:
            Attempt normal load first; if manifest is missing the configured
            text feature column, retry with text features disabled.
        """
        try:
            return FeatureDataset(manifest_path=manifest_path, **dataset_args)
        except ValueError as exc:
            text_col = dataset_args.get("text_feature_col")
            if text_col and "Text feature column" in str(exc):
                log.info(
                    "Manifest %s missing text feature column '%s'; reloading without text features.",
                    manifest_path,
                    text_col,
                )
                fallback_args = dict(dataset_args)
                fallback_args["text_feature_col"] = None
                return FeatureDataset(manifest_path=manifest_path, **fallback_args)
            raise

    dataset_train = _load_dataset(train_manifest_path)
    log.info(f"Loaded training data from: {train_manifest_path}")

    auto_split_enabled = bool(OmegaConf.select(cfg, "data.auto_split.enabled", default=False))
    if auto_split_enabled:
        val_fraction = float(OmegaConf.select(cfg, "data.auto_split.val_fraction", default=0.1))
        stratify_enabled = bool(OmegaConf.select(cfg, "data.auto_split.stratify", default=True))
        split_seed = OmegaConf.select(cfg, "data.auto_split.seed", default=None)
        split_seed = cfg.utils.seed if split_seed is None else int(split_seed)
        group_column = "volumename"
        group_separator = "_"
        group_remove_last = True
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("data.auto_split.val_fraction must be between 0 and 1.")

        manifest_df = dataset_train.manifest
        label_cols = list(cfg.training.target_labels)
        stratify_labels = None

        if group_column not in manifest_df.columns:
            raise ValueError(
                f"Auto-split requires '{group_column}' column in the manifest."
            )

        group_series = manifest_df[group_column].astype(str)
        if group_remove_last:
            group_keys = group_series.str.rsplit(str(group_separator), n=1).str[0]
        else:
            group_keys = group_series

        label_frame = manifest_df[label_cols].fillna(0).astype(int)
        group_label_frame = (
            pd.concat([group_keys.rename("group_key"), label_frame], axis=1)
            .groupby("group_key", sort=False)
            .max()
        )

        if stratify_enabled:
            if len(label_cols) == 1:
                stratify_labels = group_label_frame.iloc[:, 0]
            else:
                stratify_labels = group_label_frame.astype(str).agg("|".join, axis=1)

        try:
            train_groups, val_groups = train_test_split(
                group_label_frame.index.values,
                test_size=val_fraction,
                random_state=split_seed,
                shuffle=True,
                stratify=stratify_labels,
            )
        except ValueError as exc:
            log.warning(
                "Grouped stratified split failed (%s); falling back to random split.",
                exc,
            )
            train_groups, val_groups = train_test_split(
                group_label_frame.index.values,
                test_size=val_fraction,
                random_state=split_seed,
                shuffle=True,
                stratify=None,
            )

        train_mask = group_keys.isin(train_groups)
        val_mask = group_keys.isin(val_groups)
        train_idx = manifest_df.index[train_mask].values
        val_idx = manifest_df.index[val_mask].values
        log.info(
            "Auto-split grouped by '%s' (%d groups).",
            group_column,
            group_label_frame.shape[0],
        )

        dataset_val = Subset(dataset_train, val_idx)
        dataset_train = Subset(dataset_train, train_idx)
        dataloader_train = DataLoader(dataset_train, shuffle=True, **loader_args)
        dataloader_val = DataLoader(dataset_val, shuffle=False, **loader_args)
        log.info(
            "Auto-split enabled: %d train / %d val (val_fraction=%.2f).",
            len(train_idx),
            len(val_idx),
            val_fraction,
        )
    else:
        dataloader_train = DataLoader(dataset_train, shuffle=True, **loader_args)
        # Instantiate Validation Loader
        val_manifest_path = os.path.normpath(
            os.path.join(cfg.paths.manifest_dir, cfg.data.val_manifest)
        )
        dataset_val = _load_dataset(val_manifest_path)
        dataloader_val = DataLoader(dataset_val, shuffle=False, **loader_args)
        log.info(f"Loaded validation data from: {val_manifest_path}")
    
    # --- 4. Initialize Model ---
    # We can now safely log the model_name from the config
    log.info(f"Instantiating model: {cfg.model.model_name}") 
    
    # Instantiate the model (e.g., MLP) using the 'params' node
    # This prevents passing 'model_name' to the constructor
    model = hydra.utils.instantiate(
        cfg.model.params, 
        _target_=cfg.model._target_, # We must now pass _target_ manually
        out_features=num_labels,     # We still add 'out_features' dynamically
        _recursive_=False
    ).to(device)

    wandb_handle = init_wandb_run(
        cfg=cfg,
        run_dir=run_dir,
        model=model,
        train_dataset_size=len(dataset_train) if hasattr(dataset_train, "__len__") else None,
        val_dataset_size=len(dataset_val) if hasattr(dataset_val, "__len__") else None,
    )
    if wandb_handle:
        wandb_module, wandb_run = wandb_handle
    else:
        resolved_logger_name = OmegaConf.select(cfg, "logging.logger_name", default=None)
        if resolved_logger_name:
            log.info("Logger %s configured but W&B run was not started.", resolved_logger_name)
        else:
            log.info("Experiment logger disabled via configuration.")

    metrics_path = run_dir / "metrics.jsonl"
    best_metrics_path = run_dir / "best_metrics.json"

    try:
        # --- 5. Initialize Optimizer and Loss ---
        weight_decay = float(OmegaConf.select(cfg, "training.weight_decay", default=0.0))
        log.info(
            "Instantiating optimizer (LR: %s, weight_decay: %s)",
            cfg.training.learning_rate,
            weight_decay,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.training.learning_rate, weight_decay=weight_decay
        )

        log.info("Instantiating loss function.")
        criterion = hydra.utils.instantiate(cfg.training.loss)

        resume_enabled = bool(OmegaConf.select(cfg, "training.resume.enabled", default=False))
        resume_override = OmegaConf.select(cfg, "training.resume.state_path", default=None)

        resume_state, resume_state_path = load_resume_state(
            cfg=cfg,
            resume_enabled=resume_enabled,
            default_state_path=state_path,
            override_path=resume_override,
        )

        if resume_enabled and resume_state is None:
            latest_state = _find_latest_state(cfg, exclude_run_dir=run_dir)
            if latest_state and latest_state != resume_state_path:
                try:
                    resume_state = torch_load_full(latest_state)
                    resume_state_path = latest_state
                except Exception as exc:
                    log.warning("Failed to load fallback resume checkpoint at %s: %s", latest_state, exc)

        if resume_state:
            model_state = resume_state.get("model")
            optimizer_state = resume_state.get("optimizer")
            if model_state:
                model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)

            start_epoch = int(resume_state.get("epoch", 0)) + 1
            best_val_auprc = float(
                resume_state.get(
                    "best_val_auprc",
                    resume_state.get("best_primary_metric", resume_state.get("best_val_auroc", 0.0)),
                )
            )
            best_val_auroc = float(resume_state.get("best_val_auroc", 0.0))
            best_val_f1_macro = float(resume_state.get("best_val_f1_macro", 0.0))
            epochs_no_improve = int(resume_state.get("epochs_no_improve", 0))
            best_model_state = resume_state.get("best_model_state")

            restore_rng_state(resume_state.get("rng_state"))

            log.info(
                "Resuming training from %s (next epoch=%d, best_val_auprc=%.4f)",
                resume_state_path,
                start_epoch,
                best_val_auprc,
            )
        elif resume_enabled:
            log.info("Resume requested but no checkpoint restored; starting fresh training.")

        # Ensure there is always a state snapshot for the current run, even if we
        # exit before completing an epoch.
        save_training_state(
            path=state_path,
            epoch=start_epoch - 1,
            model=model,
            optimizer=optimizer,
            best_val_auprc=best_val_auprc,
            best_val_auroc=best_val_auroc,
            epochs_no_improve=epochs_no_improve,
            best_model_state=best_model_state,
        )
        _update_latest_run_pointer(
            cfg=cfg,
            run_dir=run_dir,
            checkpoint_path=None,
            state_path=state_path,
            best_val_auprc=best_val_auprc,
            best_val_auroc=best_val_auroc,
            best_val_f1_macro=best_val_f1_macro,
        )

        # --- 7. Run Training Loop ---
        log.info(
            "Starting training for %s epochs (beginning at epoch %s).",
            cfg.training.max_epochs,
            start_epoch,
        )

        # Early stopping parameters
        patience = OmegaConf.select(
            cfg, "training.early_stopping_patience", default=10
        )
        log.info(f"Early stopping patience set to {patience} epochs.")

        try:
            for epoch in range(start_epoch, cfg.training.max_epochs + 1):

                # Run one epoch of training
                train_loss = train_epoch(
                    model, dataloader_train, optimizer, criterion, device
                )

                # Run one epoch of validation
                val_loss, val_metrics = evaluate_epoch(
                    model,
                    dataloader_val,
                    criterion,
                    device,
                    label_names=list(cfg.training.target_labels),
                )

                val_auprc = float(val_metrics.get("auprc", 0.0))
                val_auroc = float(val_metrics.get("auroc", 0.0))
                val_f1_macro = float(val_metrics.get("f1_macro", 0.0))

                log.info(
                    f"Epoch {epoch}/{cfg.training.max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUPRC: {val_auprc:.4f} | "
                    f"Val AUROC: {val_auroc:.4f}"
                    f" | Val F1 (macro): {val_f1_macro:.4f}"
                )

                prev_best = best_val_auprc
                improved = val_auprc > prev_best

                # --- Early Stopping Check ---
                if improved:
                    best_val_auprc = val_auprc
                    best_val_auroc = val_auroc
                    best_val_f1_macro = val_f1_macro
                    epochs_no_improve = 0
                    # Store the state of the best performing model
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_metrics = {
                        "epoch": int(epoch),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "val_metrics": val_metrics,
                    }
                    try:
                        best_metrics_path.write_text(json.dumps(best_metrics, indent=2))
                    except Exception as exc:
                        log.warning("Failed to write best_metrics.json: %s", exc)
                    log.debug(
                        "New best val AUPRC: %.4f (AUROC: %.4f, F1-macro: %.4f). Resetting patience.",
                        best_val_auprc,
                        best_val_auroc,
                        best_val_f1_macro,
                    )
                else:
                    epochs_no_improve += 1
                    log.debug(
                        f"Val AUPRC did not improve. Patience: {epochs_no_improve}/{patience}"
                    )

                wandb_payload: Optional[Dict[str, Any]] = None
                if wandb_run:
                    wandb_payload = build_epoch_payload(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_metrics=val_metrics,
                        best_val_auprc=best_val_auprc,
                        best_val_auroc=best_val_auroc,
                        best_val_f1_macro=best_val_f1_macro,
                        epochs_no_improve=epochs_no_improve,
                        improved=improved,
                    )

                metrics_row = {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_metrics": val_metrics,
                    "best_val_auprc": float(best_val_auprc),
                    "best_val_auroc": float(best_val_auroc),
                    "best_val_f1_macro": float(best_val_f1_macro),
                }
                try:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics_row) + "\n")
                except Exception as exc:
                    log.warning("Failed to append metrics.jsonl: %s", exc)

                save_training_state(
                    path=state_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_val_auprc=best_val_auprc,
                    best_val_auroc=best_val_auroc,
                    epochs_no_improve=epochs_no_improve,
                    best_model_state=best_model_state,
                )
                _update_latest_run_pointer(
                    cfg=cfg,
                    run_dir=run_dir,
                    checkpoint_path=None,
                    state_path=state_path,
                    best_val_auprc=best_val_auprc,
                    best_val_auroc=best_val_auroc,
                    best_val_f1_macro=best_val_f1_macro,
                )

                # Check if patience has been exceeded
                stop_early = epochs_no_improve >= patience
                if stop_early:
                    log.info(
                        f"Early stopping triggered after {patience} epochs "
                        f"without improvement. Best Val AUPRC: {best_val_auprc:.4f}"
                    )
                    if wandb_payload is not None:
                        wandb_payload["events/early_stopping"] = 1

                if wandb_run and wandb_payload is not None:
                    log_wandb_metrics(wandb_module, wandb_payload, step=epoch)

                if stop_early:
                    break
        except KeyboardInterrupt:
            interrupted = True
            log.info("Training interrupted by user. Latest state saved to: %s", state_path)

        if not interrupted:
            log.info("Training complete.")
        else:
            log.info("Exiting early due to interruption.")

        # --- 8. Save Artifacts ---
        checkpoint_path = checkpoint_dir_path / "final_model.pt"

        # Save the best model state captured during training
        if best_model_state:
            torch.save(best_model_state, checkpoint_path)
            log.info(
                f"Best model checkpoint (Val AUPRC: {best_val_auprc:.4f}) "
                f"saved to: {checkpoint_path}"
            )
        else:
            # Fallback: Save final model if training ended early or no improvement
            torch.save(model.state_dict(), checkpoint_path)
            log.warning(
                f"Final model checkpoint saved to: {checkpoint_path} "
                "(No validation improvement was captured)"
            )

        _update_latest_run_pointer(
            cfg=cfg,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            state_path=state_path,
            best_val_auprc=best_val_auprc,
            best_val_auroc=best_val_auroc,
            best_val_f1_macro=best_val_f1_macro,
        )

        if interrupted:
            raise KeyboardInterrupt

        # Return the best validation metric
        return best_val_auprc
    finally:
        finalize_wandb_run(
            wandb_module,
            wandb_run,
            best_val_auprc,
            best_val_auroc,
            best_val_f1_macro,
            interrupted,
        )


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint that validates job naming and launches training.

    Args:
        cfg: Hydra/OmegaConf configuration populated by ``@hydra.main``.

    Returns:
        ``None``.

    Logic:
        Resolve candidate job names from environment/config/runtime, require a
        non-empty job name for predictable artifact layout, then call
        ``train_model`` and re-raise any failure after logging.
    """
    try:
        # Enforce that a hydra job name is provided so runs are named and
        # artifacts are stored predictably. This prevents accidental
        # unlabelled runs that drop into the `manual_run` fallback.
        job_name_candidates: List[str] = []

        env_job_name = os.environ.get("HYDRA_JOB_NAME")
        if env_job_name:
            job_name_candidates.append(env_job_name)

        cfg_job_name = OmegaConf.select(cfg, "hydra.job.name", default=None)
        if cfg_job_name:
            job_name_candidates.append(cfg_job_name)

        try:
            runtime_job_name = HydraConfig.get().job.name
        except ValueError:
            runtime_job_name = None
        if runtime_job_name:
            job_name_candidates.append(runtime_job_name)

        job_name = next((name for name in job_name_candidates if name), None)

        log.debug("Resolved hydra job name candidates=%s -> %s", job_name_candidates, job_name)

        if not job_name:
            raise ValueError(
                "hydra.job.name is required for training runs. "
                "Provide it via the config (configs/config.yaml) or on the CLI: "
                "python -m classification.train hydra.job.name=your_job_name"
            )

        train_model(cfg)
    except Exception as e:
        log.exception(f"An error occurred during training: {e}")
        # Optionally re-raise or exit
        raise


if __name__ == "__main__":
    main()