import json
import logging
import os
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
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
from torch.utils.data import DataLoader

from .loops import evaluate_epoch, train_epoch


log = logging.getLogger(__name__)


def _find_latest_state(cfg: DictConfig, exclude_run_dir: Optional[Path]) -> Optional[Path]:
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
    best_val_auroc: float,
) -> None:
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
            "best_val_auroc": float(best_val_auroc),
        }

        pointer_path.write_text(json.dumps(payload, indent=2))
        log.debug("Latest run pointer updated: %s", pointer_path)
    except Exception as exc:
        log.warning("Unable to update latest run pointer at %s: %s", pointer_path_str, exc)

def train_model(cfg: DictConfig) -> float:
    """
    Main training and evaluation function.

    This function orchestrates the entire pipeline:
    1. Sets up seeding for reproducibility.
    2. Loads and prepares datasets and dataloaders.
    3. Initializes the model architecture.
    4. Initializes the optimizer and loss function.
    5. Runs the training loop with validation.
    6. Logs metrics during training.
    7. Saves artifacts.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        The primary metric score (e.g., test AUROC) for optimization.
    """
    log.info("Starting training pipeline...")
    # Rendering the full OmegaConf to YAML may trigger interpolations that
    # rely on Hydra runtime state (e.g. ${hydra:...}). During unit tests we do
    # not run inside the Hydra application context so those interpolations can
    # raise. Attempt a safe render and fall back to a non-resolving str() if it
    # fails.
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
    best_val_auroc = 0.0
    epochs_no_improve = 0
    best_model_state: Optional[Any] = None

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
    dataloader_train = DataLoader(dataset_train, shuffle=True, **loader_args)
    log.info(f"Loaded training data from: {train_manifest_path}")

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

    try:
        # --- 5. Initialize Optimizer and Loss ---
        log.info(f"Instantiating optimizer (LR: {cfg.training.learning_rate})")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.training.learning_rate
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
            best_val_auroc = float(resume_state.get("best_val_auroc", 0.0))
            epochs_no_improve = int(resume_state.get("epochs_no_improve", 0))
            best_model_state = resume_state.get("best_model_state")

            restore_rng_state(resume_state.get("rng_state"))

            log.info(
                "Resuming training from %s (next epoch=%d, best_val_auroc=%.4f)",
                resume_state_path,
                start_epoch,
                best_val_auroc,
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
            best_val_auroc=best_val_auroc,
            epochs_no_improve=epochs_no_improve,
            best_model_state=best_model_state,
        )
        _update_latest_run_pointer(
            cfg=cfg,
            run_dir=run_dir,
            checkpoint_path=None,
            state_path=state_path,
            best_val_auroc=best_val_auroc,
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

                val_auroc = float(val_metrics.get("auroc", 0.0))

                log.info(
                    f"Epoch {epoch}/{cfg.training.max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUROC: {val_auroc:.4f}"
                )

                prev_best = best_val_auroc
                improved = val_auroc > prev_best

                # --- Early Stopping Check ---
                if improved:
                    best_val_auroc = val_auroc
                    epochs_no_improve = 0
                    # Store the state of the best performing model
                    best_model_state = copy.deepcopy(model.state_dict())
                    log.debug(f"New best val AUROC: {best_val_auroc:.4f}. Resetting patience.")
                else:
                    epochs_no_improve += 1
                    log.debug(
                        f"Val AUROC did not improve. Patience: {epochs_no_improve}/{patience}"
                    )

                wandb_payload: Optional[Dict[str, Any]] = None
                if wandb_run:
                    wandb_payload = build_epoch_payload(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_metrics=val_metrics,
                        best_val_auroc=best_val_auroc,
                        epochs_no_improve=epochs_no_improve,
                        improved=improved,
                    )

                save_training_state(
                    path=state_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_val_auroc=best_val_auroc,
                    epochs_no_improve=epochs_no_improve,
                    best_model_state=best_model_state,
                )
                _update_latest_run_pointer(
                    cfg=cfg,
                    run_dir=run_dir,
                    checkpoint_path=None,
                    state_path=state_path,
                    best_val_auroc=best_val_auroc,
                )

                # Check if patience has been exceeded
                stop_early = epochs_no_improve >= patience
                if stop_early:
                    log.info(
                        f"Early stopping triggered after {patience} epochs "
                        f"without improvement. Best Val AUROC: {best_val_auroc:.4f}"
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
                f"Best model checkpoint (Val AUROC: {best_val_auroc:.4f}) "
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
            best_val_auroc=best_val_auroc,
        )

        if interrupted:
            raise KeyboardInterrupt

        # Return the best validation metric
        return best_val_auroc
    finally:
        finalize_wandb_run(wandb_module, wandb_run, best_val_auroc, interrupted)


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.

    Loads the configuration and passes it to the main training function.

    Args:
        cfg: The Hydra configuration object automatically populated.
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