import json
import logging
import os
import random
import copy
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import data  
from . import models 
from .data.dataset import FeatureDataset

# Initialize a logger for this module
log = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across random,
    numpy, and torch libraries.

    Args:
        seed: The integer value to use as the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Calculates the AUROC score for multi-label classification.
    Converts logits to probabilities using sigmoid.

    Args:
        preds: Raw logits from the model of shape (B, C) or (N, C).
        targets: Ground truth labels of shape (B, C) or (N, C).

    Returns:
        A dictionary containing the computed AUROC score.
    """
    # Detach tensors, move to CPU, and convert to numpy
    # Apply sigmoid to logits to get probabilities
    preds_prob = torch.sigmoid(preds).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    try:
        # Calculate AUROC
        # 'macro' average calculates the metric for each label,
        # then finds their unweighted mean.
        auroc = roc_auc_score(targets_np, preds_prob, average="macro")
    except ValueError as e:
        # Handle cases where a mini-batch or dataset split might
        # not have positive samples for all classes.
        log.warning(f"Could not compute AUROC (likely due to missing labels): {e}")
        auroc = 0.0

    return {"auroc": auroc}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Runs a single training epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: The DataLoader for the training set.
        optimizer: The optimizer to use for updating weights.
        criterion: The loss function.
        device: The device (cuda or cpu) to run training on.

    Returns:
        The average training loss for the epoch.
    """
    # Set the model to training mode
    model.train()
    
    total_loss = 0.0
    
    # Iterate over the training data
    # Wrap dataloader with tqdm to show a progress bar per batch
    train_iter = tqdm(dataloader, desc="Train", unit="batch")
    for batch in train_iter:
        # 1. Get data and move to device
        # We only use visual features for this model
        features = batch["visual_features"].to(device)
        labels = batch["labels"].to(device)
        
        # 2. Zero the gradients
        optimizer.zero_grad()
        
        # 3. Forward pass: compute predicted outputs
        preds = model(features)
        
        # 4. Calculate the batch loss
        loss = criterion(preds, labels)
        
        # 5. Backward pass: compute gradient of the loss
        loss.backward()
        
        # 6. Perform a single optimization step
        optimizer.step()
        
        # 7. Update total loss
        total_loss += loss.item()
        # Update progress bar with current loss
        try:
            train_iter.set_postfix(train_loss=f"{loss.item():.4f}")
        except Exception:
            # If tqdm can't be updated for any reason, ignore
            pass

    # Calculate and return the average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()  # Disables gradient calculation
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Runs a single evaluation epoch.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: The DataLoader for the validation or test set.
        criterion: The loss function.
        device: The device (cuda or cpu) to run evaluation on.

    Returns:
        A tuple containing:
        - The average loss for the epoch.
        - A dictionary of computed metrics (e.g., {"auroc": 0.85}).
    """
    # Set the model to evaluation mode
    model.eval()
    
    total_loss = 0.0
    
    # Lists to store all predictions and labels for metric calculation
    all_preds = []
    all_labels = []

    # Iterate over the data
    # Use tqdm for evaluation progress as well
    eval_iter = tqdm(dataloader, desc="Eval", unit="batch")
    for batch in eval_iter:
        # 1. Get data and move to device
        features = batch["visual_features"].to(device)
        labels = batch["labels"].to(device)
        
        # 2. Forward pass: compute predicted outputs
        preds = model(features)
        
        # 3. Calculate the batch loss
        loss = criterion(preds, labels)
        
        # 4. Update total loss
        total_loss += loss.item()
        
        # 5. Store predictions and labels for metrics
        all_preds.append(preds)
        all_labels.append(labels)
        # Update progress bar with current eval loss
        try:
            eval_iter.set_postfix(eval_loss=f"{loss.item():.4f}")
        except Exception:
            pass

    # Calculate the average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    
    # --- Calculate Metrics ---
    # Concatenate all batches into single tensors
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    # Compute the metrics
    metrics = compute_metrics(all_preds_tensor, all_labels_tensor)

    return avg_loss, metrics


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
    dataset_args = {
        "data_root": cfg.paths.data_root,
        "target_labels": list(cfg.training.target_labels),
        "visual_feature_col": cfg.data.columns.visual_feature,
        # We are only doing visual, so text_feature_col is None
        "text_feature_col": cfg.data.columns.text_feature,
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
    dataset_train = FeatureDataset(manifest_path=train_manifest_path, **dataset_args)
    dataloader_train = DataLoader(dataset_train, shuffle=True, **loader_args)
    log.info(f"Loaded training data from: {train_manifest_path}")

    # Instantiate Validation Loader
    val_manifest_path = os.path.normpath(
        os.path.join(cfg.paths.manifest_dir, cfg.data.val_manifest)
    )
    dataset_val = FeatureDataset(manifest_path=val_manifest_path, **dataset_args)
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

    # --- 5. Initialize Optimizer and Loss ---
    log.info(f"Instantiating optimizer (LR: {cfg.training.learning_rate})")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.learning_rate
    )

    log.info("Instantiating loss function.")
    criterion = hydra.utils.instantiate(cfg.training.loss)

    # --- 6. Set Up Logging ---
    # TODO: Initialize W&B or TensorBoard logger based on 'cfg.logging'
    log.info(f"Using logger: {cfg.logging.logger_name} (placeholder)")

    # --- 7. Run Training Loop ---
    log.info(f"Starting training for {cfg.training.max_epochs} epochs.")

    # Early stopping parameters
    patience = OmegaConf.select(
        cfg, "training.early_stopping_patience", default=10
    )
    log.info(f"Early stopping patience set to {patience} epochs.")
    epochs_no_improve = 0
    best_val_auroc = 0.0
    best_model_state = None
    
    for epoch in range(1, cfg.training.max_epochs + 1):
        
        # Run one epoch of training
        train_loss = train_epoch(
            model, dataloader_train, optimizer, criterion, device
        )
        
        # Run one epoch of validation
        val_loss, val_metrics = evaluate_epoch(
            model, dataloader_val, criterion, device
        )
        
        val_auroc = val_metrics["auroc"]
        
        log.info(
            f"Epoch {epoch}/{cfg.training.max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f}"
        )

        # --- Early Stopping Check ---
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_no_improve = 0
            # Store the state of the best performing model
            best_model_state = copy.deepcopy(model.state_dict())
            log.debug(f"New best val AUROC: {best_val_auroc:.4f}. Resetting patience.")
        else:
            epochs_no_improve += 1
            log.debug(f"Val AUROC did not improve. Patience: {epochs_no_improve}/{patience}")

        # Check if patience has been exceeded
        if epochs_no_improve >= patience:
            log.info(
                f"Early stopping triggered after {patience} epochs "
                f"without improvement. Best Val AUROC: {best_val_auroc:.4f}"
            )
            break

    log.info("Training complete.")

    # --- 8. Save Artifacts ---
    try:
        checkpoint_dir = os.path.normpath(cfg.paths.checkpoint_dir)
    except (InterpolationKeyError, AttributeError):
        job_name = OmegaConf.select(cfg, "hydra.job.name", default="manual_run")
        checkpoint_dir = os.path.normpath(os.path.join("outputs", job_name, "checkpoints"))
        log.warning(
            "hydra.job.name not available; falling back to checkpoint directory: %s",
            checkpoint_dir,
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")

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

    # --- 9. Update Latest Run Pointer ---
    pointer_path_str = OmegaConf.select(cfg, "paths.latest_run_pointer")
    if pointer_path_str:
        try:
            original_cwd = OmegaConf.select(cfg, "hydra.runtime.cwd", default=os.getcwd())
            pointer_path = Path(original_cwd) / Path(os.path.normpath(pointer_path_str))
            pointer_path.parent.mkdir(parents=True, exist_ok=True)

            run_dir_cfg = OmegaConf.select(cfg, "paths.run_dir", default=None)
            if run_dir_cfg:
                run_dir = Path(os.path.normpath(run_dir_cfg))
                if not run_dir.is_absolute():
                    run_dir = Path(original_cwd) / run_dir
            else:
                try:
                    run_dir = Path(HydraConfig.get().runtime.output_dir)
                except Exception:
                    run_dir = Path(checkpoint_dir).parent
            run_dir = run_dir.resolve()
            payload = {
                "run_dir": str(run_dir),
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "best_val_auroc": float(best_val_auroc),
            }

            pointer_path.write_text(json.dumps(payload, indent=2))
            log.info("Latest run pointer updated: %s", pointer_path)
        except Exception as exc:
            log.warning(
                "Unable to update latest run pointer at %s: %s",
                pointer_path_str,
                exc,
            )

    # Return the best validation metric
    return best_val_auroc


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
                "python -m ct_rate_benchmarks.train hydra.job.name=your_job_name"
            )

        train_model(cfg)
    except Exception as e:
        log.exception(f"An error occurred during training: {e}")
        # Optionally re-raise or exit
        raise


if __name__ == "__main__":
    main()