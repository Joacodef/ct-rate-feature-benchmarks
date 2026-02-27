"""Optimize per-label thresholds on the validation set and evaluate on test sets.

This script executes Phase 1 of the scaling study (Model-Side Bottleneck Checks).
It loads a specific run, extracts raw probabilities from the validation set, 
searches for the threshold that maximizes F1 for each label independently, 
and then re-evaluates the test manifests using these frozen thresholds.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

from common.data.dataset import FeatureDataset
from classification.loops import evaluate_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize thresholds and evaluate.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the training run directory containing .hydra/config.yaml and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="final_model.pt",
        help="Name of the checkpoint file inside the checkpoints/ directory.",
    )
    parser.add_argument(
        "--test-manifests",
        type=str,
        default=None,
        help="Optional comma-separated list of test manifests to evaluate.",
    )
    parser.add_argument(
        "--test-manifest-dir",
        type=str,
        default=None,
        help="Optional override directory for test manifests (e.g., data/manifests/manual).",
    )
    return parser.parse_args()


def build_loader(manifest_path: str, cfg: DictConfig, shuffle: bool = False) -> DataLoader:
    """Builds a dataloader for a specific manifest."""
    dataset_args = {
        "manifest_path": manifest_path,
        "data_root": cfg.paths.data_root,
        "target_labels": list(cfg.training.target_labels),
        "visual_feature_col": cfg.data.columns.visual_feature,
        "text_feature_col": cfg.data.columns.text_feature,
        "preload": bool(OmegaConf.select(cfg, "data.preload_features", default=False)),
    }
    
    try:
        dataset = FeatureDataset(**dataset_args)
    except ValueError as exc:
        if dataset_args.get("text_feature_col") and "Text feature column" in str(exc):
            log.info(f"Manifest {manifest_path} missing text feature column; reloading without text features.")
            dataset_args["text_feature_col"] = None
            dataset = FeatureDataset(**dataset_args)
        else:
            raise

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


@torch.no_grad()
def extract_probabilities(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Runs inference to extract raw probabilities and ground truths."""
    model.eval()
    all_probs = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Extracting Probabilities", leave=False):
        features = batch["visual_features"].to(device)
        labels = batch["labels"].to(device)
        
        preds = model(features)
        probs = torch.sigmoid(preds)
        
        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        
    return np.vstack(all_probs), np.vstack(all_targets)


def optimize_per_class_thresholds(probs: np.ndarray, targets: np.ndarray, target_labels: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Grid searches the optimal threshold to maximize F1 per class."""
    num_classes = probs.shape[1]
    best_thresholds = np.zeros(num_classes)
    best_f1_scores = {}

    # Grid search between 0.01 and 0.99 with 99 steps
    threshold_grid = np.linspace(0.01, 0.99, 99)
    
    for c in range(num_classes):
        c_probs = probs[:, c]
        c_targets = targets[:, c]
        
        best_t = 0.5
        best_f1 = -1.0
        
        for t in threshold_grid:
            preds_bin = (c_probs >= t).astype(int)
            f1 = f1_score(c_targets, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                
        best_thresholds[c] = best_t
        label_name = target_labels[c] if c < len(target_labels) else f"class_{c}"
        best_f1_scores[label_name] = {"threshold": float(best_t), "val_f1": float(best_f1)}
        
    log.info(f"Mean validation F1 with optimized thresholds: {np.mean([x['val_f1'] for x in best_f1_scores.values()]):.4f}")
    return best_thresholds, best_f1_scores


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint_name

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # 1. Load Setup
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_labels = list(cfg.training.target_labels)
    negative_class_name = OmegaConf.select(cfg, "evaluation.negative_class_name", default="No pathology")

    model = hydra.utils.instantiate(
        cfg.model.params,
        _target_=cfg.model._target_,
        out_features=len(target_labels),
        _recursive_=False,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    criterion = hydra.utils.instantiate(cfg.training.loss)
    
    auto_split_enabled = bool(OmegaConf.select(cfg, "data.auto_split.enabled", default=False))
    if auto_split_enabled:
        train_manifest_path = os.path.normpath(os.path.join(cfg.paths.manifest_dir, cfg.data.train_manifest))
        
        # Load the base training dataset to perform the split
        dataset_args = {
            "manifest_path": train_manifest_path,
            "data_root": cfg.paths.data_root,
            "target_labels": target_labels,
            "visual_feature_col": cfg.data.columns.visual_feature,
            "text_feature_col": cfg.data.columns.text_feature,
            "preload": bool(OmegaConf.select(cfg, "data.preload_features", default=False)),
        }
        
        try:
            dataset_train = FeatureDataset(**dataset_args)
        except ValueError as exc:
            if dataset_args.get("text_feature_col") and "Text feature column" in str(exc):
                log.info(f"Manifest {train_manifest_path} missing text feature column; reloading without text features.")
                dataset_args["text_feature_col"] = None
                dataset_train = FeatureDataset(**dataset_args)
            else:
                raise
        
        val_fraction = float(OmegaConf.select(cfg, "data.auto_split.val_fraction", default=0.1))
        stratify_enabled = bool(OmegaConf.select(cfg, "data.auto_split.stratify", default=True))
        split_seed = OmegaConf.select(cfg, "data.auto_split.seed", default=None)
        split_seed = cfg.utils.seed if split_seed is None else int(split_seed)
        
        manifest_df = dataset_train.manifest
        group_series = manifest_df["volumename"].astype(str)
        group_keys = group_series.str.rsplit("_", n=1).str[0]
        
        label_frame = manifest_df[target_labels].fillna(0).astype(int)
        group_label_frame = (
            pd.concat([group_keys.rename("group_key"), label_frame], axis=1)
            .groupby("group_key", sort=False)
            .max()
        )
        
        stratify_labels = None
        if stratify_enabled:
            if len(target_labels) == 1:
                stratify_labels = group_label_frame.iloc[:, 0]
            else:
                stratify_labels = group_label_frame.astype(str).agg("|".join, axis=1)
                
        try:
            _, val_groups = train_test_split(
                group_label_frame.index.values,
                test_size=val_fraction,
                random_state=split_seed,
                shuffle=True,
                stratify=stratify_labels,
            )
        except ValueError:
            _, val_groups = train_test_split(
                group_label_frame.index.values,
                test_size=val_fraction,
                random_state=split_seed,
                shuffle=True,
                stratify=None,
            )
            
        val_mask = group_keys.isin(val_groups)
        val_idx = manifest_df.index[val_mask].values
        
        dataset_val = Subset(dataset_train, val_idx)
        val_loader = DataLoader(
            dataset_val,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        log.info(f"Reconstructed auto-split validation set with {len(val_idx)} samples.")
    else:
        val_manifest = os.path.normpath(os.path.join(cfg.paths.manifest_dir, cfg.data.val_manifest))
        val_loader = build_loader(val_manifest, cfg)
        log.info(f"Loaded validation data from manifest: {val_manifest}")

    # 2. Extract Validation Probabilities and Optimize Thresholds
    log.info("Extracting validation probabilities...")
    val_probs, val_targets = extract_probabilities(model, val_loader, device)
    
    log.info("Optimizing per-class thresholds for F1...")
    best_thresholds, threshold_report = optimize_per_class_thresholds(val_probs, val_targets, target_labels)

    # 3. Evaluate on Test Sets with Frozen Thresholds
    test_manifest_names = args.test_manifests.split(",") if args.test_manifests else list(cfg.data.test_manifests)
    test_manifest_dir = args.test_manifest_dir if args.test_manifest_dir else cfg.paths.manifest_dir
    
    test_results = {}
    for manifest_name in test_manifest_names:
        manifest_name = manifest_name.strip()
        manifest_path = os.path.normpath(os.path.join(test_manifest_dir, manifest_name))
        log.info(f"Evaluating test set: {manifest_name} with frozen thresholds...")
        
        test_loader = build_loader(manifest_path, cfg)
        test_loss, test_metrics = evaluate_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            label_names=target_labels,
            negative_class_name=negative_class_name,
            thresholds=best_thresholds # Inject the custom thresholds!
        )
        
        test_results[manifest_name] = {
            "loss": float(test_loss),
            "auprc": float(test_metrics.get("auprc", 0.0)),
            "auroc": float(test_metrics.get("auroc", 0.0)),
            "f1_macro_optimized": float(test_metrics.get("f1_macro", 0.0)),
            "per_class": test_metrics.get("per_class", {})
        }
        log.info(f"Test {manifest_name} | Optimized F1-Macro: {test_metrics.get('f1_macro', 0.0):.4f}")

    # 4. Save Output Report
    output_dir = run_dir / "evaluation_optimized"
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / "optimized_thresholds_report.json"
    
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint_name,
        "optimized_thresholds": threshold_report,
        "test_results_with_optimized_thresholds": test_results
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    log.info(f"Optimization complete. Report saved to {report_path}")


if __name__ == "__main__":
    main()