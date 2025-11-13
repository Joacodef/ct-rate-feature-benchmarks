# scripts/analyze_alignment.py

"""
This script analyzes the alignment between pre-computed visual and text
features loaded via a manifest. It is designed to verify the core
assumption of contrastive learning: that paired image-text features
are closer in the shared latent space than unpaired features.

It computes retrieval metrics (Recall@1) and similarity statistics.
"""

import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add the project root to the path to allow importing from 'common'
# This is necessary because this script is in a subdirectory.
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from common.data.dataset import FeatureDataset  # noqa: E402
from common.utils import set_seed  # noqa: E402

# Configure a logger for this script
log = logging.getLogger(__name__)

# Define the K values for recall@k as a single source of truth
TOP_K_VALUES = [1, 5, 10, 50]

@torch.no_grad()
def calculate_metrics(similarity_matrix: torch.Tensor) -> dict:
    """
    Calculates retrieval metrics from a similarity matrix.

    Args:
        similarity_matrix: A (N, N) tensor where S[i, j] is the
            cosine similarity between visual feature i and text feature j.

    Returns:
        A dictionary containing computed metrics.
    """
    num_samples = similarity_matrix.shape[0]
    if num_samples == 0:
        return {}

    # Create a tensor of correct indices [0, 1, 2, ..., N-1]
    # This represents the "ground truth" for paired samples.
    correct_indices = torch.arange(
        num_samples,
        device=similarity_matrix.device
    )

    # --- Mean Similarities ---
    
    # 1. Mean Paired Similarity (Diagonal)
    # We expect this value to be high.
    mean_paired_similarity = similarity_matrix.diag().mean().item()

    # 2. Mean Unpaired Similarity (Off-Diagonal)
    # We expect this value to be low.
    # Create a mask to select only off-diagonal elements
    off_diag_mask = ~torch.eye(
        num_samples,
        dtype=torch.bool,
        device=similarity_matrix.device
    )
    mean_unpaired_similarity = similarity_matrix[off_diag_mask].mean().item()

    # --- Retrieval Metrics (Recall@K) ---
    metrics = {
        "samples_analyzed": num_samples,
        "mean_paired_similarity": mean_paired_similarity,
        "mean_unpaired_similarity": mean_unpaired_similarity,
    }
    
    max_k = max(TOP_K_VALUES)

    # 3. Visual-to-Text (V->T) Recall@K
    # Get the indices of the top-k most similar text features for each visual query
    # Shape: (N, max_k)
    v_to_t_top_k_indices = torch.topk(
        similarity_matrix, k=max_k, dim=1
    ).indices
    
    # Check if the correct index is present in the top-k predictions
    # Shape: (N, max_k)
    v_to_t_hits = (v_to_t_top_k_indices == correct_indices.unsqueeze(1))

    # 4. Text-to-Visual (T->V) Recall@K
    # Get the indices of the top-k most similar visual features for each text query
    # Shape: (max_k, N)
    t_to_v_top_k_indices = torch.topk(
        similarity_matrix, k=max_k, dim=0
    ).indices

    # Check if the correct index is present in the top-k predictions
    # Shape: (max_k, N)
    t_to_v_hits = (t_to_v_top_k_indices == correct_indices.unsqueeze(0))

    # Calculate metrics for each k
    for k in TOP_K_VALUES:
        # V->T: Check hits within the first k predictions
        v_to_t_recall_at_k = v_to_t_hits[:, :k].any(dim=1).float().mean().item()
        metrics[f"v_to_t_recall_at_{k}"] = v_to_t_recall_at_k

        # T->V: Check hits within the first k predictions
        t_to_v_recall_at_k = t_to_v_hits[:k, :].any(dim=0).float().mean().item()
        metrics[f"t_to_v_recall_at_{k}"] = t_to_v_recall_at_k

    return metrics


def analyze_alignment(cfg: DictConfig) -> None:
    """
    Hydra-configured function to load features and run alignment analysis.
    """
    log.info("Starting alignment analysis...")
    try:
        cfg_repr = OmegaConf.to_yaml(cfg)
    except Exception:
        cfg_repr = str(cfg)
    log.info(f"Full configuration:\n{cfg_repr}")

    # --- 1. Validation ---
    visual_col = OmegaConf.select(cfg, "data.columns.visual_feature")
    text_col = OmegaConf.select(cfg, "data.columns.text_feature")

    if not visual_col or not text_col:
        log.error(
            "Configuration is missing feature columns. This script requires "
            "both 'data.columns.visual_feature' AND "
            "'data.columns.text_feature' to be set in the config."
        )
        log.error(
            "Please run again, overriding the configuration, e.g.:\n"
            "python -m scripts.analyze_alignment data.columns.text_feature=column_name"
        )
        raise ValueError("Missing required feature columns in configuration.")

    log.info(f"Using visual feature column: '{visual_col}'")
    log.info(f"Using text feature column:   '{text_col}'")

    # --- 2. Setup ---
    set_seed(cfg.utils.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- 3. Load Data ---
    # We analyze the training manifest by default.
    # This can be overridden on the command line, e.g.:
    # data.train_manifest=valid.csv
    target_manifest = OmegaConf.select(cfg, "data.train_manifest")
    log.info(f"Target manifest file: {target_manifest}")

    manifest_path = os.path.normpath(
        os.path.join(cfg.paths.manifest_dir, target_manifest)
    )

    dataset_args = {
        "data_root": cfg.paths.data_root,
        "target_labels": [],  # We do not need labels for this analysis
        "visual_feature_col": visual_col,
        "text_feature_col": text_col,
        "preload": bool(OmegaConf.select(cfg, "data.preload_features", default=False)),
    }
    
    loader_args = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.training.num_workers,
        "pin_memory": True,
    }

    try:
        dataset = FeatureDataset(manifest_path=manifest_path, **dataset_args)
    except FileNotFoundError:
        log.error(f"Manifest file not found at: {manifest_path}")
        return
    except ValueError as e:
        log.error(f"Failed to initialize dataset: {e}")
        return
        
    dataloader = DataLoader(dataset, shuffle=False, **loader_args)

    # --- 4. Feature Extraction Loop ---
    all_visual_features = []
    all_text_features = []

    log.info(f"Extracting features from {len(dataset)} samples...")
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                
                if "visual_features" not in batch:
                    raise KeyError(f"Batch missing 'visual_features'. Check config column: {visual_col}")
                if "text_features" not in batch:
                    raise KeyError(f"Batch missing 'text_features'. Check config column: {text_col}")
                
                all_visual_features.append(batch["visual_features"].to(device))
                all_text_features.append(batch["text_features"].to(device))

    except (FileNotFoundError, KeyError) as e:
        log.error(f"Failed during feature loading: {e}")
        log.error("Please ensure all feature files exist and manifest paths are correct.")
        return

    if not all_visual_features or not all_text_features:
        log.warning("No features were loaded. Cannot perform analysis.")
        return

    # Concatenate all batches into two large tensors
    V = torch.cat(all_visual_features, dim=0)
    T = torch.cat(all_text_features, dim=0)
    log.info(f"Visual features tensor shape: {V.shape}")
    log.info(f"Text features tensor shape:   {T.shape}")
    
    if V.shape[0] != T.shape[0]:
        log.error("Mismatch in sample count between visual and text features.")
        return

    # --- 5. Compute Similarity Matrix ---
    log.info("Calculating L2-normalized features...")
    # L2-normalize features to compute cosine similarity via dot product
    V_norm = V / V.norm(dim=1, keepdim=True)
    T_norm = T / T.norm(dim=1, keepdim=True)

    log.info("Computing (N, N) similarity matrix...")
    # Calculate the (N, N) similarity matrix
    # S[i, j] = similarity(Visual_i, Text_j)
    similarity_matrix = V_norm @ T_norm.T
    log.info(f"Similarity matrix shape: {similarity_matrix.shape}")

    # --- 6. Calculate Metrics ---
    log.info("Calculating alignment metrics...")
    metrics = calculate_metrics(similarity_matrix)

    # --- 7. Log Results ---
    log.info("--- Alignment Analysis Results ---")
    log.info(f"Samples Analyzed: {metrics.get('samples_analyzed', 0)}")
    log.info(f"Mean Paired Similarity (Diagonal):   {metrics.get('mean_paired_similarity', 0.0):.6f}")
    log.info(f"Mean Unpaired Similarity (Off-Diag): {metrics.get('mean_unpaired_similarity', 0.0):.6f}")
    
    # Define the K values (must match 'calculate_metrics' function)
    log.info("--- Visual-to-Text Retrieval (V->T) ---")
    for k in TOP_K_VALUES:
        metric_name = f"v_to_t_recall_at_{k}"
        metric_name = f"v_to_t_recall_at_{k}"
        metric_value = metrics.get(metric_name, 0.0)
        log.info(f"  Recall@{k:<2}:  {metric_value:.4f}")

    log.info("--- Text-to-Visual Retrieval (T->V) ---")
    for k in TOP_K_VALUES:
        metric_name = f"t_to_v_recall_at_{k}"
        metric_name = f"t_to_v_recall_at_{k}"
        metric_value = metrics.get(metric_name, 0.0)
        log.info(f"  Recall@{k:<2}:  {metric_value:.4f}")
        
    log.info("------------------------------------")


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point for the alignment analysis script.
    """
    try:
        analyze_alignment(cfg)
    except Exception as e:
        log.exception(f"An error occurred during alignment analysis: {e}")
        raise


if __name__ == "__main__":
    main()