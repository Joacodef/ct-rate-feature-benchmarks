# scripts/analyze_alignment.py

"""
Analyze retrieval alignment between visual and text feature embeddings.

The script loads paired features from a manifest, builds one or more
ground-truth masks, and computes retrieval/similarity metrics.

Ground-truth masks:
- Instance-level: inferred from a grouping column after parsing exam IDs.
- Semantic-level: inferred from exact label-vector equality (optional).
"""

import logging
import os
import sys

import hydra
import torch
import numpy as np 
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add project root so imports from `common` resolve when run as a script.
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from common.data.dataset import FeatureDataset  # noqa: E402
from common.utils import set_seed  # noqa: E402

log = logging.getLogger(__name__)

# K values for Recall@K.
TOP_K_VALUES = [1, 5, 10, 50]

# K values for NDCG@K.
NDCG_K_VALUES = [5, 10, 50]

@torch.no_grad()
def calculate_metrics(
    similarity_matrix: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    prefix: str = "",
    include_sample_count: bool = True,
) -> dict:
    """Compute retrieval and similarity metrics from a similarity matrix.

    Args:
        similarity_matrix: ``(N, N)`` tensor where ``S[i, j]`` is similarity
            between visual query ``i`` and text candidate ``j``.
        gt_mask: ``(N, N)`` boolean tensor where ``True`` marks relevant
            visual-text pairs.
        prefix: Optional metric-name prefix (for semantic metrics).
        include_sample_count: Whether to include ``samples_analyzed``.

    Returns:
        Flat metric dictionary containing similarity means, Recall@K, MRR,
        MAP, and NDCG@K for both V->T and T->V retrieval directions.

    Logic:
        Validate matrix shapes, compute paired/unpaired similarity means,
        derive ranked hits via ``topk``, and aggregate retrieval metrics from
        those ranked hit tensors.
    """
    num_samples = similarity_matrix.shape[0]
    if num_samples == 0:
        return {}

    if gt_mask.shape != similarity_matrix.shape:
        raise ValueError(
            f"gt_mask shape ({gt_mask.shape}) must match "
            f"similarity_matrix shape ({similarity_matrix.shape})"
        )

    # Mean similarities: instance-level uses diagonal pairs; semantic-level
    # uses all pairs marked by the provided mask.
    if prefix:
        paired_values = similarity_matrix[gt_mask]
        if paired_values.numel():
            mean_paired_similarity = paired_values.mean().item()
        else:
            mean_paired_similarity = float("nan")

        unpaired_values = similarity_matrix[~gt_mask]
        if unpaired_values.numel():
            mean_unpaired_similarity = unpaired_values.mean().item()
        else:
            mean_unpaired_similarity = float("nan")
    else:
        paired_values = similarity_matrix.diag()
        mean_paired_similarity = paired_values.mean().item()

        # Unpaired values are all non-diagonal elements.
        unpaired_mask = torch.ones_like(
            similarity_matrix, dtype=torch.bool
        ).fill_diagonal_(False)
        
        unpaired_values = similarity_matrix[unpaired_mask]
        if unpaired_values.numel():
            mean_unpaired_similarity = unpaired_values.mean().item()
        else:
            mean_unpaired_similarity = float("nan")

    # Retrieval metrics.
    metrics = {}

    if include_sample_count:
        count_key = f"{prefix}samples_analyzed" if prefix else "samples_analyzed"
        metrics[count_key] = num_samples

    metrics[f"{prefix}mean_paired_similarity"] = mean_paired_similarity
    metrics[f"{prefix}mean_unpaired_similarity"] = mean_unpaired_similarity

    max_k = min(max(TOP_K_VALUES), num_samples)

    # Visual-to-Text (V->T) top-k hits.
    v_to_t_top_k_indices = torch.topk(
        similarity_matrix, k=max_k, dim=1
    ).indices
    v_to_t_hits = gt_mask.gather(dim=1, index=v_to_t_top_k_indices)

    # Text-to-Visual (T->V) top-k hits.
    t_to_v_top_k_indices = torch.topk(
        similarity_matrix, k=max_k, dim=0
    ).indices
    t_to_v_hits = gt_mask.T.gather(dim=1, index=t_to_v_top_k_indices.T)

    # Recall@K.
    for k in TOP_K_VALUES:
        if k > max_k:
            continue
            
        v_to_t_recall_at_k = v_to_t_hits[:, :k].any(dim=1).float().mean().item()
        metrics[f"{prefix}v_to_t_recall_at_{k}"] = v_to_t_recall_at_k

        t_to_v_recall_at_k = t_to_v_hits[:, :k].any(dim=1).float().mean().item()
        metrics[f"{prefix}t_to_v_recall_at_{k}"] = t_to_v_recall_at_k

    # Mean Reciprocal Rank (MRR).
    v_to_t_ranks = torch.where(v_to_t_hits)[1]  # Get column indices of hits
    v_to_t_first_ranks = torch.full((num_samples,), max_k + 1, dtype=torch.long, device=similarity_matrix.device)
    
    # For each query, keep the first relevant rank.
    for i in tqdm(range(num_samples), desc="Computing V->T MRR", leave=False):
        sample_hits = torch.where(v_to_t_hits[i])[0]
        if sample_hits.numel() > 0:
            v_to_t_first_ranks[i] = sample_hits[0]
    
    # Convert ranks to reciprocal ranks.
    v_to_t_reciprocal_ranks = 1.0 / (v_to_t_first_ranks.float() + 1.0)
    v_to_t_reciprocal_ranks[v_to_t_first_ranks > max_k] = 0.0  # No hit found
    v_to_t_mrr = v_to_t_reciprocal_ranks.mean().item()
    metrics[f"{prefix}v_to_t_mrr"] = v_to_t_mrr

    # Symmetric T->V MRR.
    t_to_v_ranks = torch.where(t_to_v_hits)[1]
    t_to_v_first_ranks = torch.full((num_samples,), max_k + 1, dtype=torch.long, device=similarity_matrix.device)
    
    for i in tqdm(range(num_samples), desc="Computing T->V MRR", leave=False):
        sample_hits = torch.where(t_to_v_hits[i])[0]
        if sample_hits.numel() > 0:
            t_to_v_first_ranks[i] = sample_hits[0]
    
    t_to_v_reciprocal_ranks = 1.0 / (t_to_v_first_ranks.float() + 1.0)
    t_to_v_reciprocal_ranks[t_to_v_first_ranks > max_k] = 0.0
    t_to_v_mrr = t_to_v_reciprocal_ranks.mean().item()
    metrics[f"{prefix}t_to_v_mrr"] = t_to_v_mrr

    # Mean Average Precision (MAP).
    v_to_t_ap_list = []
    for i in tqdm(range(num_samples), desc="Computing V->T MAP", leave=False):
        hits_at_i = v_to_t_hits[i].float()
        num_relevant = hits_at_i.sum().item()
        
        if num_relevant == 0:
            v_to_t_ap_list.append(0.0)
            continue
        
        cumsum_hits = torch.cumsum(hits_at_i, dim=0)
        ranks = torch.arange(1, max_k + 1, device=similarity_matrix.device, dtype=torch.float32)
        precisions_at_k = cumsum_hits / ranks
        
        ap = (precisions_at_k * hits_at_i).sum().item() / num_relevant
        v_to_t_ap_list.append(ap)
    
    v_to_t_map = sum(v_to_t_ap_list) / len(v_to_t_ap_list) if v_to_t_ap_list else 0.0
    metrics[f"{prefix}v_to_t_map"] = v_to_t_map

    # Symmetric T->V MAP.
    t_to_v_ap_list = []
    for i in tqdm(range(num_samples), desc="Computing T->V MAP", leave=False):
        hits_at_i = t_to_v_hits[i].float()
        num_relevant = hits_at_i.sum().item()
        
        if num_relevant == 0:
            t_to_v_ap_list.append(0.0)
            continue
        
        cumsum_hits = torch.cumsum(hits_at_i, dim=0)
        ranks = torch.arange(1, max_k + 1, device=similarity_matrix.device, dtype=torch.float32)
        precisions_at_k = cumsum_hits / ranks
        
        ap = (precisions_at_k * hits_at_i).sum().item() / num_relevant
        t_to_v_ap_list.append(ap)
    
    t_to_v_map = sum(t_to_v_ap_list) / len(t_to_v_ap_list) if t_to_v_ap_list else 0.0
    metrics[f"{prefix}t_to_v_map"] = t_to_v_map

    # Normalized Discounted Cumulative Gain (NDCG@K).
    for k in NDCG_K_VALUES:
        if k > max_k:
            continue
        
        # V->T NDCG@K.
        v_to_t_dcg_list = []
        v_to_t_idcg_list = []
        
        for i in tqdm(range(num_samples), desc=f"Computing V->T NDCG@{k}", leave=False):
            hits_at_i = v_to_t_hits[i, :k].float()
            num_relevant = v_to_t_hits[i].sum().item()
            
            ranks = torch.arange(1, k + 1, device=similarity_matrix.device, dtype=torch.float32)
            dcg = (hits_at_i / torch.log2(ranks + 1)).sum().item()
            v_to_t_dcg_list.append(dcg)
            
            # Ideal DCG with all relevant items ranked first.
            num_relevant_at_k = min(num_relevant, k)
            if num_relevant_at_k > 0:
                ideal_hits = torch.zeros(k, device=similarity_matrix.device)
                ideal_hits[:int(num_relevant_at_k)] = 1.0
                idcg = (ideal_hits / torch.log2(ranks + 1)).sum().item()
                v_to_t_idcg_list.append(idcg)
            else:
                v_to_t_idcg_list.append(0.0)
        
        # NDCG = DCG / IDCG.
        v_to_t_ndcg_values = [
            dcg / idcg if idcg > 0 else 0.0
            for dcg, idcg in zip(v_to_t_dcg_list, v_to_t_idcg_list)
        ]
        v_to_t_ndcg = sum(v_to_t_ndcg_values) / len(v_to_t_ndcg_values) if v_to_t_ndcg_values else 0.0
        metrics[f"{prefix}v_to_t_ndcg_at_{k}"] = v_to_t_ndcg
        
        # T->V NDCG@K.
        t_to_v_dcg_list = []
        t_to_v_idcg_list = []
        
        for i in tqdm(range(num_samples), desc=f"Computing T->V NDCG@{k}", leave=False):
            hits_at_i = t_to_v_hits[i, :k].float()
            num_relevant = t_to_v_hits[i].sum().item()
            
            ranks = torch.arange(1, k + 1, device=similarity_matrix.device, dtype=torch.float32)
            dcg = (hits_at_i / torch.log2(ranks + 1)).sum().item()
            t_to_v_dcg_list.append(dcg)
            
            num_relevant_at_k = min(num_relevant, k)
            if num_relevant_at_k > 0:
                ideal_hits = torch.zeros(k, device=similarity_matrix.device)
                ideal_hits[:int(num_relevant_at_k)] = 1.0
                idcg = (ideal_hits / torch.log2(ranks + 1)).sum().item()
                t_to_v_idcg_list.append(idcg)
            else:
                t_to_v_idcg_list.append(0.0)
        
        t_to_v_ndcg_values = [
            dcg / idcg if idcg > 0 else 0.0
            for dcg, idcg in zip(t_to_v_dcg_list, t_to_v_idcg_list)
        ]
        t_to_v_ndcg = sum(t_to_v_ndcg_values) / len(t_to_v_ndcg_values) if t_to_v_ndcg_values else 0.0
        metrics[f"{prefix}t_to_v_ndcg_at_{k}"] = t_to_v_ndcg

    return metrics


def analyze_alignment(cfg: DictConfig) -> None:
    """Run alignment analysis over configured feature manifests.

    Args:
        cfg: Hydra/OmegaConf configuration.

    Returns:
        ``None``.

    Logic:
        Validate feature/label columns, load features and optional labels,
        optionally filter normal-only samples, build similarity and ground-truth
        masks, compute metrics, and log instance/semantic retrieval summaries.
    """
    log.info("Starting alignment analysis...")
    try:
        cfg_repr = OmegaConf.to_yaml(cfg)
    except Exception:
        cfg_repr = str(cfg)
    log.info(f"Full configuration:\n{cfg_repr}")

    # 1) Validate config inputs.
    visual_col = OmegaConf.select(cfg, "data.columns.visual_feature")
    text_col = OmegaConf.select(cfg, "data.columns.text_feature")
    label_cols = OmegaConf.select(cfg, "data.columns.labels", default=[])
    
    # Default grouping column used to infer exam identity.
    grouping_col = OmegaConf.select(
        cfg, "data.columns.grouping_col", default="volumename"
    )
    
    # Optional filter that drops samples with all-zero labels.
    filter_normal_cases = OmegaConf.select(
        cfg, "analysis.filter_normal_cases", default=False
    )

    if not visual_col or not text_col:
        raise ValueError("Missing 'data.columns.visual_feature' or '...text_feature'.")
        
    if label_cols is None:
        label_cols = []
    elif OmegaConf.is_list(label_cols):
        label_cols = list(label_cols)
    elif isinstance(label_cols, (list, tuple)):
        label_cols = list(label_cols)
    elif label_cols:
        label_cols = [str(label_cols)]
    else:
        label_cols = []

    log.info(f"Using visual feature column: '{visual_col}'")
    log.info(f"Using text feature column:   '{text_col}'")
    log.info(f"Using grouping string column (to parse): '{grouping_col}'")
    log.info(f"Filter normal cases (all labels zero): {filter_normal_cases}")
    if label_cols:
        log.info(f"Semantic metrics enabled with {len(label_cols)} label columns.")
    else:
        log.info("Semantic metrics disabled (no label columns provided).")

    # 2) Runtime setup.
    set_seed(cfg.utils.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # 3) Load data.
    target_manifest = OmegaConf.select(cfg, "data.train_manifest")
    log.info(f"Target manifest file: {target_manifest}")

    manifest_path = os.path.normpath(
        os.path.join(cfg.paths.manifest_dir, target_manifest)
    )

    # Load manifest with pandas to access grouping identifiers.
    try:
        log.info(f"Loading manifest with pandas to read: '{grouping_col}'")
        df = pd.read_csv(manifest_path)
        if grouping_col not in df.columns:
            log.error(
                f"Grouping column '{grouping_col}' not found in {manifest_path}. "
                f"Available columns: {df.columns.tolist()}"
            )
            raise ValueError(f"Grouping column '{grouping_col}' not found.")
        
        all_grouping_strings = df[grouping_col].tolist()
        log.info(f"Successfully loaded {len(all_grouping_strings)} string IDs.")

    except FileNotFoundError:
        log.error(f"Manifest file not found at: {manifest_path}")
        return
    except Exception as e:
        log.error(f"Failed to load grouping strings from manifest: {e}")
        return

    # Build dataset/dataloader for feature and optional label loading.
    dataset_args = {
        "data_root": cfg.paths.data_root,
        "target_labels": label_cols,
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

    # 4) Extract features.
    all_visual_features = []
    all_text_features = []
    all_labels = []

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

                if label_cols:
                    if "labels" not in batch:
                        raise KeyError("Batch missing 'labels'. Ensure label columns are configured correctly.")
                    all_labels.append(batch["labels"].to(device))

    except (FileNotFoundError, KeyError) as e:
        log.error(f"Failed during feature loading: {e}")
        log.error("Please ensure all feature files exist and manifest paths are correct.")
        return
    except TypeError as e:
        log.error(f"Failed processing batch. Error: {e}")
        return

    if not all_visual_features or not all_text_features:
        log.warning("No features were loaded. Cannot perform analysis.")
        return

    # Concatenate all batches.
    V = torch.cat(all_visual_features, dim=0)
    T = torch.cat(all_text_features, dim=0)
    
    log.info(f"Visual features tensor shape: {V.shape}")
    log.info(f"Text features tensor shape:   {T.shape}")
    log.info(f"Total grouping strings loaded: {len(all_grouping_strings)}")
    
    if V.shape[0] != T.shape[0] or V.shape[0] != len(all_grouping_strings):
        log.error("Mismatch in sample count between features and grouping strings.")
        return

    labels_tensor = None
    if label_cols:
        if not all_labels:
            log.error("Semantic metrics requested but no labels were loaded.")
            return

        labels_tensor = torch.cat(all_labels, dim=0)
        if labels_tensor.ndim == 1:
            labels_tensor = labels_tensor.unsqueeze(1)

        if labels_tensor.shape[0] != V.shape[0]:
            log.error("Mismatch in sample count between labels and features.")
            return

        log.info(f"Label tensor shape: {labels_tensor.shape}")
    
    # Optional normal-case filtering.
    if filter_normal_cases:
        if labels_tensor is None:
            log.warning(
                "filter_normal_cases=True but no labels loaded. "
                "Cannot filter normal cases. Proceeding with all samples."
            )
        else:
            log.info("Filtering out normal cases (samples with all labels = 0)...")
            # Keep only samples with at least one positive label.
            has_pathology = labels_tensor.any(dim=1)
            num_pathology_cases = has_pathology.sum().item()
            num_normal_cases = (~has_pathology).sum().item()
            
            log.info(f"  Cases with pathology: {num_pathology_cases}")
            log.info(f"  Normal cases (filtered): {num_normal_cases}")
            
            if num_pathology_cases == 0:
                log.error("No cases with pathology found after filtering. Cannot proceed.")
                return
            
            # Apply the same filter to features, labels, and grouping strings.
            V = V[has_pathology]
            T = T[has_pathology]
            labels_tensor = labels_tensor[has_pathology]
            all_grouping_strings = [s for i, s in enumerate(all_grouping_strings) if has_pathology[i]]
            
            log.info(f"After filtering - Visual features shape: {V.shape}")
            log.info(f"After filtering - Text features shape: {T.shape}")
            log.info(f"After filtering - Grouping strings: {len(all_grouping_strings)}")

    # 5) Compute similarity matrix.
    log.info("Calculating L2-normalized features...")
    V_norm = V / V.norm(dim=1, keepdim=True)
    T_norm = T / T.norm(dim=1, keepdim=True)

    log.info("Computing (N, N) similarity matrix...")
    similarity_matrix = V_norm @ T_norm.T
    log.info(f"Similarity matrix shape: {similarity_matrix.shape}")

    # 6) Build ground-truth masks.
    log.info("Parsing exam IDs from grouping strings (e.g., 'train_1_a_1' -> 'train_1_a')")
    try:
        # Split on the final underscore and keep the shared exam prefix.
        parsed_exam_ids = [s.rsplit('_', 1)[0] for s in all_grouping_strings]
        log.info(f"Successfully parsed {len(parsed_exam_ids)} IDs.")
    except AttributeError as e:
        log.error(f"Failed to parse strings. Are 'labels' strings? Error: {e}")
        log.error(f"First 5 items in list: {all_grouping_strings[:5]}")
        raise

    log.info("Generating instance-level ground truth (GT) mask...")
    
    # Use NumPy broadcasted string comparison for efficient mask creation.
    parsed_ids_np = np.array(parsed_exam_ids)
    
    # [N, 1] == [1, N] -> [N, N]
    gt_mask_np = (parsed_ids_np[:, None] == parsed_ids_np[None, :])
    
    # Move mask to torch tensor on the active device.
    gt_mask = torch.from_numpy(gt_mask_np).to(device)

    log.info(f"GT Mask shape: {gt_mask.shape}")
    log.info(f"Total correct pairs in GT Mask: {gt_mask.sum().item()}")

    semantic_mask = None
    if labels_tensor is not None:
        log.info("Generating semantic-level ground truth (GT) mask from label equality...")
        semantic_mask = torch.all(
            labels_tensor[:, None, :] == labels_tensor[None, :, :],
            dim=-1,
        )
        semantic_mask = semantic_mask.to(device)
        log.info(f"Semantic GT Mask shape: {semantic_mask.shape}")
        log.info(f"Total correct pairs in Semantic GT Mask: {semantic_mask.sum().item()}")

    # 7) Compute metrics.
    log.info("Calculating alignment metrics...")
    metrics = {}
    metrics.update(
        calculate_metrics(
            similarity_matrix,
            gt_mask,
        )
    )

    if semantic_mask is not None:
        metrics.update(
            calculate_metrics(
                similarity_matrix,
                semantic_mask,
                prefix="semantic_",
                include_sample_count=False,
            )
        )

    # 8) Log results.
    log.info("="*70)
    log.info("ALIGNMENT ANALYSIS RESULTS")
    log.info("="*70)
    
    log.info("\n[INSTANCE-LEVEL METRICS]")
    log.info(f"  Samples Analyzed: {metrics.get('samples_analyzed', 0)}")
    log.info(f"  Mean Paired Similarity:   {metrics.get('mean_paired_similarity', float('nan')):.6f}")
    log.info(f"  Mean Unpaired Similarity: {metrics.get('mean_unpaired_similarity', float('nan')):.6f}")

    log.info("\n[VISUAL -> TEXT RETRIEVAL]")
    log.info(f"  MRR:        {metrics.get('v_to_t_mrr', 0.0):.4f}")
    log.info(f"  MAP:        {metrics.get('v_to_t_map', 0.0):.4f}")
    log.info("  Recall@K:")
    for k in TOP_K_VALUES:
        metric_value = metrics.get(f"v_to_t_recall_at_{k}", 0.0)
        log.info(f"    K={k:<2}:  {metric_value:.4f}")
    log.info("  NDCG@K:")
    for k in NDCG_K_VALUES:
        metric_value = metrics.get(f"v_to_t_ndcg_at_{k}", 0.0)
        log.info(f"    K={k:<2}:  {metric_value:.4f}")

    log.info("\n[TEXT -> VISUAL RETRIEVAL]")
    log.info(f"  MRR:        {metrics.get('t_to_v_mrr', 0.0):.4f}")
    log.info(f"  MAP:        {metrics.get('t_to_v_map', 0.0):.4f}")
    log.info("  Recall@K:")
    for k in TOP_K_VALUES:
        metric_value = metrics.get(f"t_to_v_recall_at_{k}", 0.0)
        log.info(f"    K={k:<2}:  {metric_value:.4f}")
    log.info("  NDCG@K:")
    for k in NDCG_K_VALUES:
        metric_value = metrics.get(f"t_to_v_ndcg_at_{k}", 0.0)
        log.info(f"    K={k:<2}:  {metric_value:.4f}")

    if semantic_mask is not None:
        log.info("\n" + "="*70)
        log.info("[SEMANTIC-LEVEL METRICS] (Exact Label Match)")
        log.info("="*70)
        log.info(f"  Mean Paired Similarity:   {metrics.get('semantic_mean_paired_similarity', float('nan')):.6f}")
        log.info(f"  Mean Unpaired Similarity: {metrics.get('semantic_mean_unpaired_similarity', float('nan')):.6f}")

        log.info("\n[SEMANTIC VISUAL -> TEXT RETRIEVAL]")
        log.info(f"  MRR:        {metrics.get('semantic_v_to_t_mrr', 0.0):.4f}")
        log.info(f"  MAP:        {metrics.get('semantic_v_to_t_map', 0.0):.4f}")
        log.info("  Recall@K:")
        for k in TOP_K_VALUES:
            metric_value = metrics.get(f"semantic_v_to_t_recall_at_{k}", float("nan"))
            log.info(f"    K={k:<2}:  {metric_value:.4f}")
        log.info("  NDCG@K:")
        for k in NDCG_K_VALUES:
            metric_value = metrics.get(f"semantic_v_to_t_ndcg_at_{k}", 0.0)
            log.info(f"    K={k:<2}:  {metric_value:.4f}")

        log.info("\n[SEMANTIC TEXT -> VISUAL RETRIEVAL]")
        log.info(f"  MRR:        {metrics.get('semantic_t_to_v_mrr', 0.0):.4f}")
        log.info(f"  MAP:        {metrics.get('semantic_t_to_v_map', 0.0):.4f}")
        log.info("  Recall@K:")
        for k in TOP_K_VALUES:
            metric_value = metrics.get(f"semantic_t_to_v_recall_at_{k}", float("nan"))
            log.info(f"    K={k:<2}:  {metric_value:.4f}")
        log.info("  NDCG@K:")
        for k in NDCG_K_VALUES:
            metric_value = metrics.get(f"semantic_t_to_v_ndcg_at_{k}", 0.0)
            log.info(f"    K={k:<2}:  {metric_value:.4f}")
        
    log.info("\n" + "="*70)


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for alignment analysis runs.

    Args:
        cfg: Hydra/OmegaConf config injected by ``@hydra.main``.

    Returns:
        ``None``.

    Logic:
        Execute ``analyze_alignment`` and log/re-raise failures to preserve
        non-zero exit behavior for callers.
    """
    try:
        analyze_alignment(cfg)
    except Exception as e:
        log.exception(f"An error occurred during alignment analysis: {e}")
        raise


if __name__ == "__main__":
    main()