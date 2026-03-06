"""Generate K-fold budget manifests for Phase 3 manual-label evaluation.

This script builds fold-specific hold-out test manifests and per-fold budgeted
train/val manifests so training can follow:
- split full manual dataset into K folds,
- sample budget subsets from each fold's training pool,
- evaluate each model on that fold's hold-out set.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from omegaconf import ListConfig, OmegaConf
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ManifestContext:
    manifest_dir: Path
    full_manifest_name: str
    target_labels: List[str]


def _parse_int_list(raw: str, *, name: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"{name} values must be > 0. Got: {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one positive integer.")
    return values


def _normalize_config_name(config_name: Optional[str]) -> Optional[str]:
    if not config_name:
        return None
    raw = config_name.strip().strip("\"'")
    return Path(raw.replace("\\", "/")).name


def _resolve_from_config(config_name: str, full_manifest_name: Optional[str]) -> ManifestContext:
    cfg_candidates = [Path(config_name), Path("configs") / config_name]
    cfg_path = None
    for candidate in cfg_candidates:
        if candidate.exists():
            cfg_path = candidate
            break
    if cfg_path is None:
        raise FileNotFoundError(
            f"Could not find config '{config_name}'. Looked in current dir and configs/."
        )

    cfg = OmegaConf.load(cfg_path)
    manifest_dir = OmegaConf.select(cfg, "paths.manifest_dir", default=None)
    train_manifest = OmegaConf.select(cfg, "data.train_manifest", default=None)
    labels = OmegaConf.select(cfg, "training.target_labels", default=[])

    if manifest_dir is None:
        raise ValueError("Config must define paths.manifest_dir.")

    if isinstance(labels, ListConfig):
        labels = list(labels)
    elif not isinstance(labels, list):
        labels = []

    selected_manifest = full_manifest_name or train_manifest
    if not selected_manifest:
        raise ValueError(
            "Unable to resolve full manifest. Provide --full-manifest-name or set data.train_manifest in config."
        )

    return ManifestContext(
        manifest_dir=Path(str(manifest_dir)),
        full_manifest_name=str(selected_manifest),
        target_labels=[str(label) for label in labels],
    )


def _resolve_explicit(
    full_manifest_path: str,
    target_labels: Optional[str],
) -> ManifestContext:
    manifest_path = Path(full_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Full manifest not found: {manifest_path}")

    labels: List[str] = []
    if target_labels:
        labels = [token.strip() for token in target_labels.split(",") if token.strip()]

    return ManifestContext(
        manifest_dir=manifest_path.parent,
        full_manifest_name=manifest_path.name,
        target_labels=labels,
    )


def _binary_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return (frame[column].fillna(0).astype(float) > 0.0).astype(int)


def _build_group_keys(frame: pd.DataFrame, group_column: str, separator: str, remove_last: bool) -> pd.Series:
    if group_column not in frame.columns:
        raise ValueError(f"Group column '{group_column}' not found in manifest.")

    values = frame[group_column].astype(str)
    if not remove_last:
        return values
    return values.str.rsplit(separator, n=1).str[0]


def _group_label_frame(
    frame: pd.DataFrame,
    group_keys: pd.Series,
    labels: Sequence[str],
) -> pd.DataFrame:
    if not labels:
        unique_groups = pd.Index(group_keys.unique(), name="group_key")
        return pd.DataFrame(index=unique_groups)

    payload = pd.concat([group_keys.rename("group_key"), frame[list(labels)]], axis=1)
    for label in labels:
        payload[label] = _binary_series(payload, label)

    return payload.groupby("group_key", sort=False).max()


def _multilabel_grouped_kfold(
    group_label_df: pd.DataFrame,
    k_folds: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    if group_label_df.shape[0] < k_folds:
        raise ValueError(
            f"Cannot create {k_folds} folds with only {group_label_df.shape[0]} groups."
        )

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError as exc:
        raise ImportError(
            "K-fold generation requires iterative-stratification. Install with: "
            "pip install iterative-stratification"
        ) from exc

    x_dummy = [[0.0] for _ in range(group_label_df.shape[0])]
    if group_label_df.shape[1] > 0:
        y = group_label_df.to_numpy(dtype=int)
    else:
        y = [[0] for _ in range(group_label_df.shape[0])]

    splitter = MultilabelStratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    group_names = group_label_df.index.to_numpy()
    folds: List[Tuple[List[str], List[str]]] = []
    for train_idx, test_idx in splitter.split(x_dummy, y):
        train_groups = [str(group_names[idx]) for idx in train_idx]
        test_groups = [str(group_names[idx]) for idx in test_idx]
        folds.append((train_groups, test_groups))

    return folds


def _split_sampled_train_val(
    sampled_df: pd.DataFrame,
    labels: Sequence[str],
    group_column: str,
    group_separator: str,
    group_remove_last: bool,
    val_fraction: float,
    seed: int,
    val_stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(sampled_df) < 2:
        raise ValueError("Need at least 2 sampled rows to create train/val manifests.")

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    group_keys = _build_group_keys(
        sampled_df,
        group_column=group_column,
        separator=group_separator,
        remove_last=group_remove_last,
    )
    grouped = _group_label_frame(sampled_df, group_keys, labels)

    if grouped.shape[0] < 2:
        raise ValueError("Need at least 2 groups in sampled subset to create train/val splits.")

    # Calculate initial validation size based on the specified fraction.
    base_test_size = int(round(grouped.shape[0] * val_fraction))
    
    # Establish a minimum validation size equivalent to the number of target labels
    # to support optimal multi-label stratification representation.
    min_required_test_size = max(1, len(labels))
    
    # Constrain the maximum validation size to 40% of total available groups
    # to prevent severe depletion of the primary training partition.
    max_allowed_test_size = max(1, int(grouped.shape[0] * 0.4))
    
    test_size = max(base_test_size, min_required_test_size)
    test_size = min(test_size, max_allowed_test_size, grouped.shape[0] - 1)

    if val_stratify and grouped.shape[1] > 0:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        except ImportError as exc:
            raise ImportError(
                "Validation stratification requires iterative-stratification. Install with: "
                "pip install iterative-stratification"
            ) from exc

        x_dummy = [[0.0] for _ in range(grouped.shape[0])]
        y = grouped.to_numpy(dtype=int)

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )

        group_names = grouped.index.to_numpy()
        
        try:
            train_idx, val_idx = next(splitter.split(x_dummy, y))
            train_groups = [str(group_names[idx]) for idx in train_idx]
            val_groups = [str(group_names[idx]) for idx in val_idx]
        except ValueError:
            train_groups, val_groups = train_test_split(
                grouped.index.to_list(),
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=None,
            )
    else:
        train_groups, val_groups = train_test_split(
            grouped.index.to_list(),
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    train_mask = group_keys.isin(train_groups)
    val_mask = group_keys.isin(val_groups)

    train_df = sampled_df.loc[train_mask].copy().sort_index()
    val_df = sampled_df.loc[val_mask].copy().sort_index()

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("Generated an empty train or val split. Increase budget or adjust val_fraction.")

    return train_df, val_df


def _validate_labels(frame: pd.DataFrame, labels: Iterable[str]) -> None:
    label_list = [label for label in labels if label]
    if not label_list:
        return
    missing = [label for label in label_list if label not in frame.columns]
    if missing:
        raise ValueError(f"Missing target labels in full manifest: {missing}")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_index(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "seed",
        "requested_budget",
        "effective_budget",
        "fold_train_pool_rows",
        "fold_test_rows",
        "train_rows",
        "val_rows",
        "test_rows",
        "train_manifest",
        "val_manifest",
        "test_manifest",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fold-specific budget train/val/test manifests for manual-label K-fold CV."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hydra config name/path. Resolves paths.manifest_dir and training.target_labels.",
    )
    source.add_argument(
        "--full-manifest-path",
        type=str,
        default=None,
        help="Explicit path to full manual manifest CSV.",
    )

    parser.add_argument(
        "--full-manifest-name",
        type=str,
        default=None,
        help="When using --config-name, override full manifest filename inside paths.manifest_dir.",
    )
    parser.add_argument(
        "--target-labels",
        type=str,
        default=None,
        help="Comma-separated labels for explicit mode.",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        required=True,
        help="Comma-separated budget ladder (e.g., 20,50,100,250,500,800,1191).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=52,
        help="Deterministic seed used for fold assignment and budget sampling.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction used inside each sampled budget subset.",
    )
    parser.add_argument(
        "--val-stratify",
        action="store_true",
        default=False,
        help="Enable label stratification when creating train/val inside each sampled subset.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="manual_kfold_budget_splits",
        help="Output subdirectory inside manifest_dir.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="manual_kfold",
        help="Filename prefix for generated manifests.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="volumename",
        help="Column used for grouped splitting (prevents leakage across reconstructions).",
    )
    parser.add_argument(
        "--group-separator",
        type=str,
        default="_",
        help="Separator used when deriving grouped IDs.",
    )
    parser.add_argument(
        "--group-remove-last",
        action="store_true",
        default=True,
        help="Drop final token when deriving grouped IDs (default: true).",
    )
    parser.add_argument(
        "--group-keep-last",
        action="store_false",
        dest="group_remove_last",
        help="Keep full group IDs without dropping the final token.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_name = _normalize_config_name(args.config_name)
    if config_name:
        context = _resolve_from_config(config_name, args.full_manifest_name)
    else:
        if not args.full_manifest_path:
            raise ValueError("--full-manifest-path is required in explicit mode.")
        context = _resolve_explicit(args.full_manifest_path, args.target_labels)

    if args.k_folds < 2:
        raise ValueError("--k-folds must be >= 2.")

    budgets = sorted(set(_parse_int_list(args.budgets, name="budgets")))

    full_manifest_path = context.manifest_dir / context.full_manifest_name
    if not full_manifest_path.exists():
        raise FileNotFoundError(f"Full manifest not found: {full_manifest_path}")

    full_df = pd.read_csv(full_manifest_path)
    _validate_labels(full_df, context.target_labels)

    group_keys = _build_group_keys(
        full_df,
        group_column=args.group_column,
        separator=args.group_separator,
        remove_last=bool(args.group_remove_last),
    )
    grouped_labels = _group_label_frame(full_df, group_keys, context.target_labels)

    folds = _multilabel_grouped_kfold(grouped_labels, k_folds=args.k_folds, seed=args.seed)

    output_dir = context.manifest_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[dict] = []

    for fold_idx, (train_groups, test_groups) in enumerate(folds, start=1):
        fold_test_mask = group_keys.isin(test_groups)
        fold_train_mask = group_keys.isin(train_groups)

        fold_train_pool = full_df.loc[fold_train_mask].copy().sort_index()
        fold_test_df = full_df.loc[fold_test_mask].copy().sort_index()

        if len(fold_train_pool) == 0 or len(fold_test_df) == 0:
            raise RuntimeError(
                f"Fold {fold_idx} produced an empty train or test partition."
            )

        test_manifest_name = f"{args.prefix}_f{fold_idx}_test.csv"
        _write_csv(output_dir / test_manifest_name, fold_test_df)

        for budget in budgets:
            effective_budget = min(int(budget), len(fold_train_pool))
            if effective_budget < budget:
                log.warning(
                    "Fold %d budget %d exceeds train-pool size %d; capping to %d.",
                    fold_idx,
                    budget,
                    len(fold_train_pool),
                    effective_budget,
                )

            sampled_df = (
                fold_train_pool.sample(
                    n=effective_budget,
                    replace=False,
                    random_state=args.seed + (fold_idx * 1000) + int(budget),
                )
                .copy()
                .sort_index()
            )

            train_df, val_df = _split_sampled_train_val(
                sampled_df,
                labels=context.target_labels,
                group_column=args.group_column,
                group_separator=args.group_separator,
                group_remove_last=bool(args.group_remove_last),
                val_fraction=float(args.val_fraction),
                seed=args.seed + fold_idx,
                val_stratify=bool(args.val_stratify),
            )

            train_manifest_name = f"{args.prefix}_f{fold_idx}_n{budget}_s{args.seed}.csv"
            val_manifest_name = f"{args.prefix}_f{fold_idx}_n{budget}_s{args.seed}_val.csv"

            _write_csv(output_dir / train_manifest_name, train_df)
            _write_csv(output_dir / val_manifest_name, val_df)

            index_rows.append(
                {
                    "fold": fold_idx,
                    "seed": args.seed,
                    "requested_budget": int(budget),
                    "effective_budget": int(effective_budget),
                    "fold_train_pool_rows": int(len(fold_train_pool)),
                    "fold_test_rows": int(len(fold_test_df)),
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "test_rows": int(len(fold_test_df)),
                    "train_manifest": train_manifest_name,
                    "val_manifest": val_manifest_name,
                    "test_manifest": test_manifest_name,
                }
            )

            log.info(
                "Fold %d | budget=%d (effective=%d) -> train=%d val=%d test=%d",
                fold_idx,
                budget,
                effective_budget,
                len(train_df),
                len(val_df),
                len(fold_test_df),
            )

    index_path = output_dir / "manifest_index.csv"
    _write_index(index_path, index_rows)

    log.info("Generated %d fold-budget manifest rows.", len(index_rows))
    log.info("Output directory: %s", output_dir)
    log.info("Index file: %s", index_path)


if __name__ == "__main__":
    main()
