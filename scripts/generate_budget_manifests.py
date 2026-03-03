"""Generate budgeted train manifests for scaling-law experiments.

This utility creates multiple training manifests at user-specified label budgets
while keeping a fixed validation manifest. It supports reading train/valid
manifest names from a Hydra config or explicit CSV paths.

Typical use case:
- Keep validation fixed for fair model selection.
- Create train subsets at N in {100, 250, 500, 1000, ...}.
- Repeat for multiple seeds.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from omegaconf import ListConfig, OmegaConf
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ManifestSpec:
    manifest_dir: Path
    train_manifest_name: str
    val_manifest_name: str
    target_labels: List[str]
    auto_split_enabled: bool = False



def _normalize_config_name(config_name: Optional[str]) -> Optional[str]:
    """Normalize a Hydra config name, accepting accidental file paths.

    Args:
        config_name: Raw config value passed from CLI.

    Returns:
        Basename suitable for Hydra ``--config-name`` usage, or ``None``.

    Logic:
        Hydra config names are relative to ``configs``. If a full path is
        passed, reduce it to basename so downstream resolution is consistent.
    """
    if not config_name:
        return None
    raw = config_name.strip().strip("\"'")
    return Path(raw.replace("\\", "/")).name



def _parse_int_list(value: str, *, name: str) -> List[int]:
    """Parse comma-separated positive integers from CLI input.

    Args:
        value: Comma-separated integer string.
        name: Argument name for contextual error messages.

    Returns:
        Parsed positive integer list.

    Raises:
        ValueError: If list is empty or any parsed value is non-positive.

    Logic:
        Split by comma, trim whitespace, parse integers, and validate > 0.
    """
    values: List[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parsed = int(chunk)
        if parsed <= 0:
            raise ValueError(f"{name} values must be > 0. Got: {parsed}")
        values.append(parsed)
    if not values:
        raise ValueError(f"{name} must contain at least one integer.")
    return values


def _build_budget_pairs(
    budgets: Sequence[int],
    *,
    budget_type: str,
    auto_split_val_fraction: float,
) -> List[Tuple[int, int]]:
    """Build ``(requested_effective, sampled_manifest_rows)`` budget pairs.

    Args:
        budgets: User-provided integer budgets.
        budget_type: ``manifest_rows`` or ``effective_train_rows``.
        auto_split_val_fraction: Validation fraction used by auto-split.

    Returns:
        List of ``(effective_budget, manifest_budget)`` tuples.

    Raises:
        ValueError: If arguments are invalid.
    """
    if budget_type not in {"manifest_rows", "effective_train_rows"}:
        raise ValueError(f"Unsupported budget_type: {budget_type}")

    pairs: List[Tuple[int, int]] = []
    if budget_type == "manifest_rows":
        for budget in budgets:
            pairs.append((int(budget), int(budget)))
        return pairs

    if not 0.0 < auto_split_val_fraction < 1.0:
        raise ValueError("--auto-split-val-fraction must be between 0 and 1.")

    train_fraction = 1.0 - auto_split_val_fraction
    for effective_budget in budgets:
        manifest_budget = int(math.ceil(float(effective_budget) / train_fraction))
        pairs.append((int(effective_budget), manifest_budget))
    return pairs


def _load_selection_source(
    source_path: Path,
) -> Dict[Tuple[int, int], List[str]]:
    """Load precomputed selected IDs grouped by ``(budget, seed)``.

    Expected CSV columns: ``budget``, ``seed``, ``selection_value``.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Selection source CSV not found: {source_path}")

    frame = pd.read_csv(source_path)
    required = {"budget", "seed", "selection_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Selection source missing columns: {sorted(missing)}")

    selection_map: Dict[Tuple[int, int], List[str]] = {}
    grouped = frame.groupby(["budget", "seed"], sort=False)
    for (budget, seed), group in grouped:
        key = (int(budget), int(seed))
        selection_map[key] = [str(value) for value in group["selection_value"].tolist()]
    return selection_map



def _resolve_from_config(config_name: str) -> ManifestSpec:
    """Resolve manifest inputs from a Hydra config file.

    Args:
        config_name: Hydra config name/path.

    Returns:
        ``ManifestSpec`` with manifest directory, train/val names, labels,
        and ``data.auto_split.enabled`` state.

    Raises:
        FileNotFoundError: If config file cannot be found.
        ValueError: If required config keys are missing.

    Logic:
        Load config, then extract ``paths.manifest_dir``,
        ``data.train_manifest``, optional ``data.val_manifest``, optional
        ``training.target_labels``, and ``data.auto_split.enabled``.
    """
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
    manifest_dir_value = OmegaConf.select(cfg, "paths.manifest_dir", default=None)
    train_name = OmegaConf.select(cfg, "data.train_manifest", default=None)
    val_name = OmegaConf.select(cfg, "data.val_manifest", default=None)
    labels = OmegaConf.select(cfg, "training.target_labels", default=[])
    auto_split_enabled = bool(OmegaConf.select(cfg, "data.auto_split.enabled", default=False))

    if manifest_dir_value is None or train_name is None:
        raise ValueError(
            "Config must define paths.manifest_dir and data.train_manifest."
        )
    if not auto_split_enabled and (val_name is None or str(val_name).strip() == ""):
        raise ValueError(
            "Config must define data.val_manifest when data.auto_split.enabled is false."
        )

    if isinstance(labels, ListConfig):
        labels = list(labels)
    elif not isinstance(labels, list):
        labels = []

    return ManifestSpec(
        manifest_dir=Path(str(manifest_dir_value)),
        train_manifest_name=str(train_name),
        val_manifest_name="" if val_name is None else str(val_name),
        target_labels=[str(label) for label in labels],
        auto_split_enabled=auto_split_enabled,
    )



def _resolve_explicit(
    train_manifest_path: str,
    val_manifest_path: Optional[str],
    target_labels: Optional[str],
    require_val_manifest: bool,
) -> ManifestSpec:
    """Resolve manifest inputs from explicit train/val CSV paths.

    Args:
        train_manifest_path: Path to training manifest CSV.
        val_manifest_path: Optional path to validation manifest CSV.
        target_labels: Optional comma-separated label names.
        require_val_manifest: Whether explicit mode requires an input val CSV.

    Returns:
        ``ManifestSpec`` with explicit paths converted to manifest names.

    Raises:
        FileNotFoundError: If required manifest file(s) do not exist.
        ValueError: If train and val are in different folders when both are used.

    Logic:
        Validate train path always. In fixed-validation mode, also validate
        val path and shared parent directory so one ``paths.manifest_dir`` can
        reference both manifests. In generated-validation mode, val path is optional.
    """
    train_path = Path(train_manifest_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_path}")

    val_manifest_name = ""
    if require_val_manifest:
        if not val_manifest_path:
            raise ValueError(
                "--val-manifest-path is required unless --generate-val-csvs is used."
            )
        val_path = Path(val_manifest_path)
        if not val_path.exists():
            raise FileNotFoundError(f"Validation manifest not found: {val_path}")
        if train_path.parent != val_path.parent:
            raise ValueError(
                "Explicit train/val manifests must be in the same folder so one manifest_dir can reference both."
            )
        val_manifest_name = val_path.name

    labels = []
    if target_labels:
        labels = [token.strip() for token in target_labels.split(",") if token.strip()]

    return ManifestSpec(
        manifest_dir=train_path.parent,
        train_manifest_name=train_path.name,
        val_manifest_name=val_manifest_name,
        target_labels=labels,
        auto_split_enabled=False,
    )



def _build_group_keys(series: pd.Series, separator: str, remove_last: bool) -> pd.Series:
    """Build group keys from an identifier series.

    Args:
        series: Identifier column (typically ``volumename``).
        separator: Token separator used to split identifiers.
        remove_last: Whether to drop the last token.

    Returns:
        Series of group keys used for grouped sampling.

    Logic:
        Optionally collapse reconstruction-level IDs to study-level IDs by
        trimming the final separator-delimited segment.
    """
    values = series.astype(str)
    if remove_last:
        return values.str.rsplit(separator, n=1).str[0]
    return values



def _sample_rows(
    train_df: pd.DataFrame,
    budget: int,
    seed: int,
) -> pd.DataFrame:
    """Sample an exact number of rows from the training manifest.

    Args:
        train_df: Full training manifest dataframe.
        budget: Number of rows to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        Sampled dataframe sorted by original row index.

    Raises:
        ValueError: If requested budget exceeds train size.
    """
    if budget > len(train_df):
        raise ValueError(f"Requested budget {budget} exceeds train size {len(train_df)}.")
    return train_df.sample(n=budget, random_state=seed, replace=False).sort_index()


def _sample_rows_stratified(
    train_df: pd.DataFrame,
    budget: int,
    seed: int,
    target_labels: Sequence[str],
    allow_random_fallback: bool,
) -> Tuple[pd.DataFrame, str]:
    """Sample rows with multilabel stratification using iterative stratification.

    Args:
        train_df: Full training manifest dataframe.
        budget: Number of rows to sample.
        seed: Random seed for reproducible sampling.
        target_labels: Label columns used for stratification.
        allow_random_fallback: Whether to fallback to random sampling on failure.

    Returns:
        Tuple of sampled dataframe and strategy note (`stratified` or fallback note).

    Raises:
        ImportError: If multilabel stratification dependency is missing.
        ValueError: If budget is invalid or labels are missing.

    Logic:
        Use ``MultilabelStratifiedShuffleSplit`` to select ``budget`` rows while
        preserving multilabel prevalence as closely as possible.
    """
    if budget > len(train_df):
        raise ValueError(f"Requested budget {budget} exceeds train size {len(train_df)}.")
    if not target_labels:
        raise ValueError(
            "Stratified sampling requires target labels. Provide training.target_labels in config "
            "or pass --target-labels in explicit mode."
        )

    missing_cols = [label for label in target_labels if label not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Stratified sampling missing label columns: {missing_cols}")

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "Stratified mode requires 'iterative-stratification'. Install with: "
            "pip install iterative-stratification"
        ) from exc

    y = train_df[list(target_labels)].fillna(0).astype(int).to_numpy()
    x_dummy = np.zeros((len(train_df), 1), dtype=np.float32)

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=budget,
        random_state=seed,
    )

    # Multilabel splitter may fail for edge cases (very tiny budget / rare labels).
    try:
        _, selected_idx = next(splitter.split(x_dummy, y))
        selected_idx = np.array(selected_idx, dtype=int)
    except Exception as exc:
        if not allow_random_fallback:
            raise RuntimeError(
                "Multilabel stratification failed and fallback is disabled."
            ) from exc
        log.warning(
            "Stratified sampling failed for budget=%d, seed=%d (%s). Falling back to random rows.",
            budget,
            seed,
            exc,
        )
        fallback = _sample_rows(train_df, budget=budget, seed=seed)
        return fallback, "random_fallback"

    # Enforce exact budget size defensively if splitter returns off-by-one size.
    if len(selected_idx) != budget:
        rng = np.random.default_rng(seed)
        if len(selected_idx) > budget:
            selected_idx = rng.choice(selected_idx, size=budget, replace=False)
        else:
            needed = budget - len(selected_idx)
            all_idx = np.arange(len(train_df))
            mask = np.ones(len(train_df), dtype=bool)
            mask[selected_idx] = False
            extra_pool = all_idx[mask]
            extra_idx = rng.choice(extra_pool, size=needed, replace=False)
            selected_idx = np.concatenate([selected_idx, extra_idx])

    sampled = train_df.iloc[np.array(selected_idx, dtype=int)].sort_index()
    return sampled, "stratified"



def _sample_groups(
    train_df: pd.DataFrame,
    budget: int,
    seed: int,
    group_column: str,
    group_separator: str,
    group_remove_last: bool,
) -> pd.DataFrame:
    """Sample full groups while targeting a requested row budget.

    Args:
        train_df: Full training manifest dataframe.
        budget: Target number of rows.
        seed: Random seed for reproducible group order.
        group_column: Column used to derive group IDs.
        group_separator: Delimiter used to parse grouped IDs.
        group_remove_last: Whether to remove final token from IDs.

    Returns:
        Group-preserving sampled dataframe, sorted by original index.

    Raises:
        ValueError: If grouping column is missing.

    Logic:
        Build group IDs, shuffle groups with seed, and greedily add full
        groups until approaching budget to reduce leakage risk.
    """
    if group_column not in train_df.columns:
        raise ValueError(f"Group column '{group_column}' not found in train manifest.")

    group_keys = _build_group_keys(train_df[group_column], group_separator, group_remove_last)
    with_groups = train_df.assign(_group_key=group_keys)
    group_sizes = with_groups.groupby("_group_key", sort=False).size().sort_values(ascending=False)

    # Randomized greedy fill to keep entire groups while approaching the requested budget.
    shuffled_groups = group_sizes.sample(frac=1.0, random_state=seed)
    selected_groups: List[str] = []
    running = 0
    for key, group_size in shuffled_groups.items():
        if running + int(group_size) <= budget or not selected_groups:
            selected_groups.append(str(key))
            running += int(group_size)

    sampled = with_groups[with_groups["_group_key"].isin(selected_groups)].drop(columns=["_group_key"])

    # If a single selected group is larger than budget, this mode cannot hit budget exactly.
    if len(sampled) > budget and len(selected_groups) == 1:
        log.warning(
            "Group-based sampling selected one group with %d rows for budget=%d. "
            "Keeping full group to avoid leakage.",
            len(sampled),
            budget,
        )

    return sampled.sort_index()


def _binary_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Convert a label column to binary indicator values (0/1)."""
    return (frame[column].fillna(0).astype(float) > 0.0).astype(int)


def _simulate_grouped_auto_split(
    sampled_df: pd.DataFrame,
    *,
    target_labels: Sequence[str],
    val_fraction: float,
    split_seed: int,
    stratify_enabled: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate grouped auto-split behavior from training for a sampled subset."""
    if "volumename" not in sampled_df.columns:
        raise ValueError("Auto-split coverage enforcement requires 'volumename' column.")
    if len(sampled_df) < 2:
        raise ValueError("Need at least 2 rows to simulate train/val split.")

    group_series = sampled_df["volumename"].astype(str)
    group_keys = group_series.str.rsplit("_", n=1).str[0]

    if target_labels:
        label_frame = sampled_df[list(target_labels)].copy()
        for label in target_labels:
            label_frame[label] = _binary_series(sampled_df, label)
        group_label_frame = (
            pd.concat([group_keys.rename("group_key"), label_frame], axis=1)
            .groupby("group_key", sort=False)
            .max()
        )
    else:
        group_label_frame = pd.DataFrame(index=pd.Index(group_keys.unique(), name="group_key"))

    if group_label_frame.shape[0] < 2:
        raise ValueError("Need at least 2 groups to simulate grouped auto-split.")

    stratify_labels = None
    if stratify_enabled and target_labels:
        if len(target_labels) == 1:
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
    except ValueError:
        train_groups, val_groups = train_test_split(
            group_label_frame.index.values,
            test_size=val_fraction,
            random_state=split_seed,
            shuffle=True,
            stratify=None,
        )

    train_mask = group_keys.isin(train_groups)
    val_mask = group_keys.isin(val_groups)

    train_split = sampled_df.loc[train_mask].copy()
    val_split = sampled_df.loc[val_mask].copy()
    return train_split, val_split


def _split_coverage_deficits(
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    *,
    target_labels: Sequence[str],
) -> List[dict]:
    """Return missing-class deficits per label in train/val splits."""
    deficits: List[dict] = []
    for label in target_labels:
        train_values = set(_binary_series(train_split, label).unique().tolist())
        val_values = set(_binary_series(val_split, label).unique().tolist())
        missing_train = sorted(list({0, 1}.difference(train_values)))
        missing_val = sorted(list({0, 1}.difference(val_values)))
        if missing_train:
            deficits.append(
                {
                    "partition": "train",
                    "label": label,
                    "missing_classes": missing_train,
                }
            )
        if missing_val:
            deficits.append(
                {
                    "partition": "val",
                    "label": label,
                    "missing_classes": missing_val,
                }
            )
    return deficits


def _enforce_auto_split_label_coverage(
    sampled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    *,
    target_labels: Sequence[str],
    split_seed: int,
    val_fraction: float,
    stratify_enabled: bool,
    max_extra_rows: int,
) -> Tuple[pd.DataFrame, str]:
    """Augment sampled rows until split-level label coverage is satisfied or exhausted."""
    if not target_labels:
        return sampled_df, "coverage_skipped_no_labels"

    initial_size = len(sampled_df)
    if initial_size == 0:
        return sampled_df, "coverage_skipped_empty_sample"

    current = sampled_df.copy()
    pool = train_df.loc[~train_df.index.isin(current.index)].copy()
    attempts = 0

    while True:
        try:
            train_split, val_split = _simulate_grouped_auto_split(
                current,
                target_labels=target_labels,
                val_fraction=val_fraction,
                split_seed=split_seed,
                stratify_enabled=stratify_enabled,
            )
        except ValueError as exc:
            deficits = [{"partition": "split", "label": "__split__", "missing_classes": [str(exc)]}]
        else:
            deficits = _split_coverage_deficits(
                train_split,
                val_split,
                target_labels=target_labels,
            )

        if not deficits:
            added = len(current) - initial_size
            if added == 0:
                return current, "coverage_ok"
            return current, f"coverage_augmented_{added}"

        if len(pool) == 0:
            log.warning(
                "Coverage enforcement could not satisfy all labels (pool exhausted). Remaining deficits=%s",
                deficits,
            )
            added = len(current) - initial_size
            return current, f"coverage_unresolved_pool_exhausted_{added}"

        if max_extra_rows > 0 and (len(current) - initial_size) >= max_extra_rows:
            log.warning(
                "Coverage enforcement reached max extra rows (%d). Remaining deficits=%s",
                max_extra_rows,
                deficits,
            )
            added = len(current) - initial_size
            return current, f"coverage_unresolved_max_extra_{added}"

        score = pd.Series(0, index=pool.index, dtype=int)
        for deficit in deficits:
            label = deficit.get("label")
            missing_classes = deficit.get("missing_classes", [])
            if label not in pool.columns:
                continue
            label_values = _binary_series(pool, label)
            for missing_class in missing_classes:
                if missing_class in (0, 1):
                    score = score + (label_values == int(missing_class)).astype(int)

        if int(score.max()) <= 0:
            rng = np.random.default_rng(split_seed + attempts)
            chosen_index = int(rng.choice(pool.index.to_numpy(), size=1, replace=False)[0])
            chosen_rows = pool.loc[[chosen_index]]
        else:
            top_score = int(score.max())
            top_candidates = score[score == top_score].index.to_numpy()
            rng = np.random.default_rng(split_seed + attempts)
            chosen_index = int(rng.choice(top_candidates, size=1, replace=False)[0])
            chosen_rows = pool.loc[[chosen_index]]

        current = pd.concat([current, chosen_rows], axis=0).sort_index()
        pool = pool.drop(index=chosen_rows.index)
        attempts += 1



def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a dataframe as CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)



def _write_manifest_index(index_path: Path, rows: Sequence[dict]) -> None:
    """Write a manifest index CSV summarizing generated budget subsets.

    Args:
        index_path: Output path for index CSV.
        rows: One metadata row per generated subset.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "budget",
        "seed",
        "budget_type",
        "requested_effective_train_rows",
        "adjusted_effective_train_rows",
        "sampled_manifest_rows",
        "sample_unit",
        "sampling_strategy",
        "strategy_note",
        "requested_train_rows",
        "actual_train_rows",
        "train_manifest",
        "val_manifest",
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_selection_index(index_path: Path, rows: Sequence[dict]) -> None:
    """Write selected IDs for cross-manifest reproducibility.

    Columns: ``budget``, ``seed``, ``selection_value``.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["budget", "seed", "selection_value"]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _relative_name(path: Path, base_dir: Path) -> str:
    """Return a POSIX-style relative path for Hydra manifest references."""
    relative = os.path.relpath(path.resolve(), start=base_dir.resolve())
    return str(Path(relative)).replace("\\", "/")



def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for budgeted manifest generation.

    Returns:
        Parsed ``argparse.Namespace`` containing source and sampling options.

    Logic:
        Support config-driven and explicit-path modes, plus reproducible
        budget/seed sampling controls and output layout options.
    """
    parser = argparse.ArgumentParser(
        description="Generate budgeted train manifests while keeping validation fixed."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hydra config name/path to resolve paths.manifest_dir, data.train_manifest, and data.val_manifest.",
    )
    source.add_argument(
        "--train-manifest-path",
        type=str,
        default=None,
        help="Explicit train manifest CSV path (requires --val-manifest-path).",
    )

    parser.add_argument(
        "--val-manifest-path",
        type=str,
        default=None,
        help="Explicit validation manifest CSV path when --train-manifest-path is used.",
    )
    parser.add_argument(
        "--target-labels",
        type=str,
        default=None,
        help="Optional comma-separated target labels when using explicit paths (for validation checks only).",
    )
    parser.add_argument(
        "--auto-split-enabled",
        action="store_true",
        help="Deprecated alias for train-only explicit mode. Prefer --generate-val-csvs for fixed-split workflows.",
    )
    parser.add_argument(
        "--generate-val-csvs",
        action="store_true",
        help="Generate both train and val CSVs per (budget,seed) using grouped split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction used when --generate-val-csvs is enabled.",
    )
    parser.add_argument(
        "--val-stratify",
        action="store_true",
        default=False,
        help="Enable label-based stratification in generated train/val split (default: false for cross-source alignment).",
    )
    parser.add_argument(
        "--val-no-stratify",
        action="store_false",
        dest="val_stratify",
        help="Disable label-based stratification in generated train/val split.",
    )

    parser.add_argument(
        "--budgets",
        type=str,
        required=True,
        help="Comma-separated train subset sizes (e.g., 100,250,500,1000).",
    )
    parser.add_argument(
        "--budget-type",
        choices=["manifest_rows", "effective_train_rows"],
        default="manifest_rows",
        help="Interpret --budgets as raw manifest rows or effective train rows after auto-split.",
    )
    parser.add_argument(
        "--auto-split-val-fraction",
        type=float,
        default=0.1,
        help="Deprecated alias for --val-fraction in budget conversion contexts.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated random seeds for repeated subsets (e.g., 42,1234,2025).",
    )
    parser.add_argument(
        "--sample-unit",
        choices=["rows", "groups"],
        default="rows",
        help="Sample individual rows or full groups.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "stratified"],
        default="stratified",
        help="Row sampling strategy (default: stratified). 'stratified' uses multilabel iterative stratification.",
    )
    parser.add_argument(
        "--disable-random-fallback",
        action="store_true",
        help="Fail instead of falling back to random when stratified sampling is infeasible.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="volumename",
        help="Grouping column used when --sample-unit=groups.",
    )
    parser.add_argument(
        "--group-separator",
        type=str,
        default="_",
        help="Separator used to derive base group key when --sample-unit=groups.",
    )
    parser.add_argument(
        "--group-remove-last",
        action="store_true",
        help="If set, derive group key by removing the final segment after separator.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="budget_splits",
        help="Subfolder (inside manifest_dir) where generated manifests are written.",
    )
    parser.add_argument(
        "--copy-validation",
        action="store_true",
        help="Copy fixed validation manifest into output subfolder for self-contained manifest_dir usage.",
    )
    parser.add_argument(
        "--selection-column",
        type=str,
        default="volumename",
        help="Column used to export/import selected IDs for cross-manifest alignment.",
    )
    parser.add_argument(
        "--selection-source-path",
        type=str,
        default=None,
        help="Optional CSV with columns budget,seed,selection_value to force identical selections.",
    )
    parser.add_argument(
        "--selection-export-path",
        type=str,
        default=None,
        help="Optional CSV output path to store selected IDs (budget,seed,selection_value).",
    )
    parser.add_argument(
        "--disable-label-coverage-enforcement",
        action="store_true",
        help="Disable post-sampling augmentation that enforces label class coverage in simulated auto-split train/val partitions.",
    )
    parser.add_argument(
        "--coverage-max-extra-rows",
        type=int,
        default=0,
        help="Maximum extra rows added for coverage repair per split (0 means no limit).",
    )

    return parser.parse_args()



def _validate_labels_presence(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    labels: Iterable[str],
) -> None:
    """Validate that target labels exist in train (and optional val) manifests.

    Args:
        train_df: Training manifest dataframe.
        val_df: Optional validation manifest dataframe.
        labels: Expected target label names.

    Raises:
        ValueError: If one or more labels are missing from required manifests.
    """
    label_list = [label for label in labels if label]
    if not label_list:
        return
    missing_train = [label for label in label_list if label not in train_df.columns]
    missing_val = []
    if val_df is not None:
        missing_val = [label for label in label_list if label not in val_df.columns]
    if missing_train or missing_val:
        raise ValueError(
            "Missing target labels in manifests. "
            f"train_missing={missing_train}, val_missing={missing_val}"
        )



def main() -> None:
    """CLI entrypoint for generating budgeted train manifests.

    Returns:
        ``None``.

    Logic:
        Resolve input manifests, validate labels, generate subsets for each
        ``(seed, budget)`` pair, and write an index for downstream training.
    """
    # 1) Parse and resolve source manifests.
    args = _parse_args()

    config_name = _normalize_config_name(args.config_name)
    if config_name:
        spec = _resolve_from_config(config_name)
    else:
        if not args.train_manifest_path:
            raise ValueError(
                "--train-manifest-path is required when --config-name is not used."
            )
        require_val_manifest = not bool(args.generate_val_csvs) and not bool(args.auto_split_enabled)
        spec = _resolve_explicit(
            train_manifest_path=args.train_manifest_path,
            val_manifest_path=args.val_manifest_path,
            target_labels=args.target_labels,
            require_val_manifest=require_val_manifest,
        )

    budgets = sorted(set(_parse_int_list(args.budgets, name="budgets")))
    seeds = _parse_int_list(args.seeds, name="seeds")
    allow_random_fallback = not bool(args.disable_random_fallback)
    if not 0.0 < float(args.val_fraction) < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")

    if args.auto_split_enabled:
        log.warning("--auto-split-enabled is deprecated in generator semantics.")

    budget_pairs = _build_budget_pairs(
        budgets,
        budget_type=str(args.budget_type),
        auto_split_val_fraction=float(args.val_fraction),
    )

    if bool(args.generate_val_csvs) and str(args.budget_type) == "effective_train_rows":
        raise ValueError(
            "--budget-type effective_train_rows is not supported with --generate-val-csvs. "
            "Use plain budget sizes (manifest_rows) as total subset sizes."
        )

    selection_source: Optional[Dict[Tuple[int, int], List[str]]] = None
    if args.selection_source_path:
        selection_source = _load_selection_source(Path(args.selection_source_path))

    # 2) Load train/validation manifests and validate label columns.
    train_path = spec.manifest_dir / spec.train_manifest_name
    if not train_path.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_path}")

    train_df = pd.read_csv(train_path)
    val_df: Optional[pd.DataFrame] = None
    val_path: Optional[Path] = None
    if bool(args.generate_val_csvs):
        # Validation CSVs are generated later per (budget, seed).
        pass
    elif not spec.auto_split_enabled:
        val_path = spec.manifest_dir / spec.val_manifest_name
        if not val_path.exists():
            raise FileNotFoundError(f"Validation manifest not found: {val_path}")
        val_df = pd.read_csv(val_path)
    elif spec.val_manifest_name:
        log.info(
            "Config has data.auto_split.enabled=true; ignoring fixed data.val_manifest=%s for budget generation.",
            spec.val_manifest_name,
        )

    _validate_labels_presence(train_df, val_df, spec.target_labels)

    # 3) Prepare output location and fixed validation reference.
    output_dir = spec.manifest_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.generate_val_csvs):
        if args.copy_validation:
            log.warning("--copy-validation is ignored when --generate-val-csvs is enabled.")
        val_manifest_name_for_generated = ""
    elif spec.auto_split_enabled:
        if args.copy_validation:
            log.warning(
                "--copy-validation is ignored when data.auto_split.enabled=true because no fixed val manifest is used."
            )
        val_manifest_name_for_generated = spec.val_manifest_name
    elif args.copy_validation:
        fixed_val_name = spec.val_manifest_name
        fixed_val_path = output_dir / fixed_val_name
        if val_df is None:
            raise RuntimeError("Internal error: validation dataframe is missing while copy-validation is enabled.")
        _write_csv(fixed_val_path, val_df)
        val_manifest_name_for_generated = fixed_val_name
    else:
        if val_path is None:
            raise RuntimeError("Internal error: validation path is missing in fixed-validation mode.")
        val_manifest_name_for_generated = _relative_name(val_path, output_dir)

    # 4) Generate one train manifest per (seed, budget) pair.
    index_rows: List[dict] = []
    selection_rows: List[dict] = []

    if args.selection_column not in train_df.columns:
        raise ValueError(
            f"Selection column '{args.selection_column}' not found in train manifest."
        )

    if str(args.budget_type) == "effective_train_rows" and not bool(args.generate_val_csvs):
        log.warning(
            "budget-type=effective_train_rows active. Budgets are converted using val_fraction=%.3f.",
            float(args.val_fraction),
        )
    elif str(args.budget_type) == "effective_train_rows" and bool(args.generate_val_csvs):
        log.warning(
            "budget-type=effective_train_rows is incompatible with --generate-val-csvs and should be avoided.",
        )
    elif bool(args.generate_val_csvs):
        log.info(
            "Generating fixed train/val manifests per budget with val_fraction=%.3f (val_stratify=%s).",
            float(args.val_fraction),
            bool(args.val_stratify),
        )

    coverage_enforcement_enabled = not bool(args.disable_label_coverage_enforcement)
    if bool(args.generate_val_csvs) and coverage_enforcement_enabled and spec.target_labels:
        log.info(
            "Label coverage enforcement enabled for generated splits: val_fraction=%.3f, stratify=%s, max_extra_rows=%d",
            float(args.val_fraction),
            bool(args.val_stratify),
            int(args.coverage_max_extra_rows),
        )
    elif bool(args.generate_val_csvs) and coverage_enforcement_enabled and not spec.target_labels:
        log.warning(
            "Label coverage enforcement is enabled but no target labels were provided in explicit mode; "
            "pass --target-labels to enforce per-label class coverage."
        )

    def _sample_without_selection(
        *,
        budget: int,
        seed: int,
    ) -> Tuple[pd.DataFrame, str]:
        strategy_note_local = str(args.sampling_strategy)
        if args.sample_unit == "rows":
            if args.sampling_strategy == "stratified":
                sampled_local, strategy_note_local = _sample_rows_stratified(
                    train_df,
                    budget=budget,
                    seed=seed,
                    target_labels=spec.target_labels,
                    allow_random_fallback=allow_random_fallback,
                )
            else:
                sampled_local = _sample_rows(train_df, budget=budget, seed=seed)
        else:
            if args.sampling_strategy == "stratified":
                log.warning(
                    "sample-unit=groups does not support stratified selection yet; using random group sampling."
                )
                strategy_note_local = "random_groups"
            sampled_local = _sample_groups(
                train_df,
                budget=budget,
                seed=seed,
                group_column=args.group_column,
                group_separator=args.group_separator,
                group_remove_last=args.group_remove_last,
            )
            if len(sampled_local) == 0:
                raise RuntimeError(
                    f"Group sampling produced 0 rows for budget={budget}, seed={seed}."
                )
        return sampled_local, strategy_note_local

    for seed in seeds:
        for requested_effective_budget, sampled_manifest_budget in budget_pairs:
            adjusted_effective_budget = requested_effective_budget
            adjusted_sampled_manifest_budget = sampled_manifest_budget
            if adjusted_sampled_manifest_budget > len(train_df):
                adjusted_sampled_manifest_budget = len(train_df)
                if str(args.budget_type) == "effective_train_rows":
                    adjusted_effective_budget = int(
                        math.floor(
                            len(train_df) * (1.0 - float(args.val_fraction))
                        )
                    )
                else:
                    adjusted_effective_budget = adjusted_sampled_manifest_budget

                log.warning(
                    "Requested budget=%d (sampled_manifest_rows=%d) exceeds train size=%d; "
                    "auto-capping to sampled_manifest_rows=%d and effective_train_rows=%d.",
                    requested_effective_budget,
                    sampled_manifest_budget,
                    len(train_df),
                    adjusted_sampled_manifest_budget,
                    adjusted_effective_budget,
                )

            strategy_note = args.sampling_strategy
            if selection_source is not None:
                selection_key = (requested_effective_budget, seed)
                selected_values = selection_source.get(selection_key)
                if selected_values is None:
                    log.warning(
                        "No selection rows for budget=%d, seed=%d in %s; falling back to %s sampling.",
                        requested_effective_budget,
                        seed,
                        args.selection_source_path,
                        args.sampling_strategy,
                    )
                    sampled_df, strategy_note = _sample_without_selection(
                        budget=adjusted_sampled_manifest_budget,
                        seed=seed,
                    )
                    strategy_note = f"{strategy_note}|selection_missing_fallback"
                else:
                    selection_set = set(selected_values)
                    sampled_df = train_df[
                        train_df[args.selection_column].astype(str).isin(selection_set)
                    ].sort_index()

                    if len(sampled_df) == 0:
                        log.warning(
                            "Selection source matched 0 rows for budget=%d, seed=%d on column '%s'; "
                            "falling back to %s sampling.",
                            requested_effective_budget,
                            seed,
                            args.selection_column,
                            args.sampling_strategy,
                        )
                        sampled_df, strategy_note = _sample_without_selection(
                            budget=adjusted_sampled_manifest_budget,
                            seed=seed,
                        )
                        strategy_note = f"{strategy_note}|selection_unmatched_fallback"
                    else:
                        if len(sampled_df) != len(selection_set):
                            log.warning(
                                "Selection source partially matched for budget=%d, seed=%d: matched=%d of expected=%d.",
                                requested_effective_budget,
                                seed,
                                len(sampled_df),
                                len(selection_set),
                            )
                        strategy_note = "selection_source"
            else:
                sampled_df, strategy_note = _sample_without_selection(
                    budget=adjusted_sampled_manifest_budget,
                    seed=seed,
                )

            if bool(args.generate_val_csvs) and coverage_enforcement_enabled and spec.target_labels:
                sampled_df, coverage_note = _enforce_auto_split_label_coverage(
                    sampled_df,
                    train_df,
                    target_labels=spec.target_labels,
                    split_seed=seed,
                    val_fraction=float(args.val_fraction),
                    stratify_enabled=bool(args.val_stratify),
                    max_extra_rows=int(args.coverage_max_extra_rows),
                )
                strategy_note = f"{strategy_note}|{coverage_note}"

            if bool(args.generate_val_csvs):
                split_train_df, split_val_df = _simulate_grouped_auto_split(
                    sampled_df,
                    target_labels=spec.target_labels,
                    val_fraction=float(args.val_fraction),
                    split_seed=seed,
                    stratify_enabled=bool(args.val_stratify),
                )

                train_out_name = f"{Path(spec.train_manifest_name).stem}_n{requested_effective_budget}_s{seed}.csv"
                val_out_name = f"{Path(spec.train_manifest_name).stem}_n{requested_effective_budget}_s{seed}_val.csv"
                train_out_path = output_dir / train_out_name
                val_out_path = output_dir / val_out_name
                _write_csv(train_out_path, split_train_df)
                _write_csv(val_out_path, split_val_df)

                out_name = train_out_name
                out_path = train_out_path
                actual_train_rows = len(split_train_df)
                val_manifest_value = val_out_name
            else:
                out_name = f"{Path(spec.train_manifest_name).stem}_n{requested_effective_budget}_s{seed}.csv"
                out_path = output_dir / out_name
                _write_csv(out_path, sampled_df)
                actual_train_rows = len(sampled_df)
                val_manifest_value = val_manifest_name_for_generated

            selected_ids = sampled_df[args.selection_column].astype(str).tolist()
            for selected_id in selected_ids:
                selection_rows.append(
                    {
                        "budget": requested_effective_budget,
                        "seed": seed,
                        "selection_value": selected_id,
                    }
                )

            index_rows.append(
                {
                    "budget": requested_effective_budget,
                    "seed": seed,
                    "budget_type": args.budget_type,
                    "requested_effective_train_rows": requested_effective_budget,
                    "adjusted_effective_train_rows": adjusted_effective_budget,
                    "sampled_manifest_rows": adjusted_sampled_manifest_budget,
                    "sample_unit": args.sample_unit,
                    "sampling_strategy": args.sampling_strategy,
                    "strategy_note": strategy_note,
                    "requested_train_rows": requested_effective_budget,
                    "actual_train_rows": actual_train_rows,
                    "train_manifest": out_name,
                    "val_manifest": val_manifest_value,
                }
            )

            log.info(
                "Wrote train subset: %s (requested=%d, actual=%d)",
                out_path,
                requested_effective_budget,
                len(sampled_df),
            )

    # 5) Write index file for easy Hydra override wiring.
    index_path = output_dir / "manifest_index.csv"
    _write_manifest_index(index_path, index_rows)

    selection_export_path = (
        Path(args.selection_export_path)
        if args.selection_export_path
        else (output_dir / "selection_index.csv")
    )
    _write_selection_index(selection_export_path, selection_rows)

    log.info("Done. Generated %d train subsets in: %s", len(index_rows), output_dir)
    log.info("Index file: %s", index_path)
    log.info("Selection index: %s", selection_export_path)
    log.info(
        "Use with Hydra overrides like: paths.manifest_dir=%s data.train_manifest=<train_manifest_from_index> data.val_manifest=<val_manifest_from_index>",
        str(output_dir).replace("\\", "/"),
    )


if __name__ == "__main__":
    main()
