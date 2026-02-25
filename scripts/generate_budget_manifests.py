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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from omegaconf import ListConfig, OmegaConf


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ManifestSpec:
    manifest_dir: Path
    train_manifest_name: str
    val_manifest_name: str
    target_labels: List[str]



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



def _resolve_from_config(config_name: str) -> ManifestSpec:
    """Resolve manifest inputs from a Hydra config file.

    Args:
        config_name: Hydra config name/path.

    Returns:
        ``ManifestSpec`` with manifest directory, train/val names, and labels.

    Raises:
        FileNotFoundError: If config file cannot be found.
        ValueError: If required config keys are missing.

    Logic:
        Load config, then extract ``paths.manifest_dir``,
        ``data.train_manifest``, ``data.val_manifest``, and optional
        ``training.target_labels``.
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

    if manifest_dir_value is None or train_name is None or val_name is None:
        raise ValueError(
            "Config must define paths.manifest_dir, data.train_manifest, and data.val_manifest."
        )

    if isinstance(labels, ListConfig):
        labels = list(labels)
    elif not isinstance(labels, list):
        labels = []

    return ManifestSpec(
        manifest_dir=Path(str(manifest_dir_value)),
        train_manifest_name=str(train_name),
        val_manifest_name=str(val_name),
        target_labels=[str(label) for label in labels],
    )



def _resolve_explicit(
    train_manifest_path: str,
    val_manifest_path: str,
    target_labels: Optional[str],
) -> ManifestSpec:
    """Resolve manifest inputs from explicit train/val CSV paths.

    Args:
        train_manifest_path: Path to training manifest CSV.
        val_manifest_path: Path to validation manifest CSV.
        target_labels: Optional comma-separated label names.

    Returns:
        ``ManifestSpec`` with explicit paths converted to manifest names.

    Raises:
        FileNotFoundError: If either manifest file does not exist.
        ValueError: If train and val are in different folders.

    Logic:
        Validate path existence and shared parent directory so a single
        ``paths.manifest_dir`` can reference both manifests.
    """
    train_path = Path(train_manifest_path)
    val_path = Path(val_manifest_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {val_path}")
    if train_path.parent != val_path.parent:
        raise ValueError(
            "Explicit train/val manifests must be in the same folder so one manifest_dir can reference both."
        )

    labels = []
    if target_labels:
        labels = [token.strip() for token in target_labels.split(",") if token.strip()]

    return ManifestSpec(
        manifest_dir=train_path.parent,
        train_manifest_name=train_path.name,
        val_manifest_name=val_path.name,
        target_labels=labels,
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



def _relative_name(path: Path, base_dir: Path) -> str:
    """Return a POSIX-style relative path for Hydra manifest references."""
    return str(path.relative_to(base_dir)).replace("\\", "/")



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
        "--budgets",
        type=str,
        required=True,
        help="Comma-separated train subset sizes (e.g., 100,250,500,1000).",
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

    return parser.parse_args()



def _validate_labels_presence(train_df: pd.DataFrame, val_df: pd.DataFrame, labels: Iterable[str]) -> None:
    """Validate that target label columns exist in both train and val manifests.

    Args:
        train_df: Training manifest dataframe.
        val_df: Validation manifest dataframe.
        labels: Expected target label names.

    Raises:
        ValueError: If one or more labels are missing from either manifest.
    """
    label_list = [label for label in labels if label]
    if not label_list:
        return
    missing_train = [label for label in label_list if label not in train_df.columns]
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
        if not args.train_manifest_path or not args.val_manifest_path:
            raise ValueError(
                "--train-manifest-path and --val-manifest-path are required when --config-name is not used."
            )
        spec = _resolve_explicit(
            train_manifest_path=args.train_manifest_path,
            val_manifest_path=args.val_manifest_path,
            target_labels=args.target_labels,
        )

    budgets = sorted(set(_parse_int_list(args.budgets, name="budgets")))
    seeds = _parse_int_list(args.seeds, name="seeds")
    allow_random_fallback = not bool(args.disable_random_fallback)

    # 2) Load train/validation manifests and validate label columns.
    train_path = spec.manifest_dir / spec.train_manifest_name
    val_path = spec.manifest_dir / spec.val_manifest_name
    if not train_path.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    _validate_labels_presence(train_df, val_df, spec.target_labels)

    # 3) Prepare output location and fixed validation reference.
    output_dir = spec.manifest_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_validation:
        fixed_val_name = spec.val_manifest_name
        fixed_val_path = output_dir / fixed_val_name
        _write_csv(fixed_val_path, val_df)
        val_manifest_name_for_generated = fixed_val_name
    else:
        val_manifest_name_for_generated = _relative_name(val_path, output_dir)

    # 4) Generate one train manifest per (seed, budget) pair.
    index_rows: List[dict] = []

    for seed in seeds:
        for budget in budgets:
            strategy_note = args.sampling_strategy
            if args.sample_unit == "rows":
                if args.sampling_strategy == "stratified":
                    sampled_df, strategy_note = _sample_rows_stratified(
                        train_df,
                        budget=budget,
                        seed=seed,
                        target_labels=spec.target_labels,
                        allow_random_fallback=allow_random_fallback,
                    )
                else:
                    sampled_df = _sample_rows(train_df, budget=budget, seed=seed)
            else:
                if args.sampling_strategy == "stratified":
                    log.warning(
                        "sample-unit=groups does not support stratified selection yet; using random group sampling."
                    )
                    strategy_note = "random_groups"
                sampled_df = _sample_groups(
                    train_df,
                    budget=budget,
                    seed=seed,
                    group_column=args.group_column,
                    group_separator=args.group_separator,
                    group_remove_last=args.group_remove_last,
                )
                if len(sampled_df) == 0:
                    raise RuntimeError(
                        f"Group sampling produced 0 rows for budget={budget}, seed={seed}."
                    )

            out_name = f"{Path(spec.train_manifest_name).stem}_n{budget}_s{seed}.csv"
            out_path = output_dir / out_name
            _write_csv(out_path, sampled_df)

            index_rows.append(
                {
                    "budget": budget,
                    "seed": seed,
                    "sample_unit": args.sample_unit,
                    "sampling_strategy": args.sampling_strategy,
                    "strategy_note": strategy_note,
                    "requested_train_rows": budget,
                    "actual_train_rows": len(sampled_df),
                    "train_manifest": out_name,
                    "val_manifest": val_manifest_name_for_generated,
                }
            )

            log.info(
                "Wrote train subset: %s (requested=%d, actual=%d)",
                out_path,
                budget,
                len(sampled_df),
            )

    # 5) Write index file for easy Hydra override wiring.
    index_path = output_dir / "manifest_index.csv"
    _write_manifest_index(index_path, index_rows)

    log.info("Done. Generated %d train subsets in: %s", len(index_rows), output_dir)
    log.info("Index file: %s", index_path)
    log.info(
        "Use with Hydra overrides like: paths.manifest_dir=%s data.train_manifest=<train_manifest_from_index> data.val_manifest=<val_manifest_from_index>",
        str(output_dir).replace("\\", "/"),
    )


if __name__ == "__main__":
    main()
