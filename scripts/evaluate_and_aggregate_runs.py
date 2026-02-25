"""Evaluate trained run folders on test manifests and aggregate results.

This script discovers training runs under a root directory, evaluates each
checkpoint on configured test manifests, writes a per-run CSV, and writes a
by-budget summary CSV (mean/std across seeds).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from classification.evaluate import evaluate_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch evaluation and aggregation.

    Returns:
        Parsed ``argparse.Namespace``.

    Logic:
        Configure run discovery, evaluation overrides, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate run folders and aggregate test metrics by budget."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        required=True,
        help="Root directory containing run subfolders (e.g., outputs/gpt_budget).",
    )
    parser.add_argument(
        "--test-manifest-dir",
        type=str,
        required=True,
        help="Manifest directory containing test manifests referenced by config.",
    )
    parser.add_argument(
        "--test-manifests",
        type=str,
        default=None,
        help="Optional comma-separated test manifest names override.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Label source tag stored in CSV rows (e.g., gpt or manual).",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="final_model.pt",
        help="Checkpoint filename inside each run's checkpoints directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/aggregated_results",
        help="Directory where aggregation CSV files are written.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output prefix. Defaults to runs-root folder name.",
    )
    parser.add_argument(
        "--eval-subdir",
        type=str,
        default="evaluation_aggregate",
        help="Subfolder name under each run used to store detailed eval artifacts.",
    )
    return parser.parse_args()


def _infer_source(runs_root: Path) -> str:
    """Infer source tag from runs root folder name.

    Args:
        runs_root: Root directory containing runs.

    Returns:
        Inferred source label string.
    """
    name = runs_root.name.lower()
    if "gpt" in name:
        return "gpt"
    if "manual" in name:
        return "manual"
    return runs_root.name


def _parse_test_manifests(raw: Optional[str]) -> Optional[List[str]]:
    """Parse optional comma-separated test manifest names.

    Args:
        raw: Raw CLI string.

    Returns:
        List of manifest names or ``None`` when not provided.
    """
    if not raw:
        return None
    values = [token.strip() for token in raw.split(",") if token.strip()]
    return values or None


def _discover_runs(runs_root: Path, checkpoint_name: str) -> List[Path]:
    """Discover run directories that contain config and checkpoint artifacts.

    Args:
        runs_root: Root folder under which runs are stored.
        checkpoint_name: Expected checkpoint filename.

    Returns:
        Sorted list of run directories.

    Logic:
        Find ``.hydra/config.yaml`` files and keep those whose parent run folder
        contains ``checkpoints/<checkpoint_name>``.
    """
    runs: List[Path] = []
    for cfg_path in runs_root.rglob(".hydra/config.yaml"):
        run_dir = cfg_path.parent.parent
        if "evaluation" in {part.lower() for part in run_dir.parts}:
            continue
        checkpoint_path = run_dir / "checkpoints" / checkpoint_name
        if checkpoint_path.exists():
            runs.append(run_dir)

    runs = sorted(set(runs))
    return runs


def _extract_budget_seed(train_manifest: Optional[str]) -> Dict[str, Optional[int]]:
    """Extract budget and seed from train manifest name.

    Args:
        train_manifest: Train manifest filename.

    Returns:
        Dict with optional integer keys ``budget_n`` and ``split_seed``.
    """
    if not train_manifest:
        return {"budget_n": None, "split_seed": None}

    name = Path(train_manifest).name
    match = re.search(r"_n(\d+)_s(\d+)", name)
    if not match:
        return {"budget_n": None, "split_seed": None}

    return {
        "budget_n": int(match.group(1)),
        "split_seed": int(match.group(2)),
    }


def _load_latest_run_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    """Load validation summary metrics from run artifacts.

    Args:
        run_dir: Run directory path.

    Returns:
        Dict with best validation metrics.

    Logic:
        Prefer run-local ``best_metrics.json`` (stores best validation snapshot),
        then fallback to the final row in ``metrics.jsonl`` (which includes
        running best values), and finally fallback to run-local
        ``latest_run.json`` when present.
    """
    default_payload = {
        "best_val_auprc": None,
        "best_val_auroc": None,
        "best_val_f1_macro": None,
    }

    best_metrics_path = run_dir / "best_metrics.json"
    if best_metrics_path.exists():
        try:
            payload = json.loads(best_metrics_path.read_text(encoding="utf-8"))
            val_metrics = payload.get("val_metrics", {}) if isinstance(payload, dict) else {}
            return {
                "best_val_auprc": val_metrics.get("auprc"),
                "best_val_auroc": val_metrics.get("auroc"),
                "best_val_f1_macro": val_metrics.get("f1_macro"),
            }
        except Exception:
            pass

    metrics_jsonl_path = run_dir / "metrics.jsonl"
    if metrics_jsonl_path.exists():
        try:
            last_nonempty = None
            with metrics_jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        last_nonempty = line

            if last_nonempty is not None:
                payload = json.loads(last_nonempty)
                return {
                    "best_val_auprc": payload.get("best_val_auprc"),
                    "best_val_auroc": payload.get("best_val_auroc"),
                    "best_val_f1_macro": payload.get("best_val_f1_macro"),
                }
        except Exception:
            pass

    pointer_path = run_dir / "latest_run.json"
    if not pointer_path.exists():
        return default_payload

    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception:
        return default_payload

    return {
        "best_val_auprc": payload.get("best_val_auprc"),
        "best_val_auroc": payload.get("best_val_auroc"),
        "best_val_f1_macro": payload.get("best_val_f1_macro"),
    }


def _ensure_run_has_metrics_artifacts(run_dir: Path) -> None:
    """Warn when expected run-level metric artifacts are missing.

    Args:
        run_dir: Run directory path.

    Returns:
        ``None``.
    """
    has_best_metrics = (run_dir / "best_metrics.json").exists()
    has_metrics_jsonl = (run_dir / "metrics.jsonl").exists()
    has_latest_pointer = (run_dir / "latest_run.json").exists()

    if not (has_best_metrics or has_metrics_jsonl or has_latest_pointer):
        log.warning(
            "Run %s has no best_metrics.json, metrics.jsonl, or latest_run.json; "
            "validation metrics may be missing.",
            run_dir,
        )


def _configure_eval_cfg(
    cfg: DictConfig,
    run_dir: Path,
    checkpoint_path: Path,
    test_manifest_dir: Path,
    test_manifests: Optional[List[str]],
    eval_subdir: str,
) -> DictConfig:
    """Apply evaluation-time overrides to a loaded training config.

    Args:
        cfg: Run config loaded from ``.hydra/config.yaml``.
        run_dir: Path to run directory.
        checkpoint_path: Path to run checkpoint file.
        test_manifest_dir: Folder containing test manifests.
        test_manifests: Optional explicit test manifest names.
        eval_subdir: Per-run subdirectory for evaluation artifacts.

    Returns:
        Mutated config object ready for ``evaluate_model``.
    """
    eval_out_dir = run_dir / eval_subdir

    OmegaConf.update(cfg, "paths.checkpoint_path", str(checkpoint_path), force_add=True)
    OmegaConf.update(cfg, "paths.manifest_dir", str(test_manifest_dir).replace("\\", "/"), force_add=True)
    OmegaConf.update(cfg, "paths.output_dir", str(eval_out_dir).replace("\\", "/"), force_add=True)
    OmegaConf.update(cfg, "paths.run_dir", str(eval_out_dir).replace("\\", "/"), force_add=True)
    OmegaConf.update(cfg, "hydra.runtime.output_dir", str(eval_out_dir).replace("\\", "/"), force_add=True)

    if test_manifests is not None:
        OmegaConf.update(cfg, "data.test_manifests", test_manifests, force_add=True)

    return cfg


def _evaluate_one_run(
    run_dir: Path,
    test_manifest_dir: Path,
    checkpoint_name: str,
    test_manifests: Optional[List[str]],
    source: str,
    eval_subdir: str,
) -> Dict[str, Any]:
    """Evaluate one run and return a flat record for CSV export.

    Args:
        run_dir: Run folder containing `.hydra/config.yaml` and checkpoint.
        test_manifest_dir: Folder with test manifests.
        checkpoint_name: Model checkpoint filename.
        test_manifests: Optional test manifest overrides.
        source: Source label tag for output row.
        eval_subdir: Per-run subdirectory for evaluation artifacts.

    Returns:
        Flat dictionary containing metadata, validation, and test metrics.
    """
    cfg_path = run_dir / ".hydra" / "config.yaml"
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {cfg_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    _ensure_run_has_metrics_artifacts(run_dir)

    cfg = OmegaConf.load(cfg_path)
    cfg = _configure_eval_cfg(
        cfg=cfg,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        test_manifest_dir=test_manifest_dir,
        test_manifests=test_manifests,
        eval_subdir=eval_subdir,
    )

    test_metrics = evaluate_model(cfg)
    val_metrics = _load_latest_run_metrics(run_dir)

    train_manifest = OmegaConf.select(cfg, "data.train_manifest", default=None)
    val_manifest = OmegaConf.select(cfg, "data.val_manifest", default=None)
    parsed = _extract_budget_seed(train_manifest)

    row: Dict[str, Any] = {
        "source": source,
        "run_dir": str(run_dir).replace("\\", "/"),
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "budget_n": parsed["budget_n"],
        "split_seed": parsed["split_seed"],
        "utils_seed": OmegaConf.select(cfg, "utils.seed", default=None),
        "learning_rate": OmegaConf.select(cfg, "training.learning_rate", default=None),
        "weight_decay": OmegaConf.select(cfg, "training.weight_decay", default=None),
        "dropout": OmegaConf.select(cfg, "model.params.dropout", default=None),
        "hidden_dims": str(OmegaConf.select(cfg, "model.params.hidden_dims", default=None)),
        "batch_size": OmegaConf.select(cfg, "training.batch_size", default=None),
    }
    row.update(val_metrics)
    row.update(test_metrics)
    return row


def _aggregate_by_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric metrics by source and budget.

    Args:
        df: Per-run dataframe.

    Returns:
        Grouped dataframe with mean/std/count columns.
    """
    if df.empty:
        return df

    metric_cols = [
        col
        for col in df.columns
        if (
            col.startswith("best_val_")
            or col.startswith("test_")
        )
    ]

    numeric_df = df.copy()
    for col in metric_cols:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    grouped = (
        numeric_df.groupby(["source", "budget_n"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]

    if "budget_n_" in grouped.columns:
        grouped = grouped.rename(columns={"budget_n_": "budget_n"})
    if "source_" in grouped.columns:
        grouped = grouped.rename(columns={"source_": "source"})

    return grouped.sort_values(["source", "budget_n"], na_position="last")


def main() -> None:
    """CLI entrypoint for evaluation and aggregation.

    Returns:
        ``None``.

    Logic:
        Discover run folders, evaluate each checkpoint on test manifests,
        write per-run CSV, then compute and write by-budget mean/std summary.
    """
    args = _parse_args()

    runs_root = Path(args.runs_root)
    test_manifest_dir = Path(args.test_manifest_dir)

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")
    if not test_manifest_dir.exists():
        raise FileNotFoundError(f"Test manifest directory not found: {test_manifest_dir}")

    source = args.source or _infer_source(runs_root)
    test_manifests = _parse_test_manifests(args.test_manifests)

    run_dirs = _discover_runs(runs_root, checkpoint_name=args.checkpoint_name)
    if not run_dirs:
        raise RuntimeError(
            f"No runs with checkpoints found under {runs_root} using checkpoint name '{args.checkpoint_name}'."
        )

    log.info("Discovered %d runs under %s", len(run_dirs), runs_root)

    rows: List[Dict[str, Any]] = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        log.info("[%d/%d] Evaluating run: %s", idx, len(run_dirs), run_dir)
        row = _evaluate_one_run(
            run_dir=run_dir,
            test_manifest_dir=test_manifest_dir,
            checkpoint_name=args.checkpoint_name,
            test_manifests=test_manifests,
            source=source,
            eval_subdir=args.eval_subdir,
        )
        rows.append(row)

    per_run_df = pd.DataFrame(rows)
    by_budget_df = _aggregate_by_budget(per_run_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or runs_root.name

    per_run_path = output_dir / f"{prefix}_per_run.csv"
    by_budget_path = output_dir / f"{prefix}_by_budget.csv"

    per_run_df.to_csv(per_run_path, index=False)
    by_budget_df.to_csv(by_budget_path, index=False)

    log.info("Wrote per-run results: %s", per_run_path)
    log.info("Wrote by-budget summary: %s", by_budget_path)


if __name__ == "__main__":
    main()
