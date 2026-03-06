"""Compare per-class bottlenecks between manual and GPT models.

This script aligns per-class detailed metric reports by fold and class, then
computes paired deltas (manual - gpt) for precision, recall, and F1.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fold-paired per-class metrics between manual and GPT models."
    )
    parser.add_argument(
        "--manual-glob",
        type=str,
        required=True,
        help="Glob pattern for manual detailed metric JSON files.",
    )
    parser.add_argument(
        "--gpt-glob",
        type=str,
        required=True,
        help="Glob pattern for GPT detailed metric JSON files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/aggregated_results/per_class_bottleneck",
        help="Output prefix path. Writes <prefix>_paired.csv and <prefix>_summary.csv.",
    )
    parser.add_argument(
        "--label-manual",
        type=str,
        default="manual",
        help="Label name for manual model in outputs.",
    )
    parser.add_argument(
        "--label-gpt",
        type=str,
        default="gpt",
        help="Label name for GPT model in outputs.",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Optional markdown report output path.",
    )
    return parser.parse_args()


def _expand_glob(pattern: str) -> List[Path]:
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No files matched glob: {pattern}")
    return [Path(path).resolve() for path in matches]


def _extract_fold(manifest_name: str, file_path: Path) -> Optional[int]:
    candidates = [manifest_name, file_path.name, str(file_path)]
    for value in candidates:
        match = re.search(r"_f(\d+)_test", value)
        if match:
            return int(match.group(1))
    return None


def _load_rows(paths: Iterable[Path], source_label: str) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest_name = str(payload.get("manifest", ""))
        fold = _extract_fold(manifest_name, path)
        per_class = payload.get("per_class", {})
        if not isinstance(per_class, dict):
            continue

        for class_name, metrics in per_class.items():
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "source": source_label,
                    "file_path": str(path).replace("\\", "/"),
                    "manifest": manifest_name,
                    "fold": fold,
                    "class_name": str(class_name),
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1": float(metrics.get("f1", 0.0)),
                    "support": float(metrics.get("support", 0.0)),
                }
            )
    return rows


def _paired_comparison(manual_df: pd.DataFrame, gpt_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["fold", "class_name"]

    left = manual_df.rename(
        columns={
            "precision": "manual_precision",
            "recall": "manual_recall",
            "f1": "manual_f1",
            "support": "manual_support",
            "manifest": "manual_manifest",
            "file_path": "manual_file_path",
        }
    )
    right = gpt_df.rename(
        columns={
            "precision": "gpt_precision",
            "recall": "gpt_recall",
            "f1": "gpt_f1",
            "support": "gpt_support",
            "manifest": "gpt_manifest",
            "file_path": "gpt_file_path",
        }
    )

    merged = left.merge(right, on=key_cols, how="inner")
    if merged.empty:
        raise RuntimeError(
            "No fold/class overlap found between manual and GPT detailed metric files."
        )

    merged["delta_precision"] = merged["manual_precision"] - merged["gpt_precision"]
    merged["delta_recall"] = merged["manual_recall"] - merged["gpt_recall"]
    merged["delta_f1"] = merged["manual_f1"] - merged["gpt_f1"]
    return merged.sort_values(["class_name", "fold"])


def _summary_table(paired_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        paired_df.groupby("class_name", dropna=False)[
            [
                "manual_precision",
                "manual_recall",
                "manual_f1",
                "gpt_precision",
                "gpt_recall",
                "gpt_f1",
                "delta_precision",
                "delta_recall",
                "delta_f1",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]

    if "class_name_" in grouped.columns:
        grouped = grouped.rename(columns={"class_name_": "class_name"})

    return grouped.sort_values("delta_f1_mean", ascending=False)


def _write_markdown(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_cols = [
        "class_name",
        "manual_f1_mean",
        "manual_f1_std",
        "gpt_f1_mean",
        "gpt_f1_std",
        "delta_f1_mean",
        "delta_f1_std",
        "delta_recall_mean",
        "delta_precision_mean",
    ]

    lines = [
        "# Per-Class Bottleneck Summary",
        "",
        "Positive deltas mean manual > GPT (potential GPT bottleneck).",
        "",
        "| Class | Manual F1 mean | Manual F1 std | GPT F1 mean | GPT F1 std | Delta F1 mean | Delta F1 std | Delta Recall mean | Delta Precision mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in summary_df[display_cols].iterrows():
        lines.append(
            "| {class_name} | {manual_f1_mean:.4f} | {manual_f1_std:.4f} | {gpt_f1_mean:.4f} | {gpt_f1_std:.4f} | {delta_f1_mean:.4f} | {delta_f1_std:.4f} | {delta_recall_mean:.4f} | {delta_precision_mean:.4f} |".format(
                class_name=row["class_name"],
                manual_f1_mean=float(row["manual_f1_mean"]),
                manual_f1_std=float(row["manual_f1_std"]),
                gpt_f1_mean=float(row["gpt_f1_mean"]),
                gpt_f1_std=float(row["gpt_f1_std"]),
                delta_f1_mean=float(row["delta_f1_mean"]),
                delta_f1_std=float(row["delta_f1_std"]),
                delta_recall_mean=float(row["delta_recall_mean"]),
                delta_precision_mean=float(row["delta_precision_mean"]),
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()

    manual_paths = _expand_glob(args.manual_glob)
    gpt_paths = _expand_glob(args.gpt_glob)

    manual_rows = _load_rows(manual_paths, args.label_manual)
    gpt_rows = _load_rows(gpt_paths, args.label_gpt)

    manual_df = pd.DataFrame(manual_rows)
    gpt_df = pd.DataFrame(gpt_rows)

    if manual_df.empty:
        raise RuntimeError("No manual per-class rows were loaded.")
    if gpt_df.empty:
        raise RuntimeError("No GPT per-class rows were loaded.")

    if manual_df["fold"].isna().any() or gpt_df["fold"].isna().any():
        raise ValueError(
            "Could not extract fold IDs from one or more files. "
            "Expected names like *_f1_test*."
        )

    manual_df["fold"] = manual_df["fold"].astype(int)
    gpt_df["fold"] = gpt_df["fold"].astype(int)

    paired_df = _paired_comparison(manual_df, gpt_df)
    summary_df = _summary_table(paired_df)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    paired_path = output_prefix.with_name(f"{output_prefix.name}_paired.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")

    paired_df.to_csv(paired_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote paired comparison: {paired_path}")
    print(f"Wrote class summary: {summary_path}")

    if args.output_markdown:
        md_path = Path(args.output_markdown)
        _write_markdown(summary_df, md_path)
        print(f"Wrote markdown summary: {md_path}")


if __name__ == "__main__":
    main()
