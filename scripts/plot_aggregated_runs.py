"""Plot aggregated budget-sweep metrics with mean±std error bars.

This script consumes one or more ``*_by_budget.csv`` files produced by
``scripts/evaluate_and_aggregate_runs.py`` and creates one line graph per
metric (AUPRC, AUROC, F1-macro by default).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean±std budget curves from aggregated by-budget CSV files."
    )
    parser.add_argument(
        "--by-budget-csvs",
        type=str,
        nargs="+",
        required=True,
        help="One or more *_by_budget.csv files.",
    )
    parser.add_argument(
        "--metric-prefix",
        type=str,
        default="test_test_manual_all.csv",
        help=(
            "Metric prefix used in columns (e.g., test_test_manual_all.csv or "
            "best_val)."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="auprc,auroc,f1_macro",
        help="Comma-separated metrics to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/aggregated_results/plots",
        help="Directory where plot files are written.",
    )
    parser.add_argument(
        "--file-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot resolution in DPI.",
    )
    return parser.parse_args()


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def _load_frames(paths: List[str]) -> pd.DataFrame:
    frames = []
    for raw_path in paths:
        csv_path = Path(raw_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        if "source" not in frame.columns:
            frame["source"] = csv_path.stem
        frame["_csv_path"] = str(csv_path)
        frames.append(frame)

    if not frames:
        raise RuntimeError("No input CSV files were loaded.")
    return pd.concat(frames, ignore_index=True)


def _required_cols(metric_prefix: str, metric_name: str) -> tuple[str, str]:
    return (
        f"{metric_prefix}_{metric_name}_mean",
        f"{metric_prefix}_{metric_name}_std",
    )


def _candidate_prefixes(metric_prefix: str) -> List[str]:
    """Build compatible metric-prefix variants for legacy/new CSV naming."""
    candidates = [metric_prefix]

    if metric_prefix.startswith("test_test_"):
        candidates.append(metric_prefix.replace("test_test_", "test_", 1))
    elif metric_prefix.startswith("test_"):
        candidates.append(metric_prefix.replace("test_", "test_test_", 1))
    else:
        candidates.append(f"test_{metric_prefix}")
        candidates.append(f"test_test_{metric_prefix}")

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _resolve_metric_prefix(df: pd.DataFrame, metric_prefix: str, metric_name: str) -> str:
    """Resolve a metric prefix against available columns with compatibility fallbacks."""
    for candidate in _candidate_prefixes(metric_prefix):
        mean_col, std_col = _required_cols(candidate, metric_name)
        if mean_col in df.columns and std_col in df.columns:
            return candidate
    raise ValueError(
        f"Could not find columns for metric '{metric_name}' using prefix '{metric_prefix}'."
    )


def _plot_metric(
    df: pd.DataFrame,
    metric_prefix: str,
    metric_name: str,
    output_dir: Path,
    file_format: str,
    dpi: int,
) -> Path:
    resolved_prefix = _resolve_metric_prefix(df, metric_prefix, metric_name)
    mean_col, std_col = _required_cols(resolved_prefix, metric_name)

    missing_cols = [
        col for col in ["budget_n", "source", mean_col, std_col] if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            "Missing required columns for metric plot "
            f"{metric_name}: {missing_cols}. "
            f"Available columns include: {list(df.columns)}"
        )

    plot_df = df.copy()
    plot_df["budget_n"] = pd.to_numeric(plot_df["budget_n"], errors="coerce")
    plot_df[mean_col] = pd.to_numeric(plot_df[mean_col], errors="coerce")
    plot_df[std_col] = pd.to_numeric(plot_df[std_col], errors="coerce").fillna(0.0)
    plot_df = plot_df.dropna(subset=["budget_n", mean_col]).sort_values("budget_n")

    if plot_df.empty:
        raise RuntimeError(f"No plottable rows found for metric '{metric_name}'.")

    fig, ax = plt.subplots(figsize=(9, 5))
    for source, source_df in plot_df.groupby("source", sort=True):
        source_df = source_df.sort_values("budget_n")
        ax.errorbar(
            source_df["budget_n"],
            source_df[mean_col],
            yerr=source_df[std_col],
            marker="o",
            capsize=4,
            linewidth=2,
            label=str(source),
        )

    title_metric = metric_name.upper().replace("F1_MACRO", "F1")
    ax.set_title(f"{title_metric} vs Budget ({resolved_prefix})")
    ax.set_xlabel("Budget (n)")
    ax.set_ylabel(f"Mean {title_metric}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Source")

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _sanitize_filename(f"{resolved_prefix}_{metric_name}.{file_format}")
    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = _parse_args()

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    if not metrics:
        raise ValueError("No metrics provided. Use --metrics with at least one metric name.")

    df = _load_frames(args.by_budget_csvs)
    output_dir = Path(args.output_dir)

    written_paths: List[Path] = []
    for metric_name in metrics:
        out_path = _plot_metric(
            df=df,
            metric_prefix=args.metric_prefix,
            metric_name=metric_name,
            output_dir=output_dir,
            file_format=args.file_format,
            dpi=args.dpi,
        )
        written_paths.append(out_path)

    print("Wrote plot files:")
    for path in written_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
