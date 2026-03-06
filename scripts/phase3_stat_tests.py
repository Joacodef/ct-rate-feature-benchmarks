"""Compute paired statistical inference for Phase 3 manual vs GPT comparisons.

This script consumes per-run CSVs produced by evaluate_and_aggregate_runs.py,
aligns manual and GPT runs by shared keys, and reports:
- paired deltas (manual - gpt),
- bootstrap confidence intervals for means,
- paired permutation-test p-values,
- Benjamini-Hochberg FDR-adjusted p-values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_csv_list(raw: str) -> List[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired CI and hypothesis tests from manual/GPT per-run CSVs."
    )
    parser.add_argument(
        "--manual-per-run",
        type=str,
        required=True,
        help="Path to manual per-run CSV.",
    )
    parser.add_argument(
        "--gpt-per-run",
        type=str,
        required=True,
        help="Path to GPT per-run CSV.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="test_primary_auprc,test_primary_auroc,test_primary_f1_macro",
        help="Comma-separated metric columns to compare.",
    )
    parser.add_argument(
        "--pair-keys",
        type=str,
        default="budget_n,cv_fold,utils_seed",
        help="Comma-separated keys used to pair manual/GPT rows.",
    )
    parser.add_argument(
        "--shared-budgets",
        type=str,
        default=None,
        help="Optional comma-separated shared budgets to include.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of paired sign-flip permutations for p-values.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level used for CI and adjusted-p rejection flag.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=52,
        help="Random seed for bootstrap/permutation reproducibility.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/aggregated_results/phase3_stats",
        help="Output prefix; writes <prefix>_paired.csv and <prefix>_summary.csv.",
    )
    return parser.parse_args()


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
) -> Tuple[float, float]:
    if values.size == 0:
        return (np.nan, np.nan)

    idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _paired_permutation_pvalue(
    deltas: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    if deltas.size == 0:
        return np.nan

    observed = abs(float(deltas.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, deltas.size))
    perm_means = np.abs((signs * deltas).mean(axis=1))
    pvalue = float((np.sum(perm_means >= observed) + 1) / (n_permutations + 1))
    return pvalue


def _benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    n = len(pvalues)
    adjusted = [np.nan] * n

    finite_items = [(idx, float(p)) for idx, p in enumerate(pvalues) if np.isfinite(p)]
    if not finite_items:
        return adjusted

    finite_items.sort(key=lambda x: x[1])
    m = len(finite_items)

    bh_values = np.empty(m, dtype=float)
    for rank, (_, pval) in enumerate(finite_items, start=1):
        bh_values[rank - 1] = pval * m / rank

    # Enforce monotonicity from largest rank to smallest.
    for i in range(m - 2, -1, -1):
        bh_values[i] = min(bh_values[i], bh_values[i + 1])

    bh_values = np.clip(bh_values, 0.0, 1.0)
    for i, (orig_idx, _) in enumerate(finite_items):
        adjusted[orig_idx] = float(bh_values[i])

    return adjusted


def _validate_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")


def _ensure_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _merge_paired_runs(
    manual_df: pd.DataFrame,
    gpt_df: pd.DataFrame,
    pair_keys: Sequence[str],
    metrics: Sequence[str],
    shared_budgets: Optional[Sequence[int]],
) -> pd.DataFrame:
    required = list(pair_keys) + list(metrics)
    _validate_columns(manual_df, required, "manual per-run CSV")
    _validate_columns(gpt_df, required, "gpt per-run CSV")

    manual = _ensure_numeric(manual_df, required)
    gpt = _ensure_numeric(gpt_df, required)

    if shared_budgets is not None:
        manual = manual[manual["budget_n"].isin(shared_budgets)].copy()
        gpt = gpt[gpt["budget_n"].isin(shared_budgets)].copy()

    # Keep one row per pairing key to avoid unintended many-to-many merges.
    keep_cols = list(pair_keys) + list(metrics)
    manual = manual[keep_cols].drop_duplicates(subset=list(pair_keys), keep="last")
    gpt = gpt[keep_cols].drop_duplicates(subset=list(pair_keys), keep="last")

    merged = manual.merge(
        gpt,
        on=list(pair_keys),
        how="inner",
        suffixes=("_manual", "_gpt"),
    )
    if merged.empty:
        raise RuntimeError(
            "No paired rows after merge. Check --pair-keys and shared budgets."
        )

    for metric in metrics:
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_manual"] - merged[f"{metric}_gpt"]
        )

    return merged


def _summarize(
    paired_df: pd.DataFrame,
    metrics: Sequence[str],
    alpha: float,
    n_bootstrap: int,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(random_seed)

    budgets = sorted(
        pd.to_numeric(paired_df["budget_n"], errors="coerce").dropna().astype(int).unique().tolist()
    )

    for budget in budgets:
        budget_df = paired_df[paired_df["budget_n"] == budget]

        for metric in metrics:
            mvals = pd.to_numeric(
                budget_df[f"{metric}_manual"], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            gvals = pd.to_numeric(
                budget_df[f"{metric}_gpt"], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            dvals = pd.to_numeric(
                budget_df[f"delta_{metric}"], errors="coerce"
            ).dropna().to_numpy(dtype=float)

            # Ensure all arrays represent the same paired rows.
            valid = budget_df[[f"{metric}_manual", f"{metric}_gpt", f"delta_{metric}"]].dropna()
            n_pairs = int(len(valid))
            if n_pairs == 0:
                continue

            mvals = valid[f"{metric}_manual"].to_numpy(dtype=float)
            gvals = valid[f"{metric}_gpt"].to_numpy(dtype=float)
            dvals = valid[f"delta_{metric}"].to_numpy(dtype=float)

            manual_ci_lo, manual_ci_hi = _bootstrap_mean_ci(
                mvals, rng=rng, n_bootstrap=n_bootstrap, alpha=alpha
            )
            gpt_ci_lo, gpt_ci_hi = _bootstrap_mean_ci(
                gvals, rng=rng, n_bootstrap=n_bootstrap, alpha=alpha
            )
            delta_ci_lo, delta_ci_hi = _bootstrap_mean_ci(
                dvals, rng=rng, n_bootstrap=n_bootstrap, alpha=alpha
            )
            pvalue = _paired_permutation_pvalue(
                dvals, rng=rng, n_permutations=n_permutations
            )

            rows.append(
                {
                    "budget_n": int(budget),
                    "metric": metric,
                    "n_pairs": n_pairs,
                    "manual_mean": float(np.mean(mvals)),
                    "manual_std": float(np.std(mvals, ddof=1)) if n_pairs > 1 else 0.0,
                    "manual_ci_lo": manual_ci_lo,
                    "manual_ci_hi": manual_ci_hi,
                    "gpt_mean": float(np.mean(gvals)),
                    "gpt_std": float(np.std(gvals, ddof=1)) if n_pairs > 1 else 0.0,
                    "gpt_ci_lo": gpt_ci_lo,
                    "gpt_ci_hi": gpt_ci_hi,
                    "delta_mean": float(np.mean(dvals)),
                    "delta_std": float(np.std(dvals, ddof=1)) if n_pairs > 1 else 0.0,
                    "delta_ci_lo": delta_ci_lo,
                    "delta_ci_hi": delta_ci_hi,
                    "p_value": pvalue,
                    "delta_ci_excludes_zero": bool(delta_ci_lo > 0.0 or delta_ci_hi < 0.0),
                }
            )

    if not rows:
        raise RuntimeError("No summary rows were generated.")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["metric", "budget_n"]).reset_index(drop=True)

    summary["p_value_adj"] = _benjamini_hochberg(summary["p_value"].tolist())
    summary["reject_h0_fdr"] = summary["p_value_adj"] < float(alpha)

    return summary


def main() -> None:
    args = _parse_args()

    metrics = _parse_csv_list(args.metrics)
    pair_keys = _parse_csv_list(args.pair_keys)
    shared_budgets = _parse_int_list(args.shared_budgets)

    manual_path = Path(args.manual_per_run)
    gpt_path = Path(args.gpt_per_run)
    if not manual_path.exists():
        raise FileNotFoundError(f"Manual per-run CSV not found: {manual_path}")
    if not gpt_path.exists():
        raise FileNotFoundError(f"GPT per-run CSV not found: {gpt_path}")

    manual_df = pd.read_csv(manual_path)
    gpt_df = pd.read_csv(gpt_path)

    paired_df = _merge_paired_runs(
        manual_df=manual_df,
        gpt_df=gpt_df,
        pair_keys=pair_keys,
        metrics=metrics,
        shared_budgets=shared_budgets,
    )

    summary_df = _summarize(
        paired_df=paired_df,
        metrics=metrics,
        alpha=float(args.alpha),
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        random_seed=int(args.random_seed),
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    paired_path = output_prefix.with_name(f"{output_prefix.name}_paired.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")

    paired_df.to_csv(paired_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote paired rows: {paired_path}")
    print(f"Wrote summary stats: {summary_path}")


if __name__ == "__main__":
    main()
