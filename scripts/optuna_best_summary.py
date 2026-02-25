"""Summarize the best trial from an Optuna study output directory.

The script scans trial folders, reads validation metrics and trial config,
selects the best trial by AUPRC (fallback AUROC), and prints a concise
summary with key hyperparameters.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple


def read_latest_metrics(trial_dir: str) -> Optional[Dict[str, Any]]:
    """Read the latest validation metrics for a trial directory.

    Args:
        trial_dir: Path to a trial directory.

    Returns:
        Parsed metric dictionary, or ``None`` if metrics cannot be loaded.

    Logic:
        Resolve ``latest_run.json`` at the trial root first, then inside
        immediate subdirectories, and return key validation metrics.
    """
    p = os.path.join(trial_dir, "latest_run.json")
    if not os.path.exists(p):
        # Fall back to dated run subfolders.
        for name in os.listdir(trial_dir):
            candidate = os.path.join(trial_dir, name, "latest_run.json")
            if os.path.exists(candidate):
                p = candidate
                break
    try:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        return {
            "auprc": j.get("best_val_auprc"),
            "auroc": j.get("best_val_auroc"),
            "f1_macro": j.get("best_val_f1_macro"),
            "primary_metric": j.get(
                "primary_metric",
                "auprc" if j.get("best_val_auprc") is not None else "auroc",
            ),
        }
    except Exception:
        return None


def read_config_yaml(trial_dir: str) -> Dict[str, Any]:
    """Read trial config from Hydra output, with regex fallback.

    Args:
        trial_dir: Path to a trial directory.

    Returns:
        Parsed config dictionary. Returns an empty dictionary when not found.

    Logic:
        Resolve ``.hydra/config.yaml`` at the trial root or dated subfolders,
        parse with PyYAML when available, then fallback to regex extraction for
        common hyperparameters.
    """
    # Resolve .hydra/config.yaml.
    hydra = os.path.join(trial_dir, ".hydra", "config.yaml")
    if not os.path.exists(hydra):
        # Fall back to dated run subfolders.
        for name in os.listdir(trial_dir):
            candidate = os.path.join(trial_dir, name, ".hydra", "config.yaml")
            if os.path.exists(candidate):
                hydra = candidate
                break
    if not os.path.exists(hydra):
        return {}
    try:
        import yaml

        with open(hydra, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        # Fallback: regex extraction for common parameters.
        params = {}
        with open(hydra, "r", encoding="utf-8") as f:
            s = f.read()
        m = re.search(r"learning_rate:\s*([0-9.eE+-]+)", s)
        if m:
            params["training"] = params.get("training", {})
            params["training"]["learning_rate"] = float(m.group(1))
        m = re.search(r"weight_decay:\s*([0-9.eE+-]+)", s)
        if m:
            params["training"] = params.get("training", {})
            params["training"]["weight_decay"] = float(m.group(1))
        m = re.search(r"dropout:\s*([0-9.eE+-]+)", s)
        if m:
            params["model"] = params.get("model", {})
            params["model"]["params"] = params["model"].get("params", {})
            params["model"]["params"]["dropout"] = float(m.group(1))
        return params


def summarize(study_dir: str, top_k: int = 10, rel_band: float = 0.01) -> None:
    """Print a summary for the best trial in a study directory.

    Args:
        study_dir: Path to an Optuna study directory containing trial folders.

    Returns:
        ``None``.

    Logic:
        Discover trial folders, load metrics and configs, rank trials by
        AUPRC with AUROC fallback, and print metrics plus key hyperparameters.
    """
    # 1) Discover candidate trial folders.
    items = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
    trials = [d for d in items if re.search(r"_t\d+$", d)]
    results = []

    # 2) Load trial metrics and config.
    for t in trials:
        td = os.path.join(study_dir, t)
        metrics = read_latest_metrics(td)
        cfg = read_config_yaml(td)
        results.append((t, metrics, cfg))

    # Keep only trials with usable metrics.
    results = [r for r in results if r[1] is not None]
    if not results:
        print(f"No usable trials found in {study_dir}")
        return

    # 3) Select best trial by AUPRC (fallback AUROC).
    def metric_value(item: Tuple[str, Dict[str, Any], Dict[str, Any]]) -> float:
        metrics = item[1] or {}
        value = metrics.get("auprc")
        if value is None:
            value = metrics.get("auroc")
        return float(value) if value is not None else float("-inf")

    ranked = sorted(results, key=metric_value, reverse=True)
    best = ranked[0]

    def get_path(d: Dict[str, Any], *keys: str) -> Any:
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def summarize_stability(
        ranked_trials: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
        top_k: int,
        rel_band: float,
    ) -> Dict[str, Any]:
        values = [metric_value(item) for item in ranked_trials]
        values = [v for v in values if v != float("-inf")]
        if not values:
            return {}

        k = min(top_k, len(values))
        top_values = values[:k]
        best_value = values[0]
        threshold = best_value * (1.0 - rel_band)
        elite = [item for item in ranked_trials if metric_value(item) >= threshold]

        hidden_counter = Counter(
            str(get_path(item[2], "model", "params", "hidden_dims")) for item in elite
        )
        dropout_counter = Counter(
            str(get_path(item[2], "model", "params", "dropout")) for item in elite
        )

        return {
            "n": len(values),
            "mean": mean(values),
            "median": median(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "top_k": k,
            "top_k_mean": mean(top_values),
            "top_k_median": median(top_values),
            "top_k_std": pstdev(top_values) if len(top_values) > 1 else 0.0,
            "top_k_min": min(top_values),
            "top_k_max": max(top_values),
            "best_minus_top_k_median": best_value - median(top_values),
            "elite_threshold": threshold,
            "elite_count": len(elite),
            "elite_hidden_mode": hidden_counter.most_common(1)[0] if hidden_counter else None,
            "elite_dropout_mode": dropout_counter.most_common(1)[0] if dropout_counter else None,
        }

    stability = summarize_stability(ranked, top_k=top_k, rel_band=rel_band)

    # 4) Print summary output.
    print(f"Study: {os.path.basename(study_dir)}")
    print(f" Best trial: {best[0]}")
    best_metrics = best[1] or {}
    if best_metrics.get("auprc") is not None:
        print(f" Best val AUPRC: {float(best_metrics['auprc']):.6f}")
    if best_metrics.get("auroc") is not None:
        print(f" Best val AUROC: {float(best_metrics['auroc']):.6f}")
    if best_metrics.get("f1_macro") is not None:
        print(f" Best val F1-macro: {float(best_metrics['f1_macro']):.6f}")
    cfg = best[2]

    # Read a small set of likely hyperparameters.
    lr = get_path(cfg, "training", "learning_rate")
    wd = get_path(cfg, "training", "weight_decay")
    dropout = get_path(cfg, "model", "params", "dropout")
    hidden = get_path(cfg, "model", "params", "hidden_dims")
    bs = get_path(cfg, "training", "batch_size")
    print(" Hyperparameters:")
    if lr is not None:
        print(f"  - learning_rate: {lr}")
    if wd is not None:
        print(f"  - weight_decay: {wd}")
    if dropout is not None:
        print(f"  - dropout: {dropout}")
    if hidden is not None:
        print(f"  - hidden_dims: {hidden}")
    if bs is not None:
        print(f"  - batch_size: {bs}")

    if stability:
        print(" Stability (objective across trials):")
        print(
            f"  - usable_trials: {stability['n']} | mean={stability['mean']:.6f} "
            f"median={stability['median']:.6f} std={stability['std']:.6f}"
        )
        print(
            f"  - top_{stability['top_k']}: mean={stability['top_k_mean']:.6f} "
            f"median={stability['top_k_median']:.6f} std={stability['top_k_std']:.6f} "
            f"min={stability['top_k_min']:.6f} max={stability['top_k_max']:.6f}"
        )
        print(
            f"  - peak_gap(best-top_{stability['top_k']}_median): "
            f"{stability['best_minus_top_k_median']:.6f}"
        )
        print(
            f"  - near_best_band: >= {stability['elite_threshold']:.6f} "
            f"({stability['elite_count']} trials)"
        )
        if stability["elite_hidden_mode"] is not None:
            mode_val, mode_count = stability["elite_hidden_mode"]
            print(f"  - near_best hidden_dims mode: {mode_val} ({mode_count} trials)")
        if stability["elite_dropout_mode"] is not None:
            mode_val, mode_count = stability["elite_dropout_mode"]
            print(f"  - near_best dropout mode: {mode_val} ({mode_count} trials)")
    print("")


def main() -> None:
    """CLI entrypoint for Optuna best-trial summary.

    Returns:
        ``None``.

    Logic:
        Resolve target study directory from CLI args (with default path),
        validate existence, and print a best-trial summary.
    """
    parser = argparse.ArgumentParser(
        description="Summarize best Optuna trial and stability diagnostics."
    )
    parser.add_argument(
        "study_dir",
        nargs="?",
        default="",
        help="Path to study directory containing trial folders.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K trials used for stability summary statistics.",
    )
    parser.add_argument(
        "--rel-band",
        type=float,
        default=0.01,
        help="Near-best relative band (e.g., 0.01 keeps trials within 1%% of best metric).",
    )
    args = parser.parse_args()

    base = args.study_dir
    if not os.path.isabs(base):
        base = os.path.abspath(base)
    if not os.path.exists(base):
        print("Path not found:", base)
        return
    summarize(base, top_k=max(1, args.top_k), rel_band=max(0.0, args.rel_band))


if __name__ == "__main__":
    main()
