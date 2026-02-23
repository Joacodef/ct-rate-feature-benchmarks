"""Run Optuna hyperparameter search for the MLP classifier.

The script samples MLP architecture/training hyperparameters, launches
`classification.train` jobs via subprocess, and optimizes the study using
validation AUPRC (fallback AUROC) read from each trial output.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import optuna
from omegaconf import OmegaConf


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Optuna search.

    Returns:
        Parsed ``argparse.Namespace`` with search-space and runtime options.

    Logic:
        Define CLI knobs for trial count, output naming, search-space bounds,
        and optional Hydra config overrides.
    """
    parser = argparse.ArgumentParser(
        description="Optuna search over MLP hidden_dims using classification.train"
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--job-prefix", type=str, default="optuna_mlp")
    parser.add_argument("--study-name", type=str, default="mlp_hidden_dims")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///outputs/optuna_mlp.db",
        help="Optuna storage URL (e.g., sqlite:///outputs/optuna_mlp.db)",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=2,
        help="Minimum number of hidden layers.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum number of hidden layers.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="64,128,256,512,1024,2048",
        help="Comma-separated hidden sizes to sample from.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hydra config name to pass to classification.train (e.g., optuna_config.yaml).",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides passed to classification.train.",
    )
    return parser.parse_args()


def _parse_sizes(size_str: str) -> List[int]:
    """Parse comma-separated hidden sizes into an integer list.

    Args:
        size_str: Comma-separated integer values.

    Returns:
        Parsed list of integer hidden sizes.

    Raises:
        ValueError: If no valid size is provided.

    Logic:
        Split by comma, trim whitespace, convert to integers, and validate
        that at least one value exists.
    """
    sizes = []
    for chunk in size_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        sizes.append(int(chunk))
    if not sizes:
        raise ValueError("--sizes must include at least one integer.")
    return sizes


def _resolve_outputs_root(config_name: Optional[str], overrides: List[str]) -> Path:
    """Resolve outputs root from overrides, config, or default.

    Args:
        config_name: Optional Hydra config filename under ``configs``.
        overrides: Hydra-style CLI overrides.

    Returns:
        Path to the outputs root directory.

    Logic:
        Prefer explicit ``paths.outputs_root=...`` overrides, then read
        ``paths.outputs_root`` from the config file, and fallback to
        ``outputs``.
    """
    # 1) Prefer explicit CLI override.
    for override in overrides:
        if override.startswith("paths.outputs_root="):
            value = override.split("=", 1)[1].strip("\"'")
            return Path(value)

    # 2) Fallback to config file value.
    if config_name:
        config_path = Path("configs") / config_name
        if config_path.exists():
            try:
                cfg = OmegaConf.load(config_path)
                resolved = OmegaConf.select(cfg, "paths.outputs_root", default="outputs")
                if resolved:
                    return Path(str(resolved))
            except Exception:
                pass

    # 3) Final default.
    return Path("outputs")


def _read_best_trial_value(outputs_root: Path, job_name: str) -> float:
    """Read the optimization objective value for a completed training job.

    Args:
        outputs_root: Root directory containing run outputs.
        job_name: Hydra job name for the trial run.

    Returns:
        Best validation AUPRC if available, otherwise best validation AUROC.
        Returns ``0.0`` when unavailable or unreadable.

    Logic:
        Load ``latest_run.json`` for the job and prioritize AUPRC as the
        primary optimization metric for backward-compatible scoring.
    """
    pointer_path = outputs_root / job_name / "latest_run.json"
    if not pointer_path.exists():
        return 0.0
    try:
        payload = json.loads(pointer_path.read_text())
        if "best_val_auprc" in payload:
            return float(payload.get("best_val_auprc", 0.0))
        return float(payload.get("best_val_auroc", 0.0))
    except Exception:
        return 0.0


def _format_hidden_dims(dims: List[int]) -> str:
    """Format hidden dimensions as a Hydra list override string.

    Args:
        dims: Hidden layer dimensions.

    Returns:
        String representation in list format (e.g., ``[512, 256]``).
    """
    return "[{}]".format(", ".join(str(dim) for dim in dims)
    )


def main() -> None:
    """CLI entrypoint for Optuna MLP hyperparameter search.

    Returns:
        ``None``.

    Logic:
        Parse inputs, validate search space, initialize study/storage, run
        trial objective evaluations through subprocess training runs, and print
        best trial summary.
    """
    # 1) Parse and validate inputs.
    args = _parse_args()
    sizes = _parse_sizes(args.sizes)

    if args.min_depth < 1 or args.max_depth < args.min_depth:
        raise ValueError("Invalid depth range.")

    # 2) Initialize outputs and Optuna study.
    outputs_root = _resolve_outputs_root(args.config_name, args.overrides)
    outputs_root.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        """Objective function evaluated by Optuna.

        Args:
            trial: Optuna trial object.

        Returns:
            Scalar objective value for the trial.

        Logic:
            Sample architecture/training hyperparameters, execute a training
            run, and read the resulting best validation metric from disk.
        """
        # Sample search-space parameters.
        depth = trial.suggest_int("depth", args.min_depth, args.max_depth)
        dims = [
            trial.suggest_categorical(f"dim_{idx}", sizes)
            for idx in range(depth)
        ]
        dims = sorted(dims, reverse=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # Build training command for this trial.
        job_name = f"{args.job_prefix}_t{trial.number:04d}"
        cmd = [
            sys.executable,
            "-m",
            "classification.train",
        ]
        if args.config_name:
            cmd.extend(["--config-name", args.config_name])
        cmd.extend(
            [
                f"hydra.job.name={job_name}",
                f"model.params.hidden_dims={_format_hidden_dims(dims)}",
                f"model.params.dropout={dropout}",
                f"training.learning_rate={learning_rate}",
                f"training.weight_decay={weight_decay}",
            ]
        )
        cmd.extend(args.overrides)

        # Execute training and return objective value.
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return 0.0

        return _read_best_trial_value(outputs_root, job_name)

    # 3) Run study optimization and print summary.
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print("Best value:", best.value)
    print("Best params:", best.params)


if __name__ == "__main__":
    main()
