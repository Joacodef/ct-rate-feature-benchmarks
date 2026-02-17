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
    for override in overrides:
        if override.startswith("paths.outputs_root="):
            value = override.split("=", 1)[1].strip("\"'")
            return Path(value)

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

    return Path("outputs")


def _read_best_val_auroc(outputs_root: Path, job_name: str) -> float:
    pointer_path = outputs_root / job_name / "latest_run.json"
    if not pointer_path.exists():
        return 0.0
    try:
        payload = json.loads(pointer_path.read_text())
        return float(payload.get("best_val_auroc", 0.0))
    except Exception:
        return 0.0


def _format_hidden_dims(dims: List[int]) -> str:
    return "[{}]".format(", ".join(str(dim) for dim in dims)
    )


def main() -> None:
    args = _parse_args()
    sizes = _parse_sizes(args.sizes)

    if args.min_depth < 1 or args.max_depth < args.min_depth:
        raise ValueError("Invalid depth range.")

    outputs_root = _resolve_outputs_root(args.config_name, args.overrides)
    outputs_root.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("depth", args.min_depth, args.max_depth)
        dims = [
            trial.suggest_categorical(f"dim_{idx}", sizes)
            for idx in range(depth)
        ]
        dims = sorted(dims, reverse=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

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

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return 0.0

        return _read_best_val_auroc(outputs_root, job_name)

    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print("Best value:", best.value)
    print("Best params:", best.params)


if __name__ == "__main__":
    main()
