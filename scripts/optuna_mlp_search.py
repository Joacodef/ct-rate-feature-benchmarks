"""Run Optuna hyperparameter search for the MLP classifier.

The script samples MLP architecture/training hyperparameters, launches
`classification.train` jobs via subprocess, and optimizes the study using
validation AUPRC (fallback AUROC) read from each trial output.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import optuna
from omegaconf import OmegaConf


def _normalize_config_name(config_name: Optional[str]) -> Optional[str]:
    """Normalize a Hydra config name, accepting accidental file paths.

    Args:
        config_name: Raw config value passed from CLI.

    Returns:
        Basename suitable for Hydra ``--config-name`` usage, or ``None``.

    Logic:
        Hydra ``--config-name`` expects a name relative to ``config_path``.
        If a file path is provided (e.g., ``./configs/foo.yaml``), reduce it
        to basename (``foo.yaml``) so downstream subprocess calls succeed.
    """
    if not config_name:
        return None

    raw = config_name.strip().strip("\"'")
    # Accept either POSIX or Windows path separators.
    normalized = Path(raw.replace("\\", "/")).name
    return normalized or raw


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
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Total target number of finished trials for the study across all runs.",
    )
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
        default=3,
        help="Maximum number of hidden layers.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="64,128,256,512,1024",
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
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=None,
        help="Optional seed for Optuna's TPE sampler to make trial suggestions reproducible.",
    )
    parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Disable Optuna pruning and always run full trials.",
    )
    parser.set_defaults(disable_pruning=None)
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=None,
        help="MedianPruner startup trial count before pruning is considered.",
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=None,
        help="MedianPruner warmup epochs per trial before pruning is considered.",
    )
    parser.add_argument(
        "--pruner-interval-steps",
        type=int,
        default=None,
        help="MedianPruner pruning check interval in epochs.",
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
        candidate_paths = [
            Path(config_name),
            Path("configs") / config_name,
        ]
        for config_path in candidate_paths:
            if not config_path.exists():
                continue
            try:
                cfg = OmegaConf.load(config_path)
                resolved = OmegaConf.select(cfg, "paths.outputs_root", default="outputs")
                if resolved:
                    return Path(str(resolved))
            except Exception:
                continue

    # 3) Final default.
    return Path("outputs")


def _resolve_optuna_seed(
    cli_seed: Optional[int],
    config_name: Optional[str],
    overrides: List[str],
) -> Optional[int]:
    """Resolve Optuna sampler seed from CLI, overrides, config, or default.

    Args:
        cli_seed: Explicit seed passed via ``--optuna-seed``.
        config_name: Optional Hydra config filename.
        overrides: Hydra-style CLI overrides.

    Returns:
        Integer seed when provided, else ``None``.

    Logic:
        Priority order is CLI flag, ``optuna.seed=...`` override, then
        ``optuna.seed`` in config file.
    """
    if cli_seed is not None:
        return int(cli_seed)

    for override in overrides:
        if override.startswith("optuna.seed="):
            value = override.split("=", 1)[1].strip("\"'")
            if value:
                return int(value)

    if config_name:
        candidate_paths = [
            Path(config_name),
            Path("configs") / config_name,
        ]
        for config_path in candidate_paths:
            if not config_path.exists():
                continue
            try:
                cfg = OmegaConf.load(config_path)
                resolved = OmegaConf.select(cfg, "optuna.seed", default=None)
                if resolved is not None:
                    return int(resolved)
            except Exception:
                continue

    return None


def _load_optuna_section(config_name: Optional[str]) -> Dict[str, object]:
    """Load the ``optuna`` section from the provided config file.

    Args:
        config_name: Optional Hydra config filename under ``configs``.

    Returns:
        Parsed ``optuna`` mapping or an empty dict when unavailable.
    """
    if not config_name:
        return {}

    candidate_paths = [
        Path(config_name),
        Path("configs") / config_name,
    ]
    for config_path in candidate_paths:
        if not config_path.exists():
            continue
        try:
            cfg = OmegaConf.load(config_path)
            section = OmegaConf.select(cfg, "optuna", default={})
            if isinstance(section, dict):
                return section
            try:
                section_container = OmegaConf.to_container(section, resolve=True)
                if isinstance(section_container, dict):
                    return section_container
            except Exception:
                continue
        except Exception:
            continue

    return {}


def _parse_bool(value: object) -> bool:
    """Parse bool-like values from CLI/config sources.

    Args:
        value: Arbitrary value.

    Returns:
        Parsed boolean.

    Raises:
        ValueError: If ``value`` cannot be parsed as boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean value: {value}")


def _resolve_optuna_pruning_settings(
    cli_disable_pruning: Optional[bool],
    cli_startup_trials: Optional[int],
    cli_warmup_steps: Optional[int],
    cli_interval_steps: Optional[int],
    config_name: Optional[str],
    overrides: List[str],
) -> Tuple[bool, int, int, int]:
    """Resolve pruning settings from CLI, overrides, config, or defaults.

    Args:
        cli_disable_pruning: Value from ``--disable-pruning`` (or ``None``).
        cli_startup_trials: Value from ``--pruner-startup-trials``.
        cli_warmup_steps: Value from ``--pruner-warmup-steps``.
        cli_interval_steps: Value from ``--pruner-interval-steps``.
        config_name: Optional Hydra config filename.
        overrides: Hydra-style CLI overrides.

    Returns:
        Tuple ``(disable_pruning, startup_trials, warmup_steps, interval_steps)``.
    """
    config_optuna = _load_optuna_section(config_name)

    disable_pruning = False
    startup_trials = 10
    warmup_steps = 5
    interval_steps = 1

    if "disable_pruning" in config_optuna:
        disable_pruning = _parse_bool(config_optuna["disable_pruning"])
    if "pruner_startup_trials" in config_optuna:
        startup_trials = int(config_optuna["pruner_startup_trials"])
    if "pruner_warmup_steps" in config_optuna:
        warmup_steps = int(config_optuna["pruner_warmup_steps"])
    if "pruner_interval_steps" in config_optuna:
        interval_steps = int(config_optuna["pruner_interval_steps"])

    for override in overrides:
        if override.startswith("optuna.disable_pruning="):
            disable_pruning = _parse_bool(override.split("=", 1)[1].strip("\"'"))
        elif override.startswith("optuna.pruner_startup_trials="):
            startup_trials = int(override.split("=", 1)[1].strip("\"'"))
        elif override.startswith("optuna.pruner_warmup_steps="):
            warmup_steps = int(override.split("=", 1)[1].strip("\"'"))
        elif override.startswith("optuna.pruner_interval_steps="):
            interval_steps = int(override.split("=", 1)[1].strip("\"'"))

    if cli_disable_pruning is not None:
        disable_pruning = bool(cli_disable_pruning)
    if cli_startup_trials is not None:
        startup_trials = int(cli_startup_trials)
    if cli_warmup_steps is not None:
        warmup_steps = int(cli_warmup_steps)
    if cli_interval_steps is not None:
        interval_steps = int(cli_interval_steps)

    return disable_pruning, startup_trials, warmup_steps, interval_steps


def _study_outputs_root(outputs_root: Path, study_name: str) -> Path:
    """Build a study-scoped outputs path under the configured outputs root.

    Args:
        outputs_root: Base outputs directory.
        study_name: Optuna study name.

    Returns:
        Path in the form ``<outputs_root>/<safe_study_name>``.

    Logic:
        Sanitize the study name to keep the resulting directory name safe and
        portable across platforms.
    """
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", study_name.strip())
    safe_name = safe_name.strip("._-") or "study"
    return outputs_root / safe_name


def _build_pruner(
    disable_pruning: bool,
    startup_trials: int,
    warmup_steps: int,
    interval_steps: int,
) -> optuna.pruners.BasePruner:
    """Build the Optuna pruner used for the study.

    Args:
        disable_pruning: Whether pruning should be disabled.
        startup_trials: Number of initial complete trials before pruning.
        warmup_steps: Number of initial reporting steps per trial before pruning.
        interval_steps: Step interval for pruning checks.

    Returns:
        Configured Optuna pruner instance.
    """
    if disable_pruning:
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=max(0, int(startup_trials)),
        n_warmup_steps=max(0, int(warmup_steps)),
        interval_steps=max(1, int(interval_steps)),
    )


def _extract_report_metric(metrics_row: Dict[str, object]) -> Optional[float]:
    """Extract a pruning/reporting metric from one metrics.jsonl row.

    Args:
        metrics_row: Parsed metrics row emitted by training.

    Returns:
        Best available validation metric (AUPRC preferred, fallback AUROC),
        or ``None`` if unavailable.
    """
    if "best_val_auprc" in metrics_row:
        return float(metrics_row.get("best_val_auprc", 0.0))

    val_metrics = metrics_row.get("val_metrics")
    if isinstance(val_metrics, dict):
        if "auprc" in val_metrics:
            return float(val_metrics.get("auprc", 0.0))
        if "auroc" in val_metrics:
            return float(val_metrics.get("auroc", 0.0))

    if "best_val_auroc" in metrics_row:
        return float(metrics_row.get("best_val_auroc", 0.0))

    return None


def _read_new_metrics_rows(metrics_path: Path, offset: int) -> Tuple[List[Dict[str, object]], int]:
    """Read newly appended metrics rows from a JSONL file.

    Args:
        metrics_path: Path to ``metrics.jsonl``.
        offset: Byte offset from which to continue reading.

    Returns:
        Tuple of parsed row list and updated byte offset.
    """
    if not metrics_path.exists():
        return [], offset

    rows: List[Dict[str, object]] = []
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
            new_offset = handle.tell()
        return rows, new_offset
    except Exception:
        return [], offset


def _resolve_trial_metrics_path(outputs_root: Path, job_name: str) -> Optional[Path]:
    """Resolve a trial's metrics.jsonl path using latest_run pointer metadata.

    Args:
        outputs_root: Root directory containing trial job folders.
        job_name: Hydra job name for the trial run.

    Returns:
        Path to ``metrics.jsonl`` when discoverable, else ``None``.
    """
    pointer_path = outputs_root / job_name / "latest_run.json"
    if not pointer_path.exists():
        return None
    try:
        payload = json.loads(pointer_path.read_text())
    except Exception:
        return None

    run_dir = payload.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir:
        return None

    return Path(run_dir) / "metrics.jsonl"


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
    args.config_name = _normalize_config_name(args.config_name)
    sizes = _parse_sizes(args.sizes)

    if args.min_depth < 1 or args.max_depth < args.min_depth:
        raise ValueError("Invalid depth range.")

    # 2) Initialize outputs and Optuna study.
    outputs_root = _resolve_outputs_root(args.config_name, args.overrides)
    study_outputs_root = _study_outputs_root(outputs_root, args.study_name)
    study_outputs_root.mkdir(parents=True, exist_ok=True)
    overrides_without_outputs_root = [
        override for override in args.overrides if not override.startswith("paths.outputs_root=")
    ]
    sampler_seed = _resolve_optuna_seed(
        cli_seed=args.optuna_seed,
        config_name=args.config_name,
        overrides=args.overrides,
    )
    disable_pruning, pruner_startup_trials, pruner_warmup_steps, pruner_interval_steps = (
        _resolve_optuna_pruning_settings(
            cli_disable_pruning=args.disable_pruning,
            cli_startup_trials=args.pruner_startup_trials,
            cli_warmup_steps=args.pruner_warmup_steps,
            cli_interval_steps=args.pruner_interval_steps,
            config_name=args.config_name,
            overrides=args.overrides,
        )
    )
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    pruner = _build_pruner(
        disable_pruning=disable_pruning,
        startup_trials=pruner_startup_trials,
        warmup_steps=pruner_warmup_steps,
        interval_steps=pruner_interval_steps,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
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
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
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
                f"paths.outputs_root={study_outputs_root.as_posix()}",
                f"model.params.hidden_dims={_format_hidden_dims(dims)}",
                f"model.params.dropout={dropout}",
                f"training.learning_rate={learning_rate}",
                f"training.weight_decay={weight_decay}",
            ]
        )
        cmd.extend(overrides_without_outputs_root)

        # Execute training, stream metrics for pruning decisions, then return objective value.
        trial_env = dict(os.environ)
        trial_env["CT_RATE_OPTUNA_RUN"] = "1"
        process = subprocess.Popen(cmd, env=trial_env)

        metrics_offset = 0
        last_reported_epoch = -1
        metrics_path: Optional[Path] = None

        while True:
            if metrics_path is None:
                metrics_path = _resolve_trial_metrics_path(study_outputs_root, job_name)

            if metrics_path is not None:
                rows, metrics_offset = _read_new_metrics_rows(metrics_path, metrics_offset)
                for row in rows:
                    epoch = row.get("epoch")
                    if not isinstance(epoch, int):
                        continue
                    metric = _extract_report_metric(row)
                    if metric is None:
                        continue
                    if epoch <= last_reported_epoch:
                        continue

                    trial.report(metric, step=epoch)
                    last_reported_epoch = epoch
                    if trial.should_prune():
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=15)
                        raise optuna.TrialPruned(
                            f"Pruned trial {trial.number} at epoch {epoch} with metric {metric:.6f}"
                        )

            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(1.0)

        if returncode != 0:
            return 0.0

        return _read_best_trial_value(study_outputs_root, job_name)

    # 3) Run study optimization and print summary.
    finished_states = (
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    )
    finished_trials = len(study.get_trials(deepcopy=False, states=finished_states))
    remaining_trials = max(args.n_trials - finished_trials, 0)

    print(
        f"Study '{args.study_name}': target_finished_trials={args.n_trials}, "
        f"already_finished={finished_trials}, to_run_now={remaining_trials}"
    )

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("Target already reached; no new trials will be created.")

    best = study.best_trial
    print("Best value:", best.value)
    print("Best params:", best.params)


if __name__ == "__main__":
    main()
