"""Shared helpers for locating evaluation artifacts and checkpoints."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError

log = logging.getLogger(__name__)


def resolve_checkpoint_dir(cfg: DictConfig) -> str:
    """Return an absolute checkpoint directory for the provided config."""
    try:
        checkpoint_dir = os.path.normpath(cfg.paths.checkpoint_dir)
    except (AttributeError, InterpolationKeyError):
        job_name = OmegaConf.select(cfg, "hydra.job.name", default="manual_run")
        checkpoint_dir = os.path.normpath(os.path.join("outputs", job_name, "checkpoints"))
        log.warning(
            "hydra.job.name not available; falling back to checkpoint directory: %s",
            checkpoint_dir,
        )
    return checkpoint_dir


def ensure_eval_run_dir_override(argv: Optional[List[str]] = None) -> None:
    """Ensure evaluation writes under the resolved run directory."""
    args = sys.argv if argv is None else argv
    if any(str(arg).startswith("hydra.run.dir=") for arg in args[1:]):
        return

    def _value(arg: str) -> str:
        return arg.split("=", 1)[1].strip("\"'") if "=" in arg else ""

    job_name: Optional[str] = None
    outputs_root = "outputs"
    pointer_override: Optional[str] = None
    checkpoint_override: Optional[str] = None

    for arg in args[1:]:
        if arg.startswith("hydra.job.name="):
            job_name = _value(arg)
        elif arg.startswith("paths.outputs_root="):
            outputs_root = _value(arg)
        elif arg.startswith("paths.latest_run_pointer="):
            pointer_override = _value(arg)
        elif arg.startswith("paths.checkpoint_path="):
            checkpoint_override = _value(arg)

    run_dir: Optional[Path] = None

    if checkpoint_override:
        checkpoint_path = Path(checkpoint_override)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if checkpoint_path.exists():
            checkpoint_parent = checkpoint_path.parent
            run_dir = checkpoint_parent.parent if checkpoint_parent.name.lower() == "checkpoints" else checkpoint_parent

    if run_dir is None:
        pointer_path: Optional[Path] = None
        if pointer_override:
            pointer_path = Path(pointer_override)
            if not pointer_path.is_absolute():
                pointer_path = Path.cwd() / pointer_path
        elif job_name:
            pointer_path = Path(outputs_root) / job_name / "latest_run.json"
            if not pointer_path.is_absolute():
                pointer_path = Path.cwd() / pointer_path

        if pointer_path and pointer_path.exists():
            try:
                payload = json.loads(pointer_path.read_text())
            except Exception:
                payload = {}

            run_dir_str = payload.get("run_dir")
            checkpoint_from_pointer = payload.get("checkpoint")

            if run_dir_str:
                candidate = Path(run_dir_str)
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                if job_name and job_name.lower() not in {part.lower() for part in candidate.parts}:
                    candidate = None
                if candidate is not None:
                    run_dir = candidate

            if run_dir is None and checkpoint_from_pointer:
                candidate = Path(checkpoint_from_pointer)
                run_dir = candidate.parent.parent

    if run_dir is None and job_name:
        candidate_root = Path(outputs_root) / job_name
        if not candidate_root.is_absolute():
            candidate_root = Path.cwd() / candidate_root
        if candidate_root.exists():
            run_dir_candidates = sorted(
                [p for p in candidate_root.iterdir() if p.is_dir() and p.name.lower() not in {"evaluation", ".hydra"}],
                key=lambda p: p.stat().st_mtime,
            )
            if run_dir_candidates:
                run_dir = run_dir_candidates[-1]

    if run_dir is not None:
        eval_root = (Path(run_dir).resolve() / "evaluation").as_posix()
        args.append(f"hydra.run.dir={eval_root}/${{now:%Y-%m-%d_%H-%M-%S}}")
        return

    fallback_root = (Path(outputs_root) / job_name / "evaluation") if job_name else Path("outputs/evaluation")
    args.append(f"hydra.run.dir={fallback_root.as_posix()}/${{now:%Y-%m-%d_%H-%M-%S}}")


def _original_cwd(cfg: DictConfig) -> Path:
    base_dir = OmegaConf.select(cfg, "hydra.runtime.cwd", default=os.getcwd())
    return Path(base_dir)


def _as_path(cfg: DictConfig, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = _original_cwd(cfg) / path
    return path


def _checkpoint_from_pointer(cfg: DictConfig) -> Optional[str]:
    pointer_str = OmegaConf.select(cfg, "paths.latest_run_pointer")
    if not pointer_str:
        return None

    pointer_path = _as_path(cfg, pointer_str)
    if not pointer_path.exists():
        return None

    try:
        payload = json.loads(pointer_path.read_text())
    except Exception as exc:
        log.warning("Failed to decode latest run pointer %s: %s", pointer_path, exc)
        return None

    checkpoint = payload.get("checkpoint")
    if checkpoint:
        checkpoint_path = _as_path(cfg, checkpoint)
        if checkpoint_path.exists():
            log.info("Resolved checkpoint via latest run pointer: %s", checkpoint_path)
            return str(checkpoint_path)

    run_dir = payload.get("run_dir")
    if run_dir:
        checkpoint_name = OmegaConf.select(cfg, "evaluation.checkpoint_name", default="final_model.pt")
        candidate = _as_path(cfg, os.path.join(run_dir, "checkpoints", checkpoint_name))
        if candidate.exists():
            log.info("Resolved checkpoint via run_dir pointer: %s", candidate)
            return str(candidate)

    return None


def resolve_evaluation_checkpoint(cfg: DictConfig) -> str:
    explicit_path = OmegaConf.select(cfg, "paths.checkpoint_path")
    if explicit_path:
        return str(_as_path(cfg, explicit_path))

    pointer_checkpoint = _checkpoint_from_pointer(cfg)
    if pointer_checkpoint:
        return pointer_checkpoint

    checkpoint_dir = resolve_checkpoint_dir(cfg)
    checkpoint_name = OmegaConf.select(cfg, "evaluation.checkpoint_name", default="final_model.pt")
    return str(Path(checkpoint_dir) / checkpoint_name)
