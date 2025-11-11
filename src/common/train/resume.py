"""Resume utilities shared across training pipelines."""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def resolve_path(cfg: DictConfig, path_str: str) -> Path:
    """Resolve ``path_str`` against Hydra's original working directory."""
    base_dir = OmegaConf.select(cfg, "hydra.runtime.cwd", default=os.getcwd())
    path = Path(os.path.normpath(path_str))
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def capture_rng_state() -> Dict[str, Any]:
    """Capture Python, NumPy, Torch (CPU/CUDA) RNG state."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """Restore previously captured RNG state (best-effort)."""
    if not state:
        return

    python_state = state.get("python") if isinstance(state, dict) else None
    if python_state is not None:
        try:
            random.setstate(python_state)
        except Exception as exc:
            log.warning("Failed to restore Python RNG state: %s", exc)

    numpy_state = state.get("numpy") if isinstance(state, dict) else None
    if numpy_state is not None:
        try:
            np.random.set_state(numpy_state)
        except Exception as exc:
            log.warning("Failed to restore NumPy RNG state: %s", exc)

    torch_state = state.get("torch") if isinstance(state, dict) else None
    if torch_state is not None:
        try:
            torch.set_rng_state(torch_state)
        except Exception as exc:
            log.warning("Failed to restore Torch RNG state: %s", exc)

    cuda_state = state.get("torch_cuda") if isinstance(state, dict) else None
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as exc:
            log.warning("Failed to restore CUDA RNG state: %s", exc)


def torch_load_full(path: Path):
    """Load a Torch checkpoint with compatibility fallbacks."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older Torch versions do not expose ``weights_only``
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """Atomically write a Torch checkpoint to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False, suffix=".tmp") as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup on failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def save_training_state(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_auroc: float,
    epochs_no_improve: int,
    best_model_state: Optional[Any],
) -> None:
    """Persist the current training state to ``path`` using an atomic write."""
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_auroc": float(best_val_auroc),
        "epochs_no_improve": int(epochs_no_improve),
        "best_model_state": best_model_state,
        "rng_state": capture_rng_state(),
        "version": 1,
    }
    _atomic_torch_save(payload, path)


def resolve_resume_state(
    cfg: DictConfig,
    resume_enabled: bool,
    default_state_path: Path,
    override_path: Optional[str],
) -> Optional[Path]:
    """Resolve which state checkpoint should be loaded for resuming."""
    state_path: Optional[Path] = None

    if override_path:
        try:
            state_path = resolve_path(cfg, override_path)
        except Exception as exc:
            log.warning("Failed to resolve resume.state_path '%s': %s", override_path, exc)

    if state_path is None and resume_enabled:
        pointer_path_str = OmegaConf.select(cfg, "paths.latest_run_pointer", default=None)
        if pointer_path_str:
            pointer_path = resolve_path(cfg, pointer_path_str)
            if pointer_path.exists():
                try:
                    payload = json.loads(pointer_path.read_text())
                    pointer_state = payload.get("state_checkpoint")
                    if pointer_state:
                        candidate = Path(pointer_state)
                        if not candidate.is_absolute():
                            candidate = pointer_path.parent / candidate
                        state_path = candidate.resolve()
                except Exception as exc:
                    log.warning("Failed to parse latest run pointer %s: %s", pointer_path, exc)

        if state_path is None:
            state_path = default_state_path

    return state_path


def load_resume_state(
    cfg: DictConfig,
    resume_enabled: bool,
    default_state_path: Path,
    override_path: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Load a resume state if configuration requests it."""
    state_path = resolve_resume_state(cfg, resume_enabled, default_state_path, override_path)

    if not state_path:
        return None, None

    if not state_path.exists():
        if resume_enabled:
            log.warning("Resume requested but checkpoint not found at %s", state_path)
        return None, state_path

    try:
        state = torch_load_full(state_path)
        return state, state_path
    except Exception as exc:
        log.warning("Failed to load resume checkpoint at %s: %s", state_path, exc)
        return None, state_path
