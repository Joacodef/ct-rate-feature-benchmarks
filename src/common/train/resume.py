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
    """Resolve a possibly relative path using Hydra runtime working directory.

    Args:
        cfg: Hydra/OmegaConf configuration that may include
            ``hydra.runtime.cwd``.
        path_str: Absolute or relative filesystem path string.

    Input:
        Any path value from config/CLI overrides where relative paths should be
        interpreted from the original run root.

    Returns:
        Absolute, normalized, and resolved ``Path``.

    Logic:
        1. Read ``hydra.runtime.cwd`` with fallback to current process CWD.
        2. Normalize incoming path text.
        3. If relative, anchor to the chosen base directory.
        4. Return resolved absolute path.
    """
    base_dir = OmegaConf.select(cfg, "hydra.runtime.cwd", default=os.getcwd())
    path = Path(os.path.normpath(path_str))
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def capture_rng_state() -> Dict[str, Any]:
    """Capture random number generator state across supported libraries.

    Input:
        No explicit parameters; reads global RNG state from Python, NumPy,
        Torch CPU, and optionally Torch CUDA.

    Returns:
        Dictionary containing serialized RNG snapshots with keys:
        - ``python``
        - ``numpy``
        - ``torch``
        - ``torch_cuda`` (only when CUDA is available)

    Logic:
        Gather deterministic replay state for each backend so resumed training
        can continue with equivalent stochastic behavior.
    """
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """Restore RNG state captured by ``capture_rng_state`` on a best-effort basis.

    Args:
        state: Optional dictionary with RNG snapshots for Python/NumPy/Torch
            and optional CUDA state.

    Input:
        Serialized RNG state dictionary, typically loaded from a checkpoint.

    Returns:
        ``None``.

    Logic:
        1. Exit early when state is empty/missing.
        2. For each backend, attempt restore inside isolated try/except.
        3. Log warnings instead of raising so resume can proceed even if one
           backend state is incompatible.
        4. Restore CUDA RNG only when CUDA is available.
    """
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
    """Load Torch checkpoint data with cross-version compatibility handling.

    Args:
        path: Path to checkpoint file.

    Input:
        Any Torch-serializable checkpoint payload.

    Returns:
        Deserialized checkpoint object (type depends on saved content).

    Logic:
        1. Prefer ``torch.load(..., weights_only=False)`` to load full payloads
           on newer Torch versions.
        2. If that signature is unsupported, retry with legacy call signature.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older Torch versions do not expose ``weights_only``
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """Write Torch payload to disk atomically via temporary-file replacement.

    Args:
        obj: Serializable object to save with ``torch.save``.
        path: Final checkpoint destination path.

    Input:
        Arbitrary checkpoint payload and target location.

    Returns:
        ``None``.

    Logic:
        1. Ensure destination directory exists.
        2. Save to a temp file in the same directory.
        3. Replace destination with ``os.replace`` for atomic swap semantics.
        4. On failure, best-effort delete temp file and re-raise.
    """
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
    best_val_auprc: float,
    best_val_auroc: float,
    epochs_no_improve: int,
    best_model_state: Optional[Any],
) -> None:
    """Persist complete training resume state to checkpoint storage.

    Args:
        path: Destination checkpoint path.
        epoch: Current epoch index to persist.
        model: Model instance whose ``state_dict`` is saved.
        optimizer: Optimizer instance whose ``state_dict`` is saved.
        best_val_auprc: Best validation AUPRC tracked so far.
        best_val_auroc: Best validation AUROC tracked so far.
        epochs_no_improve: Early-stopping counter value.
        best_model_state: Optional in-memory best-model snapshot.

    Input:
        Training loop state needed to resume optimization and metrics tracking.

    Returns:
        ``None``.

    Logic:
        1. Build a versioned payload containing model/optimizer states, scalar
           metrics/counters, best model snapshot, and RNG state.
        2. Mirror ``best_val_auprc`` into ``best_primary_metric`` for backward
           compatibility with callers expecting a primary metric key.
        3. Persist payload with atomic write to avoid partial checkpoints.
    """
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_auprc": float(best_val_auprc),
        "best_val_auroc": float(best_val_auroc),
        "best_primary_metric": float(best_val_auprc),
        "epochs_no_improve": int(epochs_no_improve),
        "best_model_state": best_model_state,
        "rng_state": capture_rng_state(),
        "version": 2,
    }
    _atomic_torch_save(payload, path)


def resolve_resume_state(
    cfg: DictConfig,
    resume_enabled: bool,
    default_state_path: Path,
    override_path: Optional[str],
) -> Optional[Path]:
    """Resolve the resume checkpoint path using overrides and fallbacks.

    Args:
        cfg: Hydra/OmegaConf configuration object.
        resume_enabled: Whether automatic resume behavior is enabled.
        default_state_path: Default checkpoint path for the active run layout.
        override_path: Optional explicit resume path override.

    Input:
        Config-driven path hints, optional latest-run pointer JSON, and local
        resume policy flag.

    Returns:
        Resolved checkpoint ``Path`` when one can be inferred, otherwise
        ``None``.

    Logic:
        Precedence order:
        1. Use explicit ``override_path`` when provided and resolvable.
        2. If resume is enabled, read ``paths.latest_run_pointer`` and extract
           ``state_checkpoint`` when available.
        3. If still unresolved and resume is enabled, fall back to
           ``default_state_path``.
        4. Log warnings for resolution/parsing errors but avoid raising.
    """
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
    """Load resume checkpoint payload and return both state and attempted path.

    Args:
        cfg: Hydra/OmegaConf configuration object.
        resume_enabled: Whether resume behavior is enabled.
        default_state_path: Default checkpoint location.
        override_path: Optional explicit checkpoint override path.

    Input:
        Resume policy and path resolution hints used to locate a state file.

    Returns:
        Tuple ``(state, state_path)`` where:
        - ``state`` is the loaded checkpoint dictionary when successful,
          otherwise ``None``.
        - ``state_path`` is the resolved path attempted (or ``None`` if no path
          could be resolved).

    Logic:
        1. Resolve candidate path via ``resolve_resume_state``.
        2. If unresolved, return ``(None, None)``.
        3. If resolved but missing on disk, warn when resume is enabled and
           return ``(None, state_path)``.
        4. Attempt load with ``torch_load_full``; on failure warn and return
           ``(None, state_path)``.
    """
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
