"""Unit tests for common.train.resume utilities."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from common.train import resume


def _cfg(data: dict) -> DictConfig:
    return DictConfig(OmegaConf.create(data))


def test_resolve_path_relative(tmp_path: Path):
    cfg = _cfg({"hydra": {"runtime": {"cwd": str(tmp_path)}}})
    resolved = resume.resolve_path(cfg, os.path.join("checkpoints", "model.pt"))
    assert resolved == (tmp_path / "checkpoints" / "model.pt").resolve()


def test_capture_and_restore_rng_state_roundtrip():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    state = resume.capture_rng_state()

    py_val = random.random()
    np_val = float(np.random.rand())
    torch_val = torch.rand(1)

    resume.restore_rng_state(state)

    assert random.random() == pytest.approx(py_val)
    assert float(np.random.rand()) == pytest.approx(np_val)
    assert torch.allclose(torch.rand(1), torch_val)


def test_restore_rng_state_handles_errors(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(resume.random, "setstate", boom)
    monkeypatch.setattr(resume.np.random, "set_state", boom)
    monkeypatch.setattr(resume.torch, "set_rng_state", boom)
    monkeypatch.setattr(resume.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(resume.torch.cuda, "set_rng_state_all", boom)

    resume.restore_rng_state({
        "python": object(),
        "numpy": object(),
        "torch": object(),
        "torch_cuda": object(),
    })


def test_torch_load_full_fallback(monkeypatch, tmp_path: Path):
    calls = []

    def fake_load(path, map_location="cpu", weights_only=None):
        calls.append(weights_only)
        if weights_only is not None:
            raise TypeError("unexpected keyword")
        return {"ok": True}

    monkeypatch.setattr(resume.torch, "load", fake_load)

    result = resume.torch_load_full(tmp_path / "dummy.pt")

    assert result == {"ok": True}
    assert calls == [False, None]


def test_save_training_state_writes_payload(tmp_path: Path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "state.pt"

    resume.save_training_state(
        path=path,
        epoch=5,
        model=model,
        optimizer=optimizer,
        best_val_auroc=0.9,
        epochs_no_improve=2,
        best_model_state={"weights": 1},
    )

    assert path.exists()
    payload = resume.torch_load_full(path)
    assert payload["epoch"] == 5
    assert payload["best_val_auroc"] == pytest.approx(0.9)
    assert "rng_state" in payload


def test_resolve_resume_state_with_override(tmp_path: Path):
    cfg = _cfg({"hydra": {"runtime": {"cwd": str(tmp_path)}}})
    override = str(tmp_path / "override.pt")
    resolved = resume.resolve_resume_state(cfg, True, tmp_path / "default.pt", override)
    assert resolved == Path(override)


def test_resolve_resume_state_from_pointer(tmp_path: Path):
    state_file = tmp_path / "state.pt"
    state_file.touch()
    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({"state_checkpoint": "state.pt"}))

    cfg = _cfg({
        "hydra": {"runtime": {"cwd": str(tmp_path)}},
        "paths": {"latest_run_pointer": str(pointer)},
    })

    resolved = resume.resolve_resume_state(cfg, True, tmp_path / "fallback.pt", None)
    assert resolved == state_file.resolve()


def test_resolve_resume_state_defaults_to_provided(tmp_path: Path):
    default_path = tmp_path / "default.pt"
    cfg = _cfg({})
    resolved = resume.resolve_resume_state(cfg, True, default_path, None)
    assert resolved == default_path


def test_load_resume_state_missing_checkpoint(tmp_path: Path):
    default_path = tmp_path / "missing.pt"
    cfg = _cfg({})
    state, path = resume.load_resume_state(cfg, True, default_path, None)
    assert state is None
    assert path == default_path


def test_load_resume_state_success(tmp_path: Path):
    state_path = tmp_path / "resume.pt"
    torch.save({"epoch": 3}, state_path)

    cfg = _cfg({})
    state, resolved_path = resume.load_resume_state(cfg, True, state_path, None)

    assert resolved_path == state_path
    assert state["epoch"] == 3

