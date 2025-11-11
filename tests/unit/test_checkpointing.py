"""Unit tests for evaluation checkpoint resolution utilities."""

import json
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from common.eval import checkpointing


def _cfg(data: dict) -> DictConfig:
    return DictConfig(OmegaConf.create(data))


def test_resolve_checkpoint_dir_fallback():
    cfg = _cfg({"hydra": {"job": {"name": "demo"}}})
    resolved = checkpointing.resolve_checkpoint_dir(cfg)
    assert os.path.normpath(resolved).endswith(os.path.normpath("outputs/demo/checkpoints"))


def test_ensure_eval_run_dir_respects_existing():
    args = ["prog", "hydra.run.dir=already-set"]
    checkpointing.ensure_eval_run_dir_override(args)
    assert len([arg for arg in args if arg.startswith("hydra.run.dir=")]) == 1


def test_ensure_eval_run_dir_uses_checkpoint_override(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "job" / "2024-01-01_00-00-00"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model.pt"
    checkpoint.touch()

    args = ["prog", "hydra.job.name=job", f"paths.checkpoint_path={checkpoint}"]
    checkpointing.ensure_eval_run_dir_override(args)

    hydra_args = [arg for arg in args if arg.startswith("hydra.run.dir=")]
    assert hydra_args
    base = (run_dir.resolve() / "evaluation").as_posix()
    assert hydra_args[-1].split("=")[1].startswith(base)


def test_ensure_eval_run_dir_uses_latest_pointer(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "job" / "run-1"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model.pt"
    checkpoint.touch()

    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({"run_dir": str(run_dir), "checkpoint": str(checkpoint)}))

    args = ["prog", "hydra.job.name=job", f"paths.latest_run_pointer={pointer}"]
    checkpointing.ensure_eval_run_dir_override(args)

    hydra_args = [arg for arg in args if arg.startswith("hydra.run.dir=")]
    assert hydra_args
    expected_prefix = (run_dir.resolve() / "evaluation").as_posix()
    assert hydra_args[-1].split("=")[1].startswith(expected_prefix)


def test_ensure_eval_run_dir_chooses_most_recent(tmp_path: Path):
    job_root = tmp_path / "outputs" / "jobA"
    run_old = job_root / "old_run"
    run_new = job_root / "new_run"
    run_old.mkdir(parents=True)
    run_new.mkdir(parents=True)

    os.utime(run_old, (1, 1))
    os.utime(run_new, None)

    args = ["prog", "hydra.job.name=jobA", f"paths.outputs_root={str(tmp_path / 'outputs')}"]
    checkpointing.ensure_eval_run_dir_override(args)

    hydra_args = [arg for arg in args if arg.startswith("hydra.run.dir=")]
    assert hydra_args
    expected_prefix = (run_new.resolve() / "evaluation").as_posix()
    assert hydra_args[-1].split("=")[1].startswith(expected_prefix)


def test_resolve_evaluation_checkpoint_explicit(tmp_path: Path):
    cfg = _cfg({
        "hydra": {"runtime": {"cwd": str(tmp_path)}},
        "paths": {"checkpoint_path": "checkpoints/model.pt"},
    })
    resolved = checkpointing.resolve_evaluation_checkpoint(cfg)
    assert resolved == str((tmp_path / "checkpoints" / "model.pt").resolve())


def test_resolve_evaluation_checkpoint_from_pointer(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({"checkpoint": str(checkpoint)}))

    cfg = _cfg({
        "hydra": {"runtime": {"cwd": str(tmp_path)}},
        "paths": {"latest_run_pointer": str(pointer)},
    })

    resolved = checkpointing.resolve_evaluation_checkpoint(cfg)
    assert resolved == str(checkpoint.resolve())


def test_resolve_evaluation_checkpoint_from_run_dir(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "demo" / "runA"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "alt.pt"
    checkpoint.touch()

    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({"run_dir": str(run_dir)}))

    cfg = _cfg({
        "hydra": {"runtime": {"cwd": str(tmp_path)}},
        "paths": {"latest_run_pointer": str(pointer)},
        "evaluation": {"checkpoint_name": "alt.pt"},
    })

    resolved = checkpointing.resolve_evaluation_checkpoint(cfg)
    assert resolved == str(checkpoint.resolve())


def test_resolve_evaluation_checkpoint_default_dir(tmp_path: Path):
    checkpoint_dir = tmp_path / "manual"
    cfg = _cfg({
        "paths": {"checkpoint_dir": str(checkpoint_dir)},
        "evaluation": {"checkpoint_name": "final.pt"},
    })

    resolved = checkpointing.resolve_evaluation_checkpoint(cfg)
    assert resolved == str(checkpoint_dir / "final.pt")
