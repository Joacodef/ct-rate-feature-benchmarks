Source package for experiments and training utilities.

This package contains the core code for loading feature datasets, model definitions, and the main training/evaluation workflow.

Contents

- `loops.py` — reusable training and evaluation loops plus AUROC metric helpers.
- `models/` — model architecture implementations (MLP, probes, etc.).
- `train.py` — training orchestration (Hydra-configured entry point).
- `evaluate.py` — checkpoint-driven evaluation entry point.

This package depends on shared modules under `common/`:

- `common/data/dataset.py` — the `FeatureDataset` loader.
- `common/utils/seed.py` — deterministic seeding utilities.
- `common/eval/checkpointing.py` — checkpoint location helpers.

Overview

The `train.py` script orchestrates dataset creation, model instantiation (via Hydra), training loops, evaluation, and checkpoint saving. It uses PyTorch for modeling and training and expects configuration values from the repository `configs/` directory (see the `hydra.main` decorator in `train.py`).

Quick start

1. Create a virtual environment and install the package in editable mode. The project uses `pyproject.toml` for dependency information.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt  # if present
```

2. Run the training entry point. When executed as a module, Hydra will load configs from the top-level `configs/` directory (the `train.py` Hydra decorator points to `../../configs`). Run from the project root.

```powershell
python -m classification.train
```

You can override Hydra config values on the command line, for example:

```powershell
python -m classification.train training.batch_size=64 training.max_epochs=10
```

What this does

- Loads manifests and feature files using `common.data.dataset.FeatureDataset`.
- Instantiates a model via Hydra configuration and adjusts output size to match the number of target labels.
- Trains with PyTorch, computes AUROC as the primary metric, and saves a final checkpoint to the configured checkpoint directory.
- Persists a `last_state.pt` training snapshot each epoch so interrupted runs can resume where they left off.

Resuming training

- Set `training.resume.enabled=true` to resume from the most recent `last_state.pt` in `paths.checkpoint_dir` (or the path recorded in `paths.latest_run_pointer`).
- Provide `training.resume.state_path=/path/to/last_state.pt` to resume from a specific snapshot.
- RNG, optimizer, and best-metric tracking are restored automatically before training continues.

Configuration

- The canonical configuration lives under the repository `configs/` directory (see `pyproject.toml` for project metadata). `train.py` expects `config.yaml` by default; other config nodes (paths, training, model, etc.) are used at runtime.

Testing

Run unit and integration tests with `pytest` from the repository root:

```powershell
pytest -q
```

Notes and recommendations

- Requires Python 3.8+ and a compatible PyTorch build for your platform.
- GPU support is used automatically when available.
- The `utils/` module is a planned location for helpers (seeding, metrics helpers, logging wrappers).

Import examples

```python
from common.data.dataset import FeatureDataset
from classification.models import MLP
```

If you need a more detailed developer guide (tests, code style, expanding models/data), add a `DEVELOPER.md` or extend this README.

