# 📦 Source (`src/`)

This directory contains the main Python source code for the project, organized as an installable package named `ct_rate_benchmarks`.

## Purpose

The code within this directory is considered "production-level" code, as opposed to experimental notebooks or standalone scripts. It contains all the core logic for the project:

* **`data/`**: Modules for loading, processing, and serving data (e.g., `FeatureDataset`).
* **`models/`**: Definitions for all model architectures (e.g., `LinearProbe`, `MLP`).
* **`train.py`**: The main training and evaluation script, which is the primary entry point for all experiments.
* **`utils/`**: (To be added) Helper functions and utilities (e.g., seeding, metrics calculation).

## 🐍 Package Structure

When you install the project using `uv pip install -e .`, this directory is installed as a package. This allows you to run the main training script as a module:

```bash
python -m ct_rate_benchmarks.train
````

It also allows the `tests/` directory to import from this package using absolute imports, such as:

```python
from ct_rate_benchmarks.data.dataset import FeatureDataset
```

