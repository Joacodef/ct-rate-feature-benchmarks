# CT-RATE Feature Benchmarks

Standardized benchmarking environment for training and evaluating downstream models using pre-computed visual (and optionally textual) features derived from CT chest scans and paired reports. This repo focuses on modeling with pre-extracted embeddings; it does not perform raw 3D image preprocessing.

## Repository structure

```
ct-rate-feature-benchmarks/
├─ configs/                # Hydra configuration (main config.yaml plus data/ and model/)
│  ├─ config.yaml
│  ├─ data/
│  │  └─ default_features.yaml
│  └─ model/
│     └─ mlp_visual.yaml
├─ data/                   # Local datasets area (features, manifests, labels, etc.)
│  ├─ features/
│  │  ├─ image/            # NPZ feature files (example: train_*.npz)
│  │  └─ text/             # (optional) text feature files
│  ├─ labels/
│  ├─ manifests/           # CSV manifests used by training (train.csv, valid.csv, tests)
│  ├─ splits/              # Predefined split CSVs
│  ├─ metadata/
│  └─ radiology_text_reports/
├─ scripts/
│  └─ prepare_manifests.py
├─ src/
│  └─ ct_rate_benchmarks/
│     ├─ data/
│     ├─ models/
│     └─ train.py          # Hydra-configured training entry point
├─ tests/
│  └─ unit/test_train_pipeline.py
├─ pyproject.toml
└─ README.md
```

## Installation

Use a virtual environment and install the project in editable mode from the repository root.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e .
```

## Running training

Training is configured with Hydra. The entry point is `ct_rate_benchmarks.train`, which loads `configs/config.yaml` by default. Run from the repository root so relative config paths resolve correctly.

```powershell
python -m ct_rate_benchmarks.train
```

Override configuration values on the command line as needed, for example to change the batch size, epochs, or point to your features root:

```powershell
python -m ct_rate_benchmarks.train training.batch_size=64 training.max_epochs=10 paths.data_root="E:/ct-rate-feature-benchmarks/data/features"
```

Key behaviors (from `src/ct_rate_benchmarks/train.py`):

- Loads manifests from `paths.manifest_dir` (default: `data/manifests`) using `FeatureDataset`.
- Resolves label count from `training.target_labels`.
- Instantiates the model via Hydra (see `configs/model/mlp_visual.yaml`).
- Trains with BCEWithLogitsLoss and reports AUROC (macro) on validation and test sets.
- Saves a checkpoint to `paths.checkpoint_dir` (default: `outputs/${hydra.job.name}/checkpoints/final_model.pt`).

## Configuration

Top-level config: `configs/config.yaml`.

- paths
  - `data_root`: root folder containing feature subfolders such as `features/image/` and `features/text/`.
  - `manifest_dir`: directory holding split manifests (e.g., `data/manifests`).
  - `output_dir`, `checkpoint_dir`: output and checkpoint locations.
- data
  - `train_manifest`, `val_manifest`, `test_manifests`: filenames inside `paths.manifest_dir`.
- training
  - `batch_size`, `num_workers`, `learning_rate`, `max_epochs`.
  - `loss`: default `torch.nn.BCEWithLogitsLoss`.
  - `target_labels`: list of label column names expected in the manifests.
- model
  - See `configs/model/mlp_visual.yaml` with `_target_` = `ct_rate_benchmarks.models.mlp.MLP` and constructor `params` (e.g., `in_features`, `hidden_dims`, `dropout`).

Data manifest expectations (see `configs/data/default_features.yaml`):

- Column `visual_feature_path` stores the relative path to each sample’s visual feature file, relative to `paths.data_root`.
- Label columns must match `training.target_labels`.
- Text feature support is optional and disabled by default (`text_feature: null`).

## Data layout

Place your precomputed features under `paths.data_root`. A common layout is:

```
<paths.data_root>/
├─ image/           # e.g., *.npz files with visual embeddings
└─ text/            # (optional) text embeddings
```

Provide split manifests in `paths.manifest_dir` (defaults to `data/manifests`):

- `train.csv`, `valid.csv`
- `test_manual_all.csv`, `test_manual_train.csv`, `test_manual_valid.csv`

The repository includes helper data and scripts under `data/` and `scripts/` for preparing and inspecting manifests (see `scripts/prepare_manifests.py`).

## Testing

Run the test suite from the repository root:

```powershell
pytest -q
```

## Notes

- Python 3.8+ is recommended.
- CUDA is used automatically when available.
- For guidance on the package internals, see `src/ct_rate_benchmarks/README.md`.