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
│  ├─ analyze_alignment.py
│  ├─ create_manual_label_files.py
│  ├─ create_volumename_csv.py
│  ├─ evaluate_and_aggregate_runs.py
│  ├─ generate_budget_manifests.py
│  ├─ optimize_thresholds.py
│  ├─ optuna_best_summary.py
│  ├─ optuna_mlp_search.py
│  ├─ plot_aggregated_runs.py
│  ├─ prepare_manifests.py
│  ├─ run_phase2_budget_sweep.ps1
│  └─ README.md
├─ src/
│  ├─ classification/
│  │  ├─ loops.py          # Training/eval loops & metrics
│  │  ├─ models/
│  │  ├─ train.py          # Hydra-configured classification entry point
│  │  └─ evaluate.py       # Checkpoint evaluation entry point
│  ├─ common/              # Shared data loaders, seeding, checkpoint helpers
├─ tests/
│  ├─ integration/
│  │  └─ test_training_pipeline.py
│  └─ unit/
│     ├─ test_checkpointing.py
│     ├─ test_evaluate.py
│     ├─ test_feature_dataset.py
│     ├─ test_loops.py
│     ├─ test_optuna_scripts.py
│     ├─ test_resume.py
│     └─ test_wandb_logging.py
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

Training is configured with Hydra. The entry point is `classification.train`, which loads `configs/config.yaml` by default. Run from the repository root so relative config paths resolve correctly.

```powershell
python -m classification.train
```

Override configuration values on the command line as needed, for example to change the batch size, epochs, or point to your features root:

```powershell
python -m classification.train training.batch_size=64 training.max_epochs=10 paths.data_root="data/features/CT-CLIP_v2"
```

Resume an interrupted run by enabling the resume flag or pointing directly to a saved training snapshot:

```powershell
python -m classification.train training.resume.enabled=true
# or
python -m classification.train training.resume.state_path="outputs/your_job/latest_run/checkpoints/last_state.pt"
```

Key behaviors (from `src/classification/train.py`):

- Loads manifests from `paths.manifest_dir` (default: `data/manifests`) using `FeatureDataset`.
- Resolves label count from `training.target_labels`.
- Instantiates the model via Hydra (see `configs/model/mlp_visual.yaml`).
- Trains with BCEWithLogitsLoss and optimizes validation AUPRC (macro) as the primary metric.
- Also reports validation/test AUROC (macro), F1 (macro), and per-class precision/recall/F1/support metrics.
- Saves a checkpoint to `paths.checkpoint_dir` (default: `outputs/${hydra.job.name}/checkpoints/final_model.pt`).

## Configuration

Top-level config: `configs/config.yaml`.

- paths
  - `data_root`: root folder used to resolve relative feature paths from manifests (default: `data/features/CT-CLIP_v2`).
  - `manifest_dir`: directory holding split manifests (e.g., `data/manifests`).
  - `output_dir`, `checkpoint_dir`: output and checkpoint locations.
- data
  - `train_manifest`, `val_manifest`, `test_manifests`: filenames inside `paths.manifest_dir`.
- training
  - `batch_size`, `num_workers`, `learning_rate`, `max_epochs`.
  - `loss`: default `torch.nn.BCEWithLogitsLoss`.
  - `target_labels`: list of label column names expected in the manifests.
- model
  - See `configs/model/mlp_visual.yaml` with `_target_` = `classification.models.mlp.MLP` and constructor `params` (e.g., `in_features`, `hidden_dims`, `dropout`).

Data manifest expectations (see `configs/data/default_features.yaml`):

- Column `visual_feature_path` stores the relative path to each sample’s visual feature file, relative to `paths.data_root`.
- Column `text_feature_path` is optional. Training automatically falls back to visual-only mode if the column is absent, but keep the key in the config if you plan to add text features later.
- Label columns must match `training.target_labels`.
- Text feature support is optional. The default config exposes a `text_feature_path` column, and the loader will automatically drop it if the manifest does not provide the field.

## Data layout

Place your precomputed features under `paths.data_root` (default: `data/features/CT-CLIP_v2`). The manifest columns should contain paths relative to this root. A common layout is:

```
<paths.data_root>/
├─ image/           # e.g., *.npz files with visual embeddings
└─ text/            # (optional) text embeddings
```

Provide split manifests in `paths.manifest_dir` (defaults to `data/manifests`):

- `train.csv`, `valid.csv`
- `test_manual_all.csv`, `test_manual_train.csv`, `test_manual_valid.csv`

The repository includes helper data and scripts under `data/` and `scripts/` for manifest prep, search/sweep workflows, and post-hoc analysis. See `scripts/README.md` for full per-script details.

## Scripts

Current utility scripts in `scripts/` include:

- Data and manifest preparation:
  - `prepare_manifests.py` for building training/evaluation manifests.
  - `create_volumename_csv.py` for creating base CSVs from feature filenames.
  - `create_manual_label_files.py` for generating manual-label `train.csv`/`val.csv` from accession mappings.
- Retrieval/alignment analysis:
  - `analyze_alignment.py` for visual-text retrieval metrics and optional semantic matching metrics.
- Hyperparameter search:
  - `optuna_mlp_search.py` to run Optuna sweeps over MLP/training hyperparameters.
  - `optuna_best_summary.py` to summarize the best Optuna trial in a study folder.
- Budget/scaling workflows:
  - `generate_budget_manifests.py` to create budgeted train manifests across budgets and seeds.
  - `run_phase2_budget_sweep.ps1` to orchestrate large manual/GPT budget sweeps.
  - `evaluate_and_aggregate_runs.py` to evaluate many runs and export per-run plus by-budget CSVs.
  - `plot_aggregated_runs.py` to plot scaling curves from aggregated results.
- Evaluation calibration:
  - `optimize_thresholds.py` to tune per-label validation thresholds and re-evaluate test manifests.

Example commands:

```powershell
python .\scripts\prepare_manifests.py --labels_csv data/labels.csv --split_csv data/train_split.csv --output_file data/manifests/train.csv
python .\scripts\analyze_alignment.py data.train_manifest="train_medium.csv"
python .\scripts\optuna_mlp_search.py --study-name mlp_hidden_dims --n-trials 20
python .\scripts\generate_budget_manifests.py --config-name optuna_manual_labels_config.yaml --budgets 100,250,500 --seeds 42,1234 --copy-validation
python .\scripts\evaluate_and_aggregate_runs.py --runs-root outputs/manual_budget --test-manifest-dir data/manifests/manual --source manual --output-prefix manual_budget
python .\scripts\plot_aggregated_runs.py --by-budget-csvs outputs/aggregated_results/manual_labels_by_budget.csv --metric-prefix test_manual_all.csv --output-dir outputs/aggregated_results/plots
```

## Testing

Run the test suite from the repository root:

```powershell
pytest -q
```

## Notes

- Python 3.10+ is required.
- CUDA is used automatically when available.
- For guidance on the package internals, see `src/classification/README.md`.