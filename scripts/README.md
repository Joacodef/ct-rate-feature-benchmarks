# Scripts

Utility entry points that sit alongside the core training package. Run them from the project root so module imports and relative paths resolve correctly.

## analyze_alignment.py

Computes retrieval metrics between precomputed visual and text features referenced by a manifest. The script:

- Loads feature tensors via `common.data.dataset.FeatureDataset`.
- Normalizes feature vectors and builds a similarity matrix.
- Generates an instance-level ground-truth mask by trimming the reconstruction suffix from identifiers such as `train_2_a_1`.
- Reports extensive cross-modal retrieval metrics including Recall@$K$ ($K \in \{1, 5, 10, 50\}$), Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and Normalized Discounted Cumulative Gain (NDCG@$K$ for $K \in \{5, 10, 50\}$).
- Evaluates both visual-to-text and text-to-visual retrieval directions.
- Optionally emits "semantic" metrics (prefixed with `semantic_`) when the active config exposes `data.columns.labels`; these treat any pair with matching label vectors as a correct retrieval.

Example:

```powershell
python .\scripts\analyze_alignment.py data.train_manifest="train_medium.csv"

```

Relevant config knobs:

* `data.columns.visual_feature`, `data.columns.text_feature`: manifest columns storing feature paths (both are currently required).
* `data.columns.labels`: list of label columns for semantic metrics.
* `data.columns.grouping_col`: grouping string (defaults to `volumename`).
* `analysis.filter_normal_cases`: when `true`, excludes all-normal cases (requires labels).

## create_volumename_csv.py

Small utility script for generating a foundational manifest CSV based on available extracted feature files.

* Scans a configured input directory for `.npz` feature files.
* Generates a CSV file containing a `VolumeName` column by appending `.nii.gz` to each feature stem.
* Uses hard-coded `feature_dir` and `output_csv` paths inside the script; edit those values before running.

Example:

```powershell
python .\scripts\create_volumename_csv.py

```

## prepare_manifests.py

Helper for generating or transforming multimodal manifest CSVs prior to training. This script is CLI-driven (`argparse`) and takes explicit input/output CSV paths.

* Merges master label declarations, predefined dataset split assignments, and optionally full text reports into a unified dataset manifest.
* Generates relative paths for visual and text features based on configured directory prefixes and file extensions.
* Preserves report text columns natively to support retrieval applications and text-feature classifiers.
* Optionally expands labels to reconstruction-level keys using `--metadata_csv`.

Typical invocation:

```powershell
python .\scripts\prepare_manifests.py --labels_csv data/labels.csv --split_csv data/train_split.csv --output_file data/manifests/train.csv

```

> Adjust arguments to match your storage layout; the script does not modify configuration files automatically.

## optuna_mlp_search.py

Runs Optuna hyperparameter search by launching repeated training jobs through the classification entry point.

* Samples MLP depth/hidden dimensions plus learning-rate, dropout, and weight decay.
* Uses each trial's latest_run.json to score the objective.
* Reads best_val_auprc as the primary trial objective and falls back to best_val_auroc for older runs.
* Writes trial folders under a study subdirectory: `<outputs_root>/<study_name>/<job_prefix>_tXXXX`.
* Enables Optuna MedianPruner by default using per-epoch metrics from each trial's `metrics.jsonl`.
* Disable pruning with `--disable-pruning` or tune via `--pruner-startup-trials`, `--pruner-warmup-steps`, and `--pruner-interval-steps`.
* The same pruning knobs can be set in config under `optuna.disable_pruning`, `optuna.pruner_startup_trials`, `optuna.pruner_warmup_steps`, and `optuna.pruner_interval_steps`.

Example:

```powershell
python .\scripts\optuna_mlp_search.py --study-name mlp_hidden_dims --n-trials 20

```

## optuna_best_summary.py

Summarizes the best trial from an Optuna outputs directory.

* Selects the best trial by AUPRC when available (fallback AUROC for older trial folders).
* Prints best AUPRC, AUROC, and F1-macro (if present) plus key hyperparameters from the trial config.

Example:

```powershell
python .\scripts\optuna_best_summary.py outputs/optuna_manual_labels_trials/mlp_hidden_dims

```

## generate_budget_manifests.py

Generates multiple budgeted train manifests for scaling-law experiments while keeping validation fixed.

* Supports two input modes:
	* `--config-name`: resolves `paths.manifest_dir`, `data.train_manifest`, and `data.val_manifest` from a Hydra config.
	* Explicit paths: `--train-manifest-path` + `--val-manifest-path`.
* Creates train subsets for every `(budget, seed)` pair.
* Writes outputs to a subfolder under the manifest directory (default: `budget_splits`).
* Emits `manifest_index.csv` with the generated train/val manifest names to use in Hydra overrides.
* Supports `--sampling-strategy stratified` for multilabel prevalence preservation (requires `iterative-stratification`).
* Supports repeated copies per budget via `--seeds` to estimate uncertainty.

Typical invocation (recommended with config):

```powershell
python .\scripts\generate_budget_manifests.py --config-name optuna_manual_labels_config.yaml --budgets 100,250,500,1000,2000 --seeds 42,1234,2025 --copy-validation

```

Stratified example:

```powershell
python .\scripts\generate_budget_manifests.py --config-name optuna_manual_labels_config.yaml --budgets 100,250,500,1000,2000 --seeds 42,1234,2025 --sampling-strategy stratified --copy-validation

```

> Stratified mode dependency: `pip install iterative-stratification`

Then train with one generated split:

```powershell
python -m classification.train --config-name optuna_manual_labels_config.yaml paths.manifest_dir=data/manifests/manual/budget_splits data.train_manifest=test_manual_train_n500_s42.csv data.val_manifest=test_manual_valid.csv

```

## evaluate_and_aggregate_runs.py

Evaluates many trained run folders on test manifests and aggregates results to CSV.

* Discovers runs under `--runs-root` by finding `.hydra/config.yaml` + `checkpoints/final_model.pt`.
* Evaluates every discovered run using the saved run config and checkpoint.
* Supports overriding test manifest directory via `--test-manifest-dir` (important when train manifests live in `budget_splits`).
* Writes:
	* `<prefix>_per_run.csv` (one row per run)
	* `<prefix>_by_budget.csv` (mean/std/count by `source` and budget `n`)

GPT example:

```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root outputs/gpt_budget --test-manifest-dir data/manifests/gpt --source gpt --output-prefix gpt_budget

```

Manual example:

```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root outputs/manual_budget --test-manifest-dir data/manifests/manual --source manual --output-prefix manual_budget

```

Optional explicit test list:

```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root outputs/gpt_budget --test-manifest-dir data/manifests/gpt --test-manifests test_manual_all.csv,test_manual_train.csv,test_manual_valid.csv

```
