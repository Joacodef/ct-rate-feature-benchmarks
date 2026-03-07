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

## generate_kfold_budget_manifests.py

Generates fold-aware manifests for Phase 3 manual-label K-fold cross-validation.

* Splits a full manual-label manifest into `K` grouped multilabel-stratified folds.
* Writes one hold-out test manifest per fold.
* For each `(fold, budget)`, samples budget rows from that fold's training pool only.
* Splits each sampled subset into train/val manifests for model selection.
* Writes a `manifest_index.csv` used by the Phase 3 sweep runner.

Example:

```powershell
python .\scripts\generate_kfold_budget_manifests.py --config-name best_manual_labels_config.yaml --full-manifest-name all.csv --k-folds 5 --budgets 20,50,100,250,500,800,1191 --seed 52 --output-subdir manual_kfold_budget_splits --prefix manual_kfold

```

> Note: If a requested budget exceeds a fold's train-pool size, the script caps it and records both requested and effective budgets in `manifest_index.csv`.

## run_phase3_manual_kfold_sweep.ps1

Runs one deterministic training job per `(fold, budget)` from the generated Phase 3 index.

* Reads `manifest_index.csv` from the K-fold generator output.
* Trains with fixed seed (`utils.seed`) and fold-specific train/val manifests.
* Supports `-LabelSource manual|gpt` with source-specific defaults for config, manifest paths, run outputs, and aggregation labels.
* Stores one run folder per fold-budget pair under source-specific runs roots.
* By default runs `evaluate_and_aggregate_runs.py` at the end.

Manual example:

```powershell
.\scripts\run_phase3_manual_kfold_sweep.ps1 -LabelSource manual -Seeds 52,123,456,789,999 -StopOnError

```

GPT example:

```powershell
.\scripts\run_phase3_manual_kfold_sweep.ps1 -LabelSource gpt -Seeds 52,123,456,789,999 -StopOnError

```

Default seeds example (no `-Seed`/`-Seeds` passed):

```powershell
.\scripts\run_phase3_manual_kfold_sweep.ps1 -LabelSource gpt -StopOnError

```

This automatically runs the Phase 3 seed set: `52,123,456,789,999`.

Options:

* `-Force`: retrain completed runs.
* `-StopOnError`: stop immediately on first failed run.
* `-SkipAggregate`: skip final aggregate evaluation.
* Seed precedence: `-Seeds` (list) > `-Seed` (single) > built-in default five seeds.
* Any default can still be overridden explicitly, e.g. `-ManifestRoot` or `-ConfigName`.

## run_phase1_linear_probe_sweep.ps1

Runs the Phase 1 linear-probe diagnostic sweep over matched budgets and seeds.

* Trains with `model=linear_probe` for strict single-layer probing on frozen features.
* Supports `-Source both|manual|gpt`.
* Uses shared defaults from the study plan:
	* Budgets: `100,500,1191`
	* Seeds: `42,123,456,789,999`
* Aggregation defaults to `--test-manifest-dir data/manifests/manual --test-manifests FINAL_TEST.csv`.
* Writes runs to:
	* `outputs/manual_budget_linear_probe`
	* `outputs/gpt_budget_linear_probe`
* By default runs `evaluate_and_aggregate_runs.py` at the end for each selected source.

Run both sources:

```powershell
.\scripts\run_phase1_linear_probe_sweep.ps1 -Source both -StopOnError

```

Run only manual source with custom budgets and skip aggregation:

```powershell
.\scripts\run_phase1_linear_probe_sweep.ps1 -Source manual -Budgets 100,250,500 -Seeds 42,123,456 -SkipAggregate

```

Run with an explicit test-manifest set:

```powershell
.\scripts\run_phase1_linear_probe_sweep.ps1 -Source both -TestManifestDir data/manifests/manual -TestManifests FINAL_TEST.csv

```

Options:

* `-Force`: retrain completed runs.
* `-StopOnError`: stop immediately on first failed run.
* `-SkipAggregate`: skip final aggregate evaluation.

## evaluate_and_aggregate_runs.py

Evaluates many trained run folders on test manifests and aggregates results to CSV.

* Discovers runs under `--runs-root` by finding `.hydra/config.yaml` + `checkpoints/final_model.pt`.
* Evaluates every discovered run using the saved run config and checkpoint.
* Supports overriding test manifest directory via `--test-manifest-dir` (important when train manifests live in `budget_splits`).
* Writes:
	* `<prefix>_per_run.csv` (one row per run)
	* `<prefix>_by_budget.csv` (mean/std/count by `source` and budget `n`)
* Adds canonical columns `test_primary_auprc`, `test_primary_auroc`, and `test_primary_f1_macro` by averaging all per-run `test_*` metrics. This is important for K-fold runs where each fold has a different hold-out manifest name.

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

## plot_aggregated_runs.py

Plots scaling curves from one or more aggregated `*_by_budget.csv` files.

* Generates one figure per metric (default: AUPRC, AUROC, F1-macro).
* Uses line plots with error bars (`mean ± std`) across budgets.
* Supports multiple sources in one plot (e.g., manual + GPT) when multiple CSVs are provided.

Manual-only example:

```powershell
python .\scripts\plot_aggregated_runs.py --by-budget-csvs outputs/aggregated_results/manual_labels_by_budget.csv --metric-prefix test_manual_all.csv --output-dir outputs/aggregated_results/plots

```

Phase 3 K-fold example (uses canonical primary test metric columns):

```powershell
python .\scripts\plot_aggregated_runs.py --by-budget-csvs outputs/aggregated_results/manual_kfold_budget_by_budget.csv --metric-prefix test_primary --output-dir outputs/aggregated_results/plots

```

Manual + GPT combined example:

```powershell
python .\scripts\plot_aggregated_runs.py --by-budget-csvs outputs/aggregated_results/manual_labels_by_budget.csv outputs/aggregated_results/gpt_labels_by_budget.csv --metric-prefix test_manual_all.csv --output-dir outputs/aggregated_results/plots

```

## compare_per_class_bottlenecks.py

Compares per-class metrics between manual and GPT models using detailed metric JSON files.

* Aligns rows by `(fold, class)` from manifest names such as `manual_kfold_f1_test.csv`.
* Computes paired deltas (`manual - gpt`) for precision, recall, and F1.
* Writes:
	* `<prefix>_paired.csv` (one row per fold/class pair)
	* `<prefix>_summary.csv` (mean/std/count by class)
* Optional markdown output for reporting.

Phase 3 Item 3 example:

```powershell
python .\scripts\compare_per_class_bottlenecks.py --manual-glob "outputs/manual_kfold_budget/f*_n1191_s52/evaluation_aggregate/detailed_metrics/manual_kfold_f*_test.csv_detailed_metrics.json" --gpt-glob "outputs/gpt_budget/train_n46438_s11111/evaluation_aggregate/detailed_metrics/manual_kfold_f*_test.csv_detailed_metrics.json" --output-prefix outputs/aggregated_results/per_class_bottleneck --output-markdown outputs/aggregated_results/per_class_bottleneck_summary.md

```

> Tip: if your CSV was generated before the metric-key normalization update and still has `test_test_...` columns, `plot_aggregated_runs.py` accepts either prefix form.
