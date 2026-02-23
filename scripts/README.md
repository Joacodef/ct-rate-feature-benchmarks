# Scripts

Utility entry points that sit alongside the core training package. Run them from the project root so Hydra configuration and relative paths resolve correctly.

## analyze_alignment.py

Computes retrieval metrics between precomputed visual and text features referenced by a manifest. The script:

- Loads feature tensors via `common.data.dataset.FeatureDataset`.
- Normalizes feature vectors and builds a similarity matrix.
- Generates an instance-level ground-truth mask by trimming the reconstruction suffix from identifiers such as `train_2_a_1`.
- Reports Recall@K and paired/unpaired similarity statistics.
- Optionally emits "semantic" metrics (prefixed with `semantic_`) when the active config exposes `data.columns.labels`; these treat any pair with matching label vectors as a correct retrieval.

Example:

```powershell
python .\scripts\analyze_alignment.py data.train_manifest="train_medium.csv"
```

Relevant config knobs:

- `data.columns.visual_feature`, `data.columns.text_feature`: manifest columns storing feature paths (text column can be absent; the loader will fall back to visual-only).
- `data.columns.labels`: list of label columns for semantic metrics.
- `data.columns.grouping_col`: grouping string (defaults to `volumename`).

## prepare_manifests.py

Helper for generating or transforming manifest CSVs prior to training. It operates on the data under `paths.data_root` and writes the resulting manifests to `paths.manifest_dir`. Because the script is environment-specific, review the inline comments before running it on new datasets.

Typical invocation:

```powershell
python .\scripts\prepare_manifests.py --input-root data/features --output-dir data/manifests
```

> Adjust arguments to match your storage layout; the script does not modify configuration files automatically.

## optuna_mlp_search.py

Runs Optuna hyperparameter search by launching repeated training jobs through the classification entry point.

- Samples MLP depth/hidden dimensions plus learning-rate, dropout, and weight decay.
- Uses each trial's latest_run.json to score the objective.
- Reads best_val_auprc as the primary trial objective and falls back to best_val_auroc for older runs.

Example:

```powershell
python .\scripts\optuna_mlp_search.py --study-name mlp_hidden_dims --n-trials 20
```

## optuna_best_summary.py

Summarizes the best trial from an Optuna outputs directory.

- Selects the best trial by AUPRC when available (fallback AUROC for older trial folders).
- Prints best AUPRC, AUROC, and F1-macro (if present) plus key hyperparameters from the trial config.

Example:

```powershell
python .\scripts\optuna_best_summary.py outputs/optuna_manual_labels_trials
```
