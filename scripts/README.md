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
