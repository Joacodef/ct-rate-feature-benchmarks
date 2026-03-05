# Phase 2 — Scaling Law Study (Core Objective)

## ACTION NEEDED: REDO SCALING LAW STUDY FOR MANUAL LABELS, REDO CONCLUSIONS

## Item 0: Generate Budget Manifests
- **Status:** planned
- **Goal:** Create reproducible training subsets for both label sources across 5 seeds ($42, 123, 456, 789, 999$).

### Notes on the budget subsets generated

- **Fixed train/val CSVs are generated per `(budget, seed)`** using `--generate-val-csvs --val-fraction 0.1`.
- **Auto-split is disabled during training in this phase** (`data.auto_split.enabled=false`) so validation is fixed and reproducible.
- **Case alignment across label sources** is enforced via a shared `selection_index.csv` (same selected `volumename` values for the overlapping budgets).

### Reproducibility contract (shared budgets)

- Shared budgets (`100,250,500,800,1000`) must compare **same features/cases**, only labels differ (manual vs GPT).
- Larger GPT-only budgets (`2000+`) can be sampled independently because there is no manual counterpart at those sizes.

### Manual Labels Manifest Generation
- **Budgets:** $20, 50, 100, 250, 500, 800, 1191$
- **Command:**
  ```powershell
  python .\scripts\generate_budget_manifests.py --train-manifest-path data\manifests\manual\test_manual_train.csv --generate-val-csvs --val-fraction 0.1 --target-labels "Arterial wall calcification,Lymphadenopathy,Lung nodule,Lung opacity,Pulmonary fibrotic sequela" --budgets 20,50,100,250,500,800,1191 --sampling-strategy random --seeds 42,123,456,789,999 --output-subdir manual_budget_splits --selection-export-path data\manifests\manual\manual_budget_splits\selection_index.csv
  ```

### GPT Labels Manifest Generation

* **Budgets:** $20, 50, 100, 250, 500, 800, 1071, 2000, 5000, 10000, 20000, 46438$
* **Command:**
```powershell
python .\scripts\generate_budget_manifests.py --train-manifest-path data\manifests\gpt\all.csv --generate-val-csvs --val-fraction 0.1 --target-labels "Arterial wall calcification,Lymphadenopathy,Lung nodule,Lung opacity,Pulmonary fibrotic sequela" --budgets 20,50,100,250,500,800,1071,2000,5000,10000,20000,46438 --sampling-strategy random --seeds 42,123,456,789,999 --output-subdir gpt_budget_splits --selection-source-path data\manifests\manual\manual_budget_splits\selection_index.csv

```

## Item 1: Label-budget ladder (Training Execution)

* **Status:** planned

### Training Sweep (Manual labels)

* **Command Pattern (Inside a loop):**
```powershell
python -m src.classification.train --config-name best_manual_labels_config.yaml data.auto_split.enabled=false data.train_manifest=manual_budget_splits/test_manual_train_n<BUDGET>_s<SEED>.csv data.val_manifest=manual_budget_splits/test_manual_train_n<BUDGET>_s<SEED>_val.csv hydra.job.name=manual_budget hydra.run.dir=outputs/manual_budget/train_n<BUDGET>_s<SEED>
```

### Training Sweep (GPT labels)

* **Command Pattern (Inside a loop):**
```powershell
python -m src.classification.train --config-name best_gpt_labels_config.yaml data.auto_split.enabled=false data.train_manifest=gpt_budget_splits/train_n<BUDGET>_s<SEED>.csv data.val_manifest=gpt_budget_splits/train_n<BUDGET>_s<SEED>_val.csv hydra.job.name=gpt_budget hydra.run.dir=outputs/gpt_budget/train_n<BUDGET>_s<SEED>
```

## Item 2: Fit scaling trends (Evaluation & Aggregation)

* **Status:** planned
* **Goal:** Evaluate the final checkpoint of every budget run on the expert test sets and aggregate the metrics (mean $\pm$ std) to find the crossover point.

### Evaluation & Aggregation (Manual labels)

* **Command:**
```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root .\outputs\manual_budget --test-manifest-dir .\data\manifests\manual\ --source manual --output-prefix manual_labels
```

### Evaluation & Aggregation (GPT labels)

* **Command:**
```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root .\outputs\gpt_budget --test-manifest-dir .\data\manifests\manual\ --source gpt --output-prefix gpt_labels
```

## Phase 2 Conclusion

* **Status:** pending execution
