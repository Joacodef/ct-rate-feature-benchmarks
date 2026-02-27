# Phase 2 — Scaling Law Study (Core Objective)

## Item 0: Generate Budget Manifests
- **Status:** planned
- **Goal:** Create reproducible training subsets for both label sources across 5 seeds ($42, 123, 456, 789, 999$).

### Manual Labels Manifest Generation
- **Budgets:** $100, 250, 500, 800, 1000$
- **Command:**
  ```powershell
  python .\scripts\generate_budget_manifests.py --train-manifest-path data\manifests\manual\test_manual_train.csv --val-manifest-path data\manifests\manual\test_manual_valid.csv --budgets 100,250,500,800,1000 --seeds 42,123,456,789,999 --output-subdir manual_budget_splits
  ```

### GPT Labels Manifest Generation

* **Budgets:** $100, 250, 500, 800, 1000, 2000, 5000, 10000, 20000, 40000$
* **Command:**
```powershell
python .\scripts\generate_budget_manifests.py --train-manifest-path data\manifests\gpt\train.csv --val-manifest-path data\manifests\gpt\valid.csv --budgets 100,250,500,800,1000,2000,5000,10000,20000,40000 --seeds 42,123,456,789,999 --output-subdir gpt_budget_splits

```



## Item 1: Label-budget ladder (Training Execution)

* **Status:** planned
* **Goal:** Train independent models on the generated subsets using the optimal hyperparameters found in Phase 0. Note: You will need to write a simple PowerShell loop (similar to `evaluate_stability.ps1`) to iterate over the generated budgets and seeds.

### Training Sweep (Manual labels)

* **Command Pattern (Inside a loop):**
```powershell
python -m src.classification.train --config-name best_manual_labels_config.yaml data.train_manifest=manual_budget_splits/test_manual_train_n<BUDGET>_s<SEED>.csv hydra.job.name=manual_budget hydra.run.dir=outputs/manual_budget/train_n<BUDGET>_s<SEED>
```



### Training Sweep (GPT labels)

* **Command Pattern (Inside a loop):**
```powershell
python -m src.classification.train --config-name best_gpt_labels_config.yaml data.train_manifest=gpt_budget_splits/train_n<BUDGET>_s<SEED>.csv hydra.job.name=gpt_budget hydra.run.dir=outputs/gpt_budget/train_n<BUDGET>_s<SEED>
```



## Item 2: Fit scaling trends (Evaluation & Aggregation)

* **Status:** planned
* **Goal:** Evaluate the final checkpoint of every budget run on the expert test sets and aggregate the metrics (mean $\pm$ std) to find the crossover point.

### Evaluation & Aggregation (Manual labels)

* **Command:**
```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root .\outputs\manual_budget --test-manifest-dir .\data\manifests\manual\ --source manual --output-prefix manual_labels
```


* **Expected Outputs:** `outputs/aggregated_results/manual_labels_per_run.csv` and `outputs/aggregated_results/manual_labels_by_budget.csv`

### Evaluation & Aggregation (GPT labels)

* **Command:**
```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root .\outputs\gpt_budget --test-manifest-dir .\data\manifests\manual\ --source gpt --output-prefix gpt_labels
```


* **Expected Outputs:** `outputs/aggregated_results/gpt_labels_per_run.csv` and `outputs/aggregated_results/gpt_labels_by_budget.csv`

## Phase 2 Conclusion

* **Status:** pending execution
