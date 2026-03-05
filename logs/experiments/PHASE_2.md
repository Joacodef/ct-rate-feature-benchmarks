# Phase 2 — Scaling Law Study (Core Objective)

## Item 0: Generate Budget Manifests
- **Status:** completed
- **Goal:** Create reproducible training subsets for both label sources across seeds.

### Notes on the budget subsets generated

- **Fixed train/val CSVs are generated per `(budget, seed)`** using `--generate-val-csvs --val-fraction 0.1`.
- **Auto-split is disabled during training in this phase** (`data.auto_split.enabled=false`) so validation is fixed and reproducible.
- **Case alignment across label sources** is enforced via a shared `selection_index.csv` (same selected `volumename` values for the overlapping budgets).

### Reproducibility contract (shared budgets)

- Shared budgets (`100,250,500,800,1000`) must compare **same features/cases**, only labels differ (manual vs GPT).
- Larger GPT-only budgets (`2000+`) can be sampled independently because there is no manual counterpart at those sizes.
- Extra seeds added since manual labels runs are quick (2344, 5678, 9012, 3456, 6789, 23423, 54321, 98765, 43210, 11111)

### Manual Labels Manifest Generation
- **Budgets:** $20, 50, 100, 250, 500, 800, 1191$
- **Command:**
  ```powershell
  python .\scripts\generate_budget_manifests.py --train-manifest-path data\manifests\manual\test_manual_train.csv --generate-val-csvs --val-fraction 0.1 --target-labels "Arterial wall calcification,Lymphadenopathy,Lung nodule,Lung opacity,Pulmonary fibrotic sequela" --budgets 20,50,100,250,500,800,1191 --sampling-strategy random --seeds 42,123,456,789,999,2344,5678,9012,3456,6789,23423,54321,98765,43210,11111 --output-subdir manual_budget_splits --selection-export-path data\manifests\manual\manual_budget_splits\selection_index.csv


### GPT Labels Manifest Generation

* **Budgets:** $20, 50, 100, 250, 500, 800, 1071, 2000, 5000, 10000, 20000, 46438$
* **Command:**

```powershell
python .\scripts\generate_budget_manifests.py --train-manifest-path data\manifests\gpt\all.csv --generate-val-csvs --val-fraction 0.1 --target-labels "Arterial wall calcification,Lymphadenopathy,Lung nodule,Lung opacity,Pulmonary fibrotic sequela" --budgets 20,50,100,250,500,800,1191,2000,5000,10000,20000,46438 --sampling-strategy random --seeds 42,123,456,789,999,2344,5678,9012,3456,6789,23423,54321,98765,43210,11111 --output-subdir gpt_budget_splits --selection-source-path data\manifests\manual\manual_budget_splits\selection_index.csv
```

## Item 1: Label-budget ladder (Training Execution)

* **Status:** completed

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

* **Status:** completed
* **Goal:** Evaluate the final checkpoint of every budget run on the expert test sets and aggregate the metrics (mean $\pm$ std) to find the crossover point.

### Evaluation & Aggregation (Manual labels)

* **Command:**

```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root .\outputs\manual_budget --test-manifest-dir .\data\manifests\manual\ --source manual --output-prefix manual_labels

```

#### Aggregated Results (Manual Labels)

| Budget | AUPRC (Mean ± Std) | AUROC (Mean ± Std) | F1-macro (Mean ± Std) | Seeds |
| --- | --- | --- | --- | --- |
| 20 | $0.4853 \pm 0.0277$ | $0.6458 \pm 0.0366$ | $0.3527 \pm 0.1423$ | 15 |
| 50 | $0.4970 \pm 0.0325$ | $0.6768 \pm 0.0276$ | $0.4736 \pm 0.0949$ | 15 |
| 100 | $0.5324 \pm 0.0317$ | $0.6989 \pm 0.0216$ | $0.5250 \pm 0.0582$ | 15 |
| 250 | $0.5987 \pm 0.0468$ | $0.7430 \pm 0.0287$ | $0.5900 \pm 0.0287$ | 15 |
| 500 | $0.5994 \pm 0.0473$ | $0.7474 \pm 0.0285$ | $0.5919 \pm 0.0308$ | 15 |
| 800 | $0.6494 \pm 0.0702$ | $0.7775 \pm 0.0469$ | $0.6143 \pm 0.0405$ | 15 |
| 1191 | $0.6716 \pm 0.0944$ | $0.7903 \pm 0.0580$ | $0.6280 \pm 0.0673$ | 15 |

### Evaluation & Aggregation (GPT labels)

* **Command:**

```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root .\outputs\gpt_budget --test-manifest-dir .\data\manifests\manual\ --source gpt --output-prefix gpt_labels
```

#### Aggregated Results (GPT Labels)

| Budget | AUPRC (Mean ± Std) | AUROC (Mean ± Std) | F1-macro (Mean ± Std) | Seeds |
| --- | --- | --- | --- | --- |
| 20 | $0.5058 \pm 0.0173$ | $0.6683 \pm 0.0174$ | $0.4366 \pm 0.0531$ | 5 |
| 50 | $0.4971 \pm 0.0199$ | $0.6582 \pm 0.0172$ | $0.5031 \pm 0.0559$ | 5 |
| 100 | $0.5048 \pm 0.0054$ | $0.6689 \pm 0.0168$ | $0.5144 \pm 0.0447$ | 5 |
| 250 | $0.5114 \pm 0.0150$ | $0.6695 \pm 0.0234$ | $0.4968 \pm 0.0561$ | 5 |
| 500 | $0.5145 \pm 0.0040$ | $0.6905 \pm 0.0071$ | $0.5213 \pm 0.0198$ | 5 |
| 800 | $0.5378 \pm 0.0151$ | $0.6991 \pm 0.0081$ | $0.5541 \pm 0.0294$ | 5 |
| 1191 | $0.5377 \pm 0.0141$ | $0.7044 \pm 0.0125$ | $0.5575 \pm 0.0110$ | 5 |
| 2000 | $0.5430 \pm 0.0105$ | $0.7078 \pm 0.0075$ | $0.5557 \pm 0.0112$ | 5 |
| 5000 | $0.5485 \pm 0.0054$ | $0.7107 \pm 0.0047$ | $0.5715 \pm 0.0107$ | 5 |
| 10000 | $0.5614 \pm 0.0133$ | $0.7147 \pm 0.0061$ | $0.5637 \pm 0.0105$ | 5 |
| 20000 | $0.5645 \pm 0.0045$ | $0.7182 \pm 0.0042$ | $0.5681 \pm 0.0030$ | 5 |
| 46438 | $0.5676 \pm 0.0124$ | $0.7169 \pm 0.0064$ | $0.5713 \pm 0.0148$ | 5 |

## Phase 2 Conclusion

The scaling law study reveals a clear dominance of expert-annotated (manual) labels over GPT-generated labels in the context of training classification heads on frozen visual features.

* **Asymptotic Ceiling:** The model trained on GPT labels reaches its performance ceiling at approximately 46,000+ samples, achieving a maximum F1-macro of $\sim 0.5713$ and an AUPRC of $\sim 0.5676$.
* **Crossover Point:** The performance curve for manual labels surpasses the asymptotic ceiling of GPT labels at an exceptionally low budget. With only $n=250$ expert-annotated samples, the model achieves an F1-macro of $0.5900$ and an AUPRC of $0.5987$, outperforming the massive GPT budget. The performance continues to scale to an F1-macro of $0.6280$ at $n=1191$.
* **Variance and Stability:** The evaluation exhibits increasing standard deviation at higher budgets, culminating in significant variance at $n=1191$ (e.g., AUPRC $\pm 0.0944$). Since these results are calculated over a fixed, relatively small final test set (258 samples), this variance does not stem from different test data, but rather reflects the model's high sensitivity to the random seeds. Random variations in weight initialization and the composition of the training/validation splits cause the model to converge to slightly different decision boundaries. When these boundaries are applied to the fixed 258-sample test set, particularly using macro-averaged metrics, minor shifts in the classification of underrepresented findings result in massive swings in the final aggregated scores.
* **Final Verdict:** The hypothesis that models trained with massive amounts of GPT-generated labels can surpass those trained on a few expert labels is rejected under the current setup. Even with the inherent limitations of the frozen CT-CLIP visual embeddings (representation bottleneck), high-quality manual labels are significantly more sample-efficient and yield a higher absolute performance ceiling.
