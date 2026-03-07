# Phase 1 - Model-Side Bottleneck Checks

## Item 1: Threshold/metric checks
- **Status:** completed
- **Goal:** Tune per-label thresholds for F1 on the validation set and recompute the test F1 using those frozen thresholds.

### Manual Labels Checkpoint
- **Command:**
```powershell
python .\scripts\optimize_thresholds.py --run-dir outputs/optuna_manual_labels_studies/best_model_evaluation_redo/s123

```

* **Results:**
* Validation F1-Macro (Optimized): $0.6325$
* Test F1-Macro (Frozen Thresholds on `FINAL_TEST.csv`): $0.5785$
* Phase 0 Baseline Test F1-Macro: $\sim 0.6107$

### GPT Labels Checkpoint

* **Command:**

```powershell
python .\scripts\optimize_thresholds.py --run-dir .\outputs\optuna_gpt_labels_studies\best_model_evaluation\best_model_seed_123\2026-02-27_14-43-16\ --test-manifest-dir data\manifests\manual

```

* **Results:**
* Validation F1-Macro (Optimized): $0.5832$
* Test F1-Macro (Frozen Thresholds on `test_manual_valid.csv`): $0.5576$
* Phase 0 Baseline Test F1-Macro: $\sim 0.5237$

## Phase 1 Conclusion

The threshold optimization process successfully maximized F1-macro on the validation sets for both models. However, when these optimized decision boundaries were applied to the held-out test sets, the performance gains evaporated (a drop for the manual model from $\sim 0.6107$ to $0.5785$, and a marginal $\sim+0.03$ increase for the GPT model).

Because we do not observe the large and consistent test-set F1 improvement required to classify optimization as the dominant bottleneck (and instead observe validation overfitting on the manual labels), **Hypothesis D (Optimization/threshold bottleneck)** is not supported as the primary explanation under this protocol. The results are more consistent with **Hypothesis A (Representation bottleneck)** in the frozen-feature setting. We proceed to the scaling study (Phase 2) under the assumption that feature representation quality is the main limiting factor.

## Item 2: Linear Probe Representation Check
- **Status:** completed
- **Goal:** Quantify how much performance comes from nonlinear head capacity versus linear separability of frozen features.

### Configuration
- **Model config:** `configs/model/linear_probe.yaml`
- **Hydra override:** `model=linear_probe`
- **Definition:** `hidden_dims=[]` in the MLP config, which yields a single linear classifier layer.

### Recommended Matrix
- **Sources:** manual, gpt
- **Budgets:** 100, 500, 1191 (shared budgets)
- **Seeds:** 42, 123, 456, 789, 999

### Command Pattern (Manual)
```powershell
python -m src.classification.train --config-name best_manual_labels_config.yaml model=linear_probe data.auto_split.enabled=false data.train_manifest=manual_budget_splits/train_n<BUDGET>_s<SEED>.csv data.val_manifest=manual_budget_splits/train_n<BUDGET>_s<SEED>_val.csv hydra.job.name=manual_budget_linear_probe hydra.run.dir=outputs/manual_budget_linear_probe/train_n<BUDGET>_s<SEED> utils.seed=<SEED>
```

### Command Pattern (GPT)
```powershell
python -m src.classification.train --config-name best_gpt_labels_config.yaml model=linear_probe data.auto_split.enabled=false data.train_manifest=gpt_budget_splits/all_n<BUDGET>_s<SEED>.csv data.val_manifest=gpt_budget_splits/all_n<BUDGET>_s<SEED>_val.csv hydra.job.name=gpt_budget_linear_probe hydra.run.dir=outputs/gpt_budget_linear_probe/train_n<BUDGET>_s<SEED> utils.seed=<SEED>
```

### Aggregation Commands
```powershell
python .\scripts\evaluate_and_aggregate_runs.py --runs-root outputs/manual_budget_linear_probe --test-manifest-dir data/manifests/manual --test-manifests FINAL_TEST.csv --source manual_linear_probe --output-prefix manual_linear_probe
python .\scripts\evaluate_and_aggregate_runs.py --runs-root outputs/gpt_budget_linear_probe --test-manifest-dir data/manifests/manual --test-manifests FINAL_TEST.csv --source gpt_linear_probe --output-prefix gpt_linear_probe
```

### Decision Rules
- If linear-probe metrics are close to MLP metrics across budgets, representation quality is likely the main ceiling.
- If linear probe is substantially lower, nonlinear head capacity is a non-trivial contributor.
- If GPT vs manual ranking changes under linear probe, treat bottleneck impact as high and update crossover claims accordingly.

### One-Command Runner
```powershell
.\scripts\run_phase1_linear_probe_sweep.ps1 -Source both -StopOnError
```

### Execution
```powershell
.\scripts\run_phase1_linear_probe_sweep.ps1 -Source both -StopOnError
```

### Aggregated Results (Linear Probe, `FINAL_TEST.csv`)

| Source | Budget | AUPRC (Mean +- Std) | AUROC (Mean +- Std) | F1-macro (Mean +- Std) | Seeds |
| --- | --- | --- | --- | --- | --- |
| Manual | 100 | 0.5207 +- 0.0192 | 0.6986 +- 0.0161 | 0.5547 +- 0.0113 | 5 |
| Manual | 500 | 0.5510 +- 0.0179 | 0.7194 +- 0.0177 | 0.5699 +- 0.0150 | 5 |
| Manual | 1191 | 0.5821 +- 0.0185 | 0.7440 +- 0.0152 | 0.5980 +- 0.0151 | 5 |
| GPT | 100 | 0.5188 +- 0.0143 | 0.6851 +- 0.0077 | 0.5514 +- 0.0126 | 5 |
| GPT | 500 | 0.5381 +- 0.0096 | 0.6937 +- 0.0094 | 0.5442 +- 0.0191 | 5 |
| GPT | 1191 | 0.5522 +- 0.0077 | 0.7067 +- 0.0037 | 0.5638 +- 0.0146 | 5 |

### Item 2 Conclusion

- Manual labels outperform GPT labels at all shared budgets under a strict linear probe.
- No GPT-over-manual crossover is observed in the tested range (100, 500, 1191).
- The manual-minus-GPT F1 gap increases with budget: +0.0033 (100), +0.0257 (500), +0.0341 (1191).
- Since the source ranking remains unchanged after collapsing head capacity to a single linear layer, head nonlinearity is not the primary driver of the Phase 2 ranking.