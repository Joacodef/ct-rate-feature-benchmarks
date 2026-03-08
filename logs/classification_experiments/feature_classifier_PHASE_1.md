# Phase 1 - Model-Side Bottleneck Checks

## Item 1: Threshold/metric checks
- **Status:** completed
- **Goal:** Tune per-label thresholds for F1 on the validation set and recompute the test F1 using those frozen thresholds.

### Checkpoint Provenance (from Phase 0 Stability Protocol)
- The checkpoints used here were not one-off trainings; they came from the Phase 0 stability sweep over seeds `42, 123, 456, 789, 999`.
- In that protocol, best Optuna hyperparameters were fixed and each seed was retrained with `utils.seed=<seed>` (and seed-specific split behavior where enabled).
- Phase 1 threshold tuning was run on one representative stability run (`seed=123`) per source.
- The "Phase 0 Baseline" values referenced below are the seed-aggregated means reported in `logs/experiments/PHASE_0.md`, not the metrics of a single run directory.

### Manual Labels Checkpoint
- **Command:**
```powershell
python .\scripts\optimize_thresholds.py --run-dir outputs/optuna_manual_labels_studies/best_model_evaluation_redo/s123

```

* **Results:**
* Validation F1-Macro (Optimized): $0.6325$
* Test F1-Macro (Frozen Thresholds on `FINAL_TEST.csv`): $0.5785$
* Phase 0 Baseline Test F1-Macro (5-seed mean): $\sim 0.6107$

### GPT Labels Checkpoint

* **Command:**

```powershell
python .\scripts\optimize_thresholds.py --run-dir .\outputs\optuna_gpt_labels_studies\best_model_evaluation\best_model_seed_123\2026-02-27_14-43-16\ --test-manifest-dir data\manifests\manual

```

* **Results:**
* Validation F1-Macro (Optimized): $0.5832$
* Test F1-Macro (Frozen Thresholds on `FINAL_TEST.csv`): $0.5963$
* Phase 0 Baseline Test F1-Macro (5-seed mean): $\sim 0.5237$

### Five-Seed Recalculation on `FINAL_TEST.csv`

- **Execution:**

```powershell
.\scripts\run_phase1_threshold_sweep.ps1 -Source both -Force -StopOnError
```

- **Aggregation artifacts:**
	- `outputs/aggregated_results/phase1_threshold_optimized_per_seed_all_metrics.csv`
	- `outputs/aggregated_results/phase1_threshold_optimized_summary_all_metrics.csv`

| Source | AUPRC (Mean +- Std) | AUROC (Mean +- Std) | F1-macro optimized (Mean +- Std) | Seeds |
| --- | --- | --- | --- | --- |
| Manual | 0.6550 +- 0.0991 | 0.7746 +- 0.0652 | 0.6181 +- 0.0538 | 5 |
| GPT | 0.5751 +- 0.0090 | 0.7157 +- 0.0044 | 0.5934 +- 0.0061 | 5 |

## Phase 1 Conclusion

Threshold optimization successfully maximized validation F1 and produced mixed test-set effects. Under the five-seed `FINAL_TEST.csv` recalculation, manual labels show a small F1 gain over the Phase 0 baseline ($0.6107 \rightarrow 0.6181$), while GPT shows a larger gain ($0.5237 \rightarrow 0.5934$).

Because improvements are not uniform across sources and manual-vs-GPT ranking remains unchanged after threshold tuning, optimization is likely a contributing factor but not a full explanation of the source gap. The evidence remains more consistent with **Hypothesis A (Representation bottleneck)** as the primary limiting factor in the frozen-feature setting.

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