# Phase 0 — Hyperparameter Optimization and Protocol Selection

## Optuna search (Manual labels)
- **Status:** completed
- **Stage:** optuna
- **Source:** Manual labels
- **Command:**
  ```powershell
  python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_manual_labels_config.yaml --study-name manual_mlp_search --n-trials 200
  ```

* **Config:** configs/optuna_manual_labels_config.yaml
* **Outputs:** outputs/optuna_manual_labels_studies/manual_mlp_search
* **Reason:** First Optuna run for manual-label tuning.

### Summary & Best Hyperparameters

* **Best trial:** `optuna_mlp_t0182`
* **Best val AUPRC:** $0.6321$
* **Best val AUROC:** $0.7214$
* **Best val F1-macro:** $0.5813$

**Hyperparameters:**

* `learning_rate`: $0.0043$
* `weight_decay`: $1.277 \times 10^{-4}$
* `dropout`: $0.4096$
* `hidden_dims`: $[512, 256, 256]$
* `batch_size`: $64$

**Stability (objective across trials):**

* `usable_trials`: 200 | mean = $0.6029$, median = $0.6067$, std = $0.0202$
* `top_10`: mean = $0.6280$, median = $0.6282$, std = $0.0032$, min = $0.6237$, max = $0.6321$

## Optuna search (GPT labels)

* **Status:** completed
* **Stage:** optuna
* **Source:** GPT labels
* **Command:**
```powershell
python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_gpt_labels_config.yaml --study-name gpt_mlp_search --n-trials 200
```


* **Config:** configs/optuna_gpt_labels_config.yaml
* **Outputs:** outputs/optuna_gpt_labels_studies/gpt_mlp_search
* **Reason:** First Optuna run for GPT-label baseline tuning.

### Summary & Best Hyperparameters

* **Best trial:** `optuna_mlp_t0181`
* **Best val AUPRC:** $0.5493$
* **Best val AUROC:** $0.7221$
* **Best val F1-macro:** $0.5601$

**Hyperparameters:**

* `learning_rate`: $0.0019$
* `weight_decay`: $3.897 \times 10^{-4}$
* `dropout`: $0.252$
* `hidden_dims`: $[512, 512]$
* `batch_size`: $64$

**Stability (objective across trials):**

* `usable_trials`: 200 | mean = $0.5445$, median = $0.5454$, std = $0.0062$
* `top_10`: mean = $0.5484$, median = $0.5483$, std = $0.0003$, min = $0.5482$, max = $0.5493$

## Stability Evaluation

### Protocol Specification

To ensure the robustness of the best hyperparameters discovered during the Optuna search, an evaluation protocol is executed over multiple independent seeds.

* **Script:** `evaluate_stability.ps1`
* **Seeds Evaluated:** $42, 123, 456, 789, 999$
* **Output Directory:** `outputs\optuna_manual_labels_studies\best_model_evaluation`

The execution overrides the base configuration with the identified best hyperparameters and dynamically assigns the seed for both the utility modules and data splitting mechanisms to evaluate stability across distinct initializations and data splits.

### Aggregated Results (Manual Labels Test Set)

| Metric | Mean | Std | Count |
| --- | --- | --- | --- |
| AUPRC | $0.6550$ | $0.0991$ | 5 |
| AUROC | $0.7746$ | $0.0652$ | 5 |
| F1-macro | $0.6107$ | $0.0729$ | 5 |

### Aggregated Results (GPT Labels Test Set)

| Metric | Mean | Std | Count |
| --- | --- | --- | --- |
| AUPRC | $0.6017$ | $0.0178$ | 5 |
| AUROC | $0.7036$ | $0.0082$ | 5 |
| F1-macro | $0.5237$ | $0.0075$ | 5 |

## Phase 0 Conclusion

The hyperparameter optimization successfully identified robust configurations for both manual and GPT-generated labels. The stability evaluation on the test set yields the following comparative insights:

* **Performance:** The model trained on manual labels outperforms the GPT-label model on AUPRC ($0.6550$ vs $0.6017$), AUROC ($0.7746$ vs $0.7036$), and F1-macro ($0.6107$ vs $0.5237$).
* **Variance:** The manual-label evaluation shows larger seed-to-seed spread (AUPRC $\pm 0.0991$, AUROC $\pm 0.0652$, F1-macro $\pm 0.0729$) than the GPT-label baseline (AUPRC $\pm 0.0178$, AUROC $\pm 0.0082$, F1-macro $\pm 0.0075$).
* **Interpretation (Phase-0 protocol):** Under this initial split-and-seed evaluation, the manual pipeline improves average performance substantially, while indicating higher sensitivity to initialization/split randomness.

These hyperparameters establish the optimized baselines for the subsequent feature evaluation phases.