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

* **Best trial:** `optuna_mlp_t0161`
* **Best val AUPRC:** $0.6374$
* **Best val AUROC:** $0.7334$
* **Best val F1-macro:** $0.5654$

**Hyperparameters:**

* `learning_rate`: $0.0044$
* `weight_decay`: $2.389 \times 10^{-5}$
* `dropout`: $0.465$
* `hidden_dims`: $[512, 64]$
* `batch_size`: $64$

**Stability (objective across trials):**

* `usable_trials`: 200 | mean = $0.6086$, median = $0.6132$, std = $0.0184$
* `top_10`: mean = $0.6321$, median = $0.6316$, std = $0.0031$, min = $0.6283$, max = $0.6374$

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
| AUPRC | $0.5282$ | $0.0132$ | 5 |
| AUROC | $0.6605$ | $0.0140$ | 5 |
| F1-macro | $0.5276$ | $0.0163$ | 5 |

### Aggregated Results (GPT Labels Test Set)

| Metric | Mean | Std | Count |
| --- | --- | --- | --- |
| AUPRC | $0.6017$ | $0.0178$ | 5 |
| AUROC | $0.7036$ | $0.0082$ | 5 |
| F1-macro | $0.5237$ | $0.0075$ | 5 |

## Phase 0 Conclusion

The hyperparameter optimization successfully identified robust configurations for both manual and GPT-generated labels. The stability evaluation on the test set yields the following comparative insights:

* **Performance:** The model trained on GPT labels outperforms the manual labels model on AUPRC ($0.6017$ vs $0.5282$) and AUROC ($0.7036$ vs $0.6605$).
* **F1-Macro:** Both models perform similarly in terms of F1-macro ($0.5237$ for GPT vs $0.5276$ for Manual).
* **Stability:** The GPT-trained model demonstrates lower variance across the 5 evaluation seeds for both AUROC ($\pm 0.0082$ vs $\pm 0.0140$) and F1-macro ($\pm 0.0075$ vs $\pm 0.0163$), indicating a more robust learning dynamic under its optimal hyperparameters.

These hyperparameters establish the optimized baselines for the subsequent feature evaluation phases.