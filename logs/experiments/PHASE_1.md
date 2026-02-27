# Phase 1 - Model-Side Bottleneck Checks

## Item 1: Threshold/metric checks
- **Status:** completed
- **Goal:** Tune per-label thresholds for F1 on the validation set and recompute the test F1 using those frozen thresholds.

### Manual Labels Checkpoint
- **Command:**
  ```powershell
  python .\scripts\optimize_thresholds.py --run-dir .\outputs\optuna_manual_labels_studies\best_model_evaluation\best_model_seed_456\2026-02-25_19-25-43\

```

* **Results:**
* Validation F1-Macro (Optimized): $0.6395$
* Test F1-Macro (Frozen Thresholds on `test_manual_valid.csv`): $0.5139$
* Phase 0 Baseline Test F1-Macro: $\sim 0.5276$



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

The threshold optimization process successfully maximized F1-macro on the validation sets for both models. However, when these optimized decision boundaries were applied to the held-out test sets, the performance gains evaporated or only yielded marginal improvements (a slight drop for the manual model, and a $\sim+0.03$ increase for the GPT model).

Because we do not observe the large F1 jump on the test set required to classify optimization as the dominant bottleneck, we firmly reject **Hypothesis D (Optimization/threshold bottleneck)**. The performance is inherently capped by the quality of the frozen CT-CLIP visual embeddings, confirming **Hypothesis A (Representation bottleneck)**. We proceed to the scaling study (Phase 2) under the assumption that the feature representations are the primary limiting factor.