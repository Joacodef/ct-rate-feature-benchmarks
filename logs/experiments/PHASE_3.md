# Phase 3 — Robustness and Sensitivity

## Main Objective
Evaluate the robustness and sensitivity of the scaling law conclusions drawn in Phase 2 by mitigating data partition biases and establishing stable asymptotic ceilings.

## Item 1: Asymptotic Evaluation (GPT Labels)
- **Status:** completed
- **Goal:** Establish a stable and precise asymptotic ceiling for models trained on GPT-generated labels.
- **Protocol:**
  - Select the optimal model checkpoint trained on the maximum GPT label budget (e.g., $N=46,438$) from Phase 2.
  - Evaluate this asymptotic model directly against the entirety of the available manual label dataset ($N=1191$).
  - Calculate and record the aggregated metrics (AUPRC, AUROC, F1-macro).
- **Results:**
  - **AUPRC:** $0.5563 \pm 0.0056$
  - **AUROC:** $0.7298 \pm 0.0029$
  - **F1-macro:** $0.5748 \pm 0.0070$

## Item 2: $K$-Fold Cross-Validation (Manual Labels)
- **Goal:** Generate a robust scaling curve for manual labels, eliminating performance variations caused by fixed train/test splits.
- **Protocol:**
  - Implement a $K$-Fold Cross-Validation strategy ($K=5$) over the complete manual label dataset ($N=1191$).
  - Within each fold, use a single deterministic seed for weight initialization.
  - Evaluate the standard budget ladder ($N=20, 50, 100, 250, 500, 800, 1191$) by sampling these subsets exclusively from the training segment of the current fold.
  - Evaluate each trained budget model on the corresponding hold-out test segment of the current fold.
  - Aggregate the metrics (mean $\pm$ std) across all $K$ folds to plot the final unbiased scaling curve.

## Item 3: Per-Class Bottleneck Analysis
- **Goal:** Identify specific LLM weaknesses in label generation by comparing class-level performance.
- **Protocol:**
  - Extract per-class metrics (Precision, Recall, F1) from the asymptotic GPT model (from Item 1).
  - Extract per-class metrics from the optimal manual model (derived from Item 2).
  - Compare both sets of metrics to isolate abnormalities or conditions where the GPT-generated labels introduce the most significant degradation or noise.