# Phase 3 — Robustness and Sensitivity

## Main Objective
Evaluate the robustness and sensitivity of the scaling law conclusions drawn in Phase 2 by mitigating data partition biases and establishing stable asymptotic ceilings.

## Item 1: Asymptotic Evaluation (GPT Labels)
- **Status:** completed
- **Goal:** Establish a stable and precise asymptotic ceiling for models trained on GPT-generated labels.
- **Protocol:**
  - Select the optimal model checkpoint trained on the maximum GPT label budget (e.g., $N=46,438$) from Phase 2.
  - Evaluate this asymptotic model across the 5 hold-out test folds generated for the manual labels $K$-Fold Cross-Validation.
  - Calculate and record the aggregated metrics (AUPRC, AUROC, F1-macro) as the mean $\pm$ standard deviation across the 5 folds.
- **Results:**
  - **AUPRC:** $0.5617 \pm 0.0355$
  - **AUROC:** $0.7296 \pm 0.0272$
  - **F1-macro:** $0.5756 \pm 0.0228$

## Item 2: $K$-Fold Cross-Validation (Manual Labels)
- **Status:** completed
- **Goal:** Generate a robust scaling curve for manual labels, eliminating performance variations caused by fixed train/test splits.
- **Protocol:**
  - Implement a $K$-Fold Cross-Validation strategy ($K=5$) over the complete manual label dataset ($N=1520$).
  - Within each fold, use a single deterministic seed for weight initialization.
  - Evaluate the standard budget ladder (with the added full test dataset) ($N=20, 50, 100, 250, 500, 800, 1191, 1520$) by sampling these subsets exclusively from the training segment of the current fold.
  - Evaluate each trained budget model on the corresponding hold-out test segment of the current fold.
  - Aggregate the metrics (mean $\pm$ std) across all $K$ folds to plot the final unbiased scaling curve.
- **Results (5-fold aggregated, primary hold-out metrics):**
  - **$N=20$:** AUPRC $0.4647 \pm 0.0348$, AUROC $0.6365 \pm 0.0334$, F1-macro $0.4240 \pm 0.0852$
  - **$N=50$:** AUPRC $0.5110 \pm 0.0282$, AUROC $0.6856 \pm 0.0169$, F1-macro $0.4824 \pm 0.1180$
  - **$N=100$:** AUPRC $0.4983 \pm 0.0275$, AUROC $0.6758 \pm 0.0171$, F1-macro $0.4516 \pm 0.1120$
  - **$N=250$:** AUPRC $0.5089 \pm 0.0335$, AUROC $0.6837 \pm 0.0233$, F1-macro $0.5263 \pm 0.0525$
  - **$N=500$:** AUPRC $0.5113 \pm 0.0125$, AUROC $0.6835 \pm 0.0203$, F1-macro $0.5477 \pm 0.0337$
  - **$N=800$:** AUPRC $0.5335 \pm 0.0197$, AUROC $0.6996 \pm 0.0180$, F1-macro $0.5451 \pm 0.0272$
  - **$N=1191$:** AUPRC $0.5359 \pm 0.0138$, AUROC $0.7008 \pm 0.0225$, F1-macro $0.5366 \pm 0.0129$
  - **$N=1520$:** AUPRC $0.5288 \pm 0.0379$, AUROC $0.6910 \pm 0.0242$, F1-macro $0.5252 \pm 0.0359$
- **Conclusions:**
  - The manual-label scaling curve demonstrates stable and continuous gains from a properly balanced few-shot baseline ($N=20$) to the mid/high-budget regime, with a robust transition towards $N \in [500, 800]$.
  - Performance plateaus around $N=800$ to $N=1191$ (AUPRC and AUROC improvements are marginal beyond this region).
  - The $N=1520$ point does not improve the curve and is slightly worse than $N=1191$, suggesting diminishing returns and/or fold-train-pool ceiling effects.
  - Variance (std) is most pronounced at low budgets (specifically F1 at $N=20$, $N=50$, and $N=100$), and generally decreases at higher budgets, corroborating the expected instability of the few-shot regime and supporting the robustness objective of Item 2.

## Item 3: Per-Class Bottleneck Analysis
- **Status:** completed
- **Goal:** Characterize label-source differences at the class level by comparing class-level performance.
- **Protocol:**
  - Extract per-class metrics (Precision, Recall, F1) from the asymptotic GPT model (from Item 1).
  - Extract per-class metrics from the optimal manual model (derived from Item 2).
  - Compare both sets of metrics to isolate abnormalities or conditions where the GPT-generated labels introduce the most significant degradation or noise.
- **Implementation Notes:**
  - Use the optimal manual model from Item 2 at $N=1191$ (best mean AUPRC/AUROC in the K-fold curve).
  - Evaluate the asymptotic GPT checkpoint (`train_n46438_s11111`) on the same 5 fold hold-out manifests (`manual_kfold_f1_test.csv` ... `manual_kfold_f5_test.csv`) to produce fold-matched detailed reports.
  - Run per-class paired comparison with:
    - `python .\scripts\compare_per_class_bottlenecks.py --manual-glob "outputs/manual_kfold_budget/f*_n1191_s52/evaluation_aggregate/detailed_metrics/manual_kfold_f*_test.csv_detailed_metrics.json" --gpt-glob "outputs/gpt_budget/train_n46438_s11111/evaluation_aggregate/detailed_metrics/manual_kfold_f*_test.csv_detailed_metrics.json" --output-prefix outputs/aggregated_results/per_class_bottleneck --output-markdown outputs/aggregated_results/per_class_bottleneck_summary.md`
  - Primary output files:
    - `outputs/aggregated_results/per_class_bottleneck_paired.csv`
    - `outputs/aggregated_results/per_class_bottleneck_summary.csv`
    - `outputs/aggregated_results/per_class_bottleneck_summary.md`
- **Results (5-fold paired comparison; positive deltas mean Manual > GPT):**
  - **Arterial wall calcification:** Manual F1 $0.7372 \pm 0.0483$, GPT F1 $0.7537 \pm 0.0578$, $\Delta$F1 $-0.0165 \pm 0.0191$
  - **Lymphadenopathy:** Manual F1 $0.5655 \pm 0.0358$, GPT F1 $0.5967 \pm 0.0389$, $\Delta$F1 $-0.0312 \pm 0.0242$
  - **Lung opacity:** Manual F1 $0.5163 \pm 0.0455$, GPT F1 $0.5490 \pm 0.0780$, $\Delta$F1 $-0.0328 \pm 0.0421$
  - **Pulmonary fibrotic sequela:** Manual F1 $0.3975 \pm 0.0501$, GPT F1 $0.4359 \pm 0.0304$, $\Delta$F1 $-0.0384 \pm 0.0615$
  - **Lung nodule:** Manual F1 $0.4666 \pm 0.0335$, GPT F1 $0.5429 \pm 0.0549$, $\Delta$F1 $-0.0763 \pm 0.0609$
- **Conclusions:**
  - For the evaluated 5-label setup, no class shows a positive mean $\Delta$F1; GPT is equal or better than manual-label models across all classes.
  - The largest performance gap appears in **Lung nodule** ($\Delta$F1 $=-0.0763 \pm 0.0609$), driven mainly by recall difference ($\Delta$recall $=-0.1243 \pm 0.0700$).
  - The smallest gap is **Arterial wall calcification** ($\Delta$F1 $=-0.0165 \pm 0.0191$), where manual shows slightly better precision but lower recall.
  - Under this protocol, the expected "GPT bottleneck" pattern is not observed for the selected targets.

## Phase 3 Interpretation Note

- This phase provides a robustness/sensitivity view and should be interpreted alongside Phase 2, not as a strict replacement of the fixed-split scaling conclusion.
- The per-class comparison uses a fixed asymptotic GPT checkpoint and fold-trained manual checkpoints, so cross-source interpretation should account for this checkpoint-selection asymmetry.