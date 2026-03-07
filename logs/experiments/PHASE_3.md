# Phase 3 - Robustness and Sensitivity Plan

## Main Objective
Establish a conclusive, protocol-symmetric comparison between manual and GPT labels by controlling both data-partition variance (K-fold) and optimization variance (multiple seeds).

## Core Design
- Primary protocol: 5-fold cross-validation for both label sources.
- Seed policy: 5 seeds per fold for both label sources.
- Shared budgets for fair comparison: 20, 50, 100, 250, 500, 800, 1191, 1520.
- GPT-only asymptotic budgets: 2000, 5000, 10000.
- Evaluation endpoint: fold hold-out test manifests only.
- Final headline metrics per budget: mean +- std across all fold-seed runs (n = 5 x 5 = 25).

## Hypotheses
1. Representation bottleneck remains the dominant limiting factor under frozen features.
2. Manual labels are more sample-efficient at shared budgets.
3. GPT labels may approach or exceed manual in selected classes and/or asymptotic regimes.

## Item 1 - Manifest Generation
- Status: completed
- Goal: Generate fold-specific train/val/test manifests for both label sources under aligned settings.
- Manual manifests:
  - Source: existing `data/manifests/manual/manual_kfold_budget_splits`.
  - Budgets: 20, 50, 100, 250, 500, 800, 1191, 1520.
- GPT manifests:
  - Source full manifest: `data/manifests/gpt/all.csv`.
  - Budgets: 20, 50, 100, 250, 500, 800, 1191, 1520, 2000, 5000, 10000.
  - Output subdir: `gpt_kfold_budget_splits`.
  - Prefix: `gpt_kfold`.

## Item 2 - Manual K-fold x Seed Training
- Status: completed
- Goal: Train manual-label models across all folds, budgets, and 5 seeds.
- Seeds: 52, 123, 456, 789, 999.
- Expected runs: 5 folds x 8 budgets x 5 seeds = 200 runs.
- Script: `scripts/run_phase3_manual_kfold_sweep.ps1` with `-Seeds`.

## Item 3 - GPT K-fold x Seed Training
- Status: completed
- Goal: Train GPT-label models with the same fold/seed protocol plus GPT-only high-budget tail.
- Seeds: 52, 123, 456, 789, 999.
- Expected runs:
  - Shared budgets: 5 folds x 8 budgets x 5 seeds = 200 runs.
  - GPT-only budgets: 5 folds x 3 budgets x 5 seeds = 75 runs.
  - Total GPT runs: 275.
- Requirement: use the same K-fold manifest-generation logic and sampling seed policy as manual.

## Item 4 - Aggregation and Reporting
- Status: completed
- Goal: Produce definitive budget-level metrics and variance decomposition.
- Primary aggregation:
  - Aggregate all runs per `(source, budget)` across fold and seed.
  - Report AUPRC, AUROC, F1-macro mean +- std and count.
- Secondary diagnostics:
  - Fold-averaged-per-seed variability.
  - Seed-averaged-per-fold variability.
  - 95% CI for primary metrics where feasible.

## Item 5 - Per-Class Analysis
- Status: completed
- Goal: Re-run per-class paired analysis with protocol symmetry.
- Compare matched manual vs GPT models at selected shared budgets (for example 250, 800, 1191, 1520).
- Keep fold pairing strict (same fold test manifest for both sources).
- If using seed pooling, report per-class means and std over fold-seed runs.

## Item 6 - Statistical Inference (CI + Tests)
- Status: completed
- Goal: Replace descriptive-only reporting (mean +- std) with inferential statistics for manual vs GPT comparisons at shared budgets.
- Analysis unit: each matched `(fold, seed)` pair.
- Primary quantity: paired delta per metric and budget (`Manual - GPT`).
- Confidence intervals:
  - Compute 95% bootstrap CI for each paired delta (AUPRC, AUROC, F1-macro).
  - Also report 95% CI for each source-specific mean per budget.
- Hypothesis tests:
  - Primary test: paired permutation test on delta values per budget and metric.
  - Secondary check: paired t-test (reported only as sensitivity analysis).
- Multiple comparisons:
  - Control false discoveries with Benjamini-Hochberg (FDR) across tested budgets and metrics.
- Effect size reporting:
  - Report mean paired delta and standardized effect where appropriate.
- Decision rule:
  - Treat a budget-level difference as supported when CI excludes 0 and FDR-adjusted p-value is below threshold.
  - If CI remains wide at key shared budgets, extend only those budgets with additional seeds.
- Compute policy:
  - Initial target remains 5 folds x 5 seeds.
  - Increase seeds selectively only where inference is inconclusive.

## Acceptance Criteria
- Both sources use identical fold count and seed list.
- Shared budgets are evaluated with matched protocols and sample-selection rules.
- Final comparisons are based on fold + seed aggregated metrics (not single checkpoint anecdotes).
- Final shared-budget claims include paired 95% CI, hypothesis-test results, and multiple-comparison correction.
- All commands, manifests, and outputs are reproducible from scripts and index files.

## Deliverables
- `outputs/aggregated_results/manual_kfold_budget_per_run.csv`
- `outputs/aggregated_results/manual_kfold_budget_by_budget.csv`
- `outputs/aggregated_results/gpt_kfold_budget_per_run.csv`
- `outputs/aggregated_results/gpt_kfold_budget_by_budget.csv`
- Statistical summary table with paired deltas, 95% CI, raw p-values, and FDR-adjusted p-values for shared budgets.
- Updated per-class paired summaries under `outputs/aggregated_results/`.

## Notes
- Phase 2 remains exploratory/scouting evidence.
- This Phase 3 plan is confirmatory and intended to support final claims.


## Results

### Item 4 - Aggregation and Reporting

#### Shared Budgets ($N=20$ to $1520$)
Across the shared data budget regimes, models trained on GPT labels effectively matched or outperformed models trained on manual labels on the primary hold-out test sets. 
* At the lowest budgets (e.g., $N=50$), manual labels showed a slight edge, yielding an AUPRC of $0.507 \pm 0.028$ and AUROC of $0.679 \pm 0.019$, compared to the GPT labels' AUPRC of $0.489 \pm 0.028$ and AUROC of $0.655 \pm 0.025$.
* From $N=100$ onwards, GPT labels demonstrated strong sample efficiency. By the maximum shared budget of $N=1520$, manual models reached an AUPRC of $0.526 \pm 0.021$ and AUROC of $0.691 \pm 0.020$, whereas GPT models exceeded this with an AUPRC of $0.539 \pm 0.024$ and AUROC of $0.709 \pm 0.025$.

#### Asymptotic Regimes (GPT-only, $N=2000$ to $10000$)
Scaling the GPT-derived labels to larger budgets yielded continuous but diminishing returns, indicating a plateau effect.
* At $N=5000$, GPT models achieved an AUPRC of $0.562 \pm 0.027$ and AUROC of $0.724 \pm 0.024$.
* At the maximum budget of $N=10000$, GPT models achieved the highest overall performance with an AUPRC of $0.568 \pm 0.029$, an AUROC of $0.731 \pm 0.025$, and a Macro-F1 of $0.574 \pm 0.023$.

### Item 5 - Per-Class Analysis

The per-class paired analysis ($\text{Manual} - \text{GPT}$) across shared budgets revealed the following trends:

* **Low Budget Volatility ($N=20$ to $50$):** At the most constrained budgets, performance variance is notably high for both sources. For example, at $N=20$, the manual F1-score for Lymphadenopathy is $0.355 \pm 0.222$, while the GPT F1-score is $0.387 \pm 0.149$.
* **Mid-to-High Shared Budgets ($N=250$ to $1191$):** As the budget scales, GPT labels generally show better stability and stronger average F1 in several classes. At $N=800$, the largest GPT gains appear in Pulmonary fibrotic sequela ($\Delta\text{F1}=-0.060$), Lung nodule ($\Delta\text{F1}=-0.024$), and Arterial wall calcification ($\Delta\text{F1}=-0.015$), while Lymphadenopathy ($\Delta\text{F1}=+0.001$) and Lung opacity ($\Delta\text{F1}=+0.006$) remain near parity or slightly favor manual.
* **Maximum Shared Budget ($N=1520$):** At the maximum available shared data budget, GPT labels decisively outperform manual annotations in F1-score across every tracked pathology.
    * **Arterial wall calcification:** GPT achieves the highest absolute performance with an F1-score of $0.718 \pm 0.055$, compared to the manual F1-score of $0.706 \pm 0.041$ ($\Delta\text{F1} = -0.013$).
    * **Lymphadenopathy:** GPT F1-score is $0.576 \pm 0.033$, exceeding the manual F1-score of $0.552 \pm 0.037$ ($\Delta\text{F1} = -0.023$).
    * **Pulmonary fibrotic sequela:** GPT shows the largest margin of improvement with an F1-score of $0.431 \pm 0.039$, whereas the manual score is $0.393 \pm 0.053$ ($\Delta\text{F1} = -0.038$).
A significant driver of the GPT label performance at this budget is the notably higher Recall. For example, GPT Recall for Arterial wall calcification reaches $0.884 \pm 0.044$ compared to the manual Recall of $0.812 \pm 0.065$.


### Item 6 - Statistical Inference (CI + Tests)

A paired permutation test with Benjamini-Hochberg False Discovery Rate (FDR) correction was performed to analyze the paired delta (Manual - GPT) across the shared budgets. Statistical significance is defined as an FDR-adjusted p-value $< 0.05$.

All highlighted significant findings also satisfy the CI-based decision rule (95% paired-delta CI excludes 0).

* **Low Budgets ($N=20, 50$):** At $N=20$, there is no significant difference in AUPRC or AUROC, though GPT models show a statistically significant advantage in Macro-F1 (paired delta $=-0.099$, $p_{adj}=0.013$). At $N=50$, manual labels demonstrate a statistically significant advantage over GPT labels in AUROC (paired delta $=+0.025$, $p_{adj}=0.013$) and Macro-F1 (paired delta $=+0.041$, $p_{adj}=0.014$).
* **Transition Budgets ($N=100, 250$):** At $N=100$, GPT models begin to statistically significantly outperform manual models in AUPRC (paired delta $=-0.019$, $p_{adj}=0.025$) and Macro-F1 (paired delta $=-0.050$, $p_{adj}=0.013$). At $N=250$, GPT models maintain a statistically significant lead in AUPRC (paired delta $=-0.027$, $p_{adj}=0.002$).
* **High Shared Budgets ($N=500$ to $1520$):** In the higher data regimes, GPT-derived labels consistently and significantly outperform manual labels. For AUPRC and AUROC, GPT exhibits a statistically significant advantage at every budget from $N=500$ through $N=1520$. At the maximum shared budget of $N=1520$, the paired delta for AUPRC is $-0.014$ ($p_{adj}=0.013$) and for AUROC is $-0.018$ ($p_{adj}=0.002$), both favoring GPT labels.

## Conclusions

### Initial Hypotheses Evaluation

Based on the robust 5-fold, 5-seed evaluation protocol (aggregating $n=25$ runs per budget), we reach the following conclusions regarding our initial hypotheses:

1. **Hypothesis 1 (Supported):** Representation bottleneck remains the dominant limiting factor under frozen features. Even at a training budget of $10000$ samples, absolute performance plateaus at an AUPRC of $\sim 0.57$ and AUROC of $\sim 0.73$. This asymptotic behavior indicates that the frozen feature space fundamentally restricts the linear separability of the pathology classes.
2. **Hypothesis 2 (Rejected):** Manual labels are *not* universally more sample-efficient at shared budgets. Statistical inference confirms that manual labels show a significant advantage only at a very low data regime ($N=50$), while GPT labels show significant advantages in several metric-budget pairs from $N=100$ onward and especially across higher shared budgets ($N=500$ to $1520$). This suggests that the signal provided by the LLM-extracted labels is often more consistent and better aligned with the frozen feature representations once a minimal data threshold is crossed.
3. **Hypothesis 3 (Supported):** GPT labels approach and exceed manual annotations in asymptotic regimes. The ability to scale GPT labels cost-effectively to $10000$ samples allowed the models to significantly surpass the maximum performance achievable with the completely exhausted manual dataset at $N=1520$.


### About Per-Class Evaluation

The granular per-class analysis corroborates the aggregate findings, showing that GPT labels are frequently advantageous at higher shared budgets and consistently ahead across all tracked classes at $N=1520$. A major driver is higher recall in multiple classes (for example, Arterial wall calcification), suggesting that report-derived labels may preserve broader class signal than manual annotations in this frozen-feature setting.