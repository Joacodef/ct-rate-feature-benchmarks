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
- Status: planned
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
- Status: planned
- Goal: Train manual-label models across all folds, budgets, and 5 seeds.
- Seeds: 52, 123, 456, 789, 999.
- Expected runs: 5 folds x 8 budgets x 5 seeds = 200 runs.
- Script: `scripts/run_phase3_manual_kfold_sweep.ps1` with `-Seeds`.

## Item 3 - GPT K-fold x Seed Training
- Status: planned
- Goal: Train GPT-label models with the same fold/seed protocol plus GPT-only high-budget tail.
- Seeds: 52, 123, 456, 789, 999.
- Expected runs:
  - Shared budgets: 5 folds x 8 budgets x 5 seeds = 200 runs.
  - GPT-only budgets: 5 folds x 3 budgets x 5 seeds = 75 runs.
  - Total GPT runs: 275.
- Requirement: use the same K-fold manifest-generation logic and sampling seed policy as manual.

## Item 4 - Aggregation and Reporting
- Status: planned
- Goal: Produce definitive budget-level metrics and variance decomposition.
- Primary aggregation:
  - Aggregate all runs per `(source, budget)` across fold and seed.
  - Report AUPRC, AUROC, F1-macro mean +- std and count.
- Secondary diagnostics:
  - Fold-averaged-per-seed variability.
  - Seed-averaged-per-fold variability.
  - 95% CI for primary metrics where feasible.

## Item 5 - Per-Class Analysis
- Status: planned
- Goal: Re-run per-class paired analysis with protocol symmetry.
- Compare matched manual vs GPT models at selected shared budgets (for example 250, 800, 1191, 1520).
- Keep fold pairing strict (same fold test manifest for both sources).
- If using seed pooling, report per-class means and std over fold-seed runs.

## Item 6 - Statistical Inference (CI + Tests)
- Status: planned
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
- `outputs/aggregated_results/manual_kfold_budget_s5_per_run.csv`
- `outputs/aggregated_results/manual_kfold_budget_s5_by_budget.csv`
- `outputs/aggregated_results/gpt_kfold_budget_s5_per_run.csv`
- `outputs/aggregated_results/gpt_kfold_budget_s5_by_budget.csv`
- Statistical summary table with paired deltas, 95% CI, raw p-values, and FDR-adjusted p-values for shared budgets.
- Updated per-class paired summaries under `outputs/aggregated_results/`.

## Notes
- Phase 2 remains exploratory/scouting evidence.
- This Phase 3 plan is confirmatory and intended to support final claims.
