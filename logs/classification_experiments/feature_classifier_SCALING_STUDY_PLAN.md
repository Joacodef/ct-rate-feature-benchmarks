
# CT-Rate Label Source Scaling Study Plan

## 1) Main Objective (Anchor)

Determine whether classifiers trained with GPT-generated labels can match or surpass classifiers trained with expert-generated labels, and characterize the scaling laws for both label sources.

Primary questions:
- At fixed model/data pipeline, does GPT-label training reach comparable or better performance than expert-label training?
- What is the label-budget crossover point: how many expert labels are needed to beat GPT-label models, and vice versa?
- Under what bottlenecks (feature, label, optimization, split/domain) do conclusions change?

---

## 2) Working Assumption

Assume there is a performance bottleneck in the current setup (frozen CT-CLIP visual features + MLP head).

Goal of this plan:
- Identify the dominant bottleneck source(s).
- Quantify how each bottleneck affects the main objective above.
- Preserve a fair comparison between GPT-label and expert-label regimes.

---

## 3) Bottleneck Taxonomy and Hypotheses

### A. Representation bottleneck (features)
Hypothesis: frozen visual embeddings are not sufficiently task-separable for downstream abnormalities.

Signals:
- Low ceiling across all classifier heads/hyperparameters.
- Small gains from MLP capacity changes.
- Stronger gains only when changing to a different precomputed feature set (outside this repository's training code path).

### B. Label bottleneck (quality/noise)
Hypothesis: GPT labels introduce noise or calibration drift relative to expert labels.

Signals:
- Lower agreement with manual labels.
- Per-class degradation concentrated in ambiguous findings.
- Performance gap narrows after noise-robust training or relabel checks.

### C. Data/split bottleneck (distribution)
Hypothesis: split mismatch or prevalence shift dominates results.

Signals:
- Large train/val/test prevalence mismatch.
- High variance across random seeds/splits.
- Performance drops mainly on one subset/domain.

### D. Optimization/threshold bottleneck
Hypothesis: model ranking is decent (AUPRC/AUROC) but operating-point selection (F1) is suboptimal.

Signals:
- AUPRC improves but F1 remains low before threshold tuning.
- Strong F1 gain from per-label threshold optimization.

### E. Capacity bottleneck (head too weak/too strong)
Hypothesis: current MLP family underfits or overfits available signal.

Signals:
- Underfit: both train/val low.
- Overfit: train high, val low.
- Non-monotonic depth/width behavior under controlled regularization.

---

## 4) Experimental Design Principles

- Keep a single evaluation protocol across all comparisons.
- Use identical splits and seeds for GPT vs expert experiments.
- Separate model-selection metric from deployment metric:
  - Selection: AUPRC (imbalance-robust)
  - Final operating point: F1 with threshold tuning on validation
- Report uncertainty (mean ± std or CI over seeds).
- Track per-label and macro/micro metrics.

---


## 5) Phase Plan

Constraint note:
- This repository trains heads on frozen, precomputed features only.
- Partial backbone unfreezing/fine-tuning is out of scope here.
- Larger-MLP capacity stress testing is covered by Phase 0 HPO and is not duplicated as a separate required item.

## Phase 0 — Hyperparameter Optimization and Protocol Selection

Goal:
- Select stable, high-performing hyperparameters for both label sources using HPO.
- Representation bottleneck check is embedded in HPO: if larger MLPs do not outperform, feature bottleneck is likely.

Deliverables:
- Frozen protocol for scaling law study (selected hyperparameters, architecture, and evaluation protocol).

Acceptance:
- HPO runs completed for both GPT and manual labels.
- Protocol is fixed for downstream scaling experiments.

## Phase 1 — Model-Side Bottleneck Checks

### Item 1 — Threshold/metric checks
- Tune per-label thresholds for F1 on validation.
- Recompute test F1 using frozen thresholds.

Interpretation:
- Large F1 jump with little AUPRC change => operating-point bottleneck.

### Item 2 — Linear Probe Representation Check
- Train a strict linear probe (single linear layer on frozen CT-CLIP features) using the same data protocol and seeds.
- Keep all training/evaluation settings fixed relative to the selected Phase 0 protocol, changing only head capacity.
- Compare against the tuned nonlinear MLP on matched budgets.

Recommended matrix:
- Sources: manual and GPT labels.
- Budgets: 100, 500, 1191 (shared budgets for direct source comparison).
- Seeds: same Phase 2/3 seed subset (minimum 3; preferred 5).
- Metrics: macro/micro AUPRC, AUROC, and threshold-tuned F1.

Interpretation:
- Small MLP vs linear-probe gap => representation likely near linear-separable; feature bottleneck more plausible.
- Large MLP vs linear-probe gap => head capacity contributes materially; pure representation bottleneck is less dominant.
- If GPT-vs-manual ranking flips between heads, classify bottleneck impact as high and revisit core claims.

Phase placement decision:
- Canonical placement: Phase 1 (model-side bottleneck checks), because it is a capacity-control diagnostic.
- Optional mini-check in late Phase 0: 1-2 pilot runs can be used for early triage, but do not replace the full Phase 1 diagnostic.

### Note on Label Quality and Distribution Checks
- Label quality (agreement, error analysis, manual audit) and distribution checks (prevalence, subgroup analysis) are performed in the data/label-generation repository, not here. Summaries or references to those analyses may be included as needed.

## Phase 2 — Scaling Law Study (Core Objective)

Construct learning curves for GPT-label and expert-label training separately.

### Item 1 — Label-budget ladder
Use matched training budgets (example):
- 100, 250, 500, 1k, 2k, 5k, 10k, full

For each budget:
- Keep architecture/protocol fixed.
- Train with multiple seeds.
- Evaluate on the same held-out expert-annotated test set (or fixed gold set).

### Item 2 — Fit scaling trends
For each source (GPT, expert):
- Fit performance vs label count curve (log-scale x-axis).
- Estimate sample efficiency and asymptotic ceiling.

### Item 3 — Crossover analysis
Compute:
- N_expert_to_beat_GPT_full
- N_GPT_to_beat_expert_full
- Budget where GPT and expert curves intersect (if any)

Interpretation categories:
- GPT dominates low-budget only.
- Expert dominates at all budgets.
- GPT catches up or surpasses at scale.

## Phase 3 — Robustness and Sensitivity

- **GPT-Label Evaluation:** Evaluate the models trained with GPT labels against the entirety of the manual label dataset ($N=1191$) to establish a stable and precise asymptotic ceiling.
- **Manual-Label Evaluation:** Implement $K$-Fold Cross-Validation across the entire manual dataset to generate a robust scaling curve, eliminating partition bias.
- Optional external follow-up (out of scope for this repo): repeat under a different precomputed feature family and compare whether crossover conclusions persist.
- Check whether conclusions about crossover are stable under this asymmetric evaluation protocol.
- **Per-Class Bottleneck Analysis:** Compare class-level metrics (Precision, Recall, F1) between the best expert-label model and the asymptotic GPT-label model to identify specific LLM weaknesses.

---

## 6) Decision Framework: Does the Bottleneck Threaten the Main Objective?

For each identified bottleneck, classify impact:

- Low impact: affects absolute scores, not relative GPT vs expert ranking.
- Medium impact: shifts crossover point but preserves qualitative conclusion.
- High impact: reverses ranking or invalidates scaling-law inference.

Action:
- If high impact is detected, rerun scaling study after mitigation before drawing final claims.

---

## 7) Minimal Experiment Matrix (MVP)

1. Baseline GPT vs expert (frozen features, MLP, same protocol, 3 seeds).
2. Threshold-tuned F1 evaluation (per-label thresholds).
3. Label-budget curves at 5 budget points per source.
4. One representation stress test (linear probe on frozen features).
5. One protocol-sensitivity stress test (fixed-test endpoint vs fold-holdout endpoint comparison).
6. Crossover estimate + uncertainty.
7. Per-class performance breakdown comparing the optimal expert model vs. the asymptotic GPT model.

This MVP is sufficient to answer the main question with defensible evidence.

---

## 8) Reporting Template

For each experiment:
- Dataset source: GPT or expert
- Label budget: N
- Model setup: frozen features, head type
- Metrics: macro/micro AUPRC, AUROC, F1
- Threshold policy: fixed 0.5 or tuned
- Seeds: mean ± std
- Notes: failures, class imbalance issues

Final report should include:
- Bottleneck attribution summary (ranked by impact).
- Scaling-law plots for GPT and expert labels.
- Crossover table with confidence intervals.
- Clear statement: whether GPT labels can surpass expert-trained models under what budget/setting.

---

## 9) Exit Criteria

The investigation is complete when:
- Dominant bottleneck(s) are identified with evidence.
- GPT vs expert scaling curves are estimated with uncertainty.
- Crossover claims are supported by fixed-protocol experiments.
- A final recommendation is produced for next-stage training strategy.
