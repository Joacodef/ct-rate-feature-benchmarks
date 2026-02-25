# Bottleneck Source Investigation Plan

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
- Stronger gains when using richer features or partial fine-tuning.

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

## Phase 0 — Reproducible Baseline

Deliverables:
- Frozen-feature MLP baseline for GPT labels.
- Frozen-feature MLP baseline for expert labels.
- Common metrics table: macro/micro AUPRC, AUROC, F1.

Acceptance:
- Runs are reproducible across 3+ seeds.
- No config/path inconsistencies.

## Phase 1 — Bottleneck Localization

### 1.1 Representation checks
- Compare baseline MLP vs larger MLP on same frozen features.
- Optional: partial unfreezing/adapter test (single controlled variant).

Interpretation:
- If all frozen-head variants plateau similarly, feature bottleneck likely dominant.

### 1.2 Label-quality checks
- GPT vs expert agreement on overlapping subset.
- Per-label error analysis (false positives/false negatives by class).
- Small manual audit of disagreement cases.

Interpretation:
- If disagreement clusters in low-performing labels, label quality contributes strongly.

### 1.3 Distribution checks
- Compare prevalence per label across train/val/test and GPT/expert sets.
- Evaluate by subgroup (if metadata available).

Interpretation:
- Large shifts indicate split/domain bottleneck.

### 1.4 Threshold/metric checks
- Tune per-label thresholds for F1 on validation.
- Recompute test F1 using frozen thresholds.

Interpretation:
- Large F1 jump with little AUPRC change => operating-point bottleneck.

## Phase 2 — Scaling Law Study (Core Objective)

Construct learning curves for GPT-label and expert-label training separately.

### 2.1 Label-budget ladder
Use matched training budgets (example):
- 100, 250, 500, 1k, 2k, 5k, 10k, full

For each budget:
- Keep architecture/protocol fixed.
- Train with multiple seeds.
- Evaluate on the same held-out expert-annotated test set (or fixed gold set).

### 2.2 Fit scaling trends
For each source (GPT, expert):
- Fit performance vs label count curve (log-scale x-axis).
- Estimate sample efficiency and asymptotic ceiling.

### 2.3 Crossover analysis
Compute:
- N_expert_to_beat_GPT_full
- N_GPT_to_beat_expert_full
- Budget where GPT and expert curves intersect (if any)

Interpretation categories:
- GPT dominates low-budget only.
- Expert dominates at all budgets.
- GPT catches up or surpasses at scale.

## Phase 3 — Robustness and Sensitivity

- Repeat key points with alternate split seed.
- Repeat with one stronger backbone setting (if feasible).
- Check whether conclusions about crossover are stable.

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
4. One representation stress test (e.g., linear probe or partial unfreeze).
4. One representation stress test (e.g., larger MLP or partial unfreeze).
5. Crossover estimate + uncertainty.

This MVP is sufficient to answer the main question with defensible evidence.

---

## 8) Reporting Template

For each experiment:
- Dataset source: GPT or expert
- Label budget: N
- Model setup: frozen/partial-ft, head type
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
