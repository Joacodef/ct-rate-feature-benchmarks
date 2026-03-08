# CT-Rate Label Source Scaling Study - Final Report

## 1) Executive Summary

Under frozen precomputed CT-CLIP features, final confirmatory evidence from Phase 3 (5-fold x 5-seed protocol with paired inference) indicates that GPT-label training is competitive and often superior to manual-label training across shared budgets from $N \ge 100$, with the strongest and most consistent gains in AUPRC/AUROC at higher shared budgets. Manual labels retain an advantage in selected low-budget regimes (notably $N=50$ for AUROC and F1-macro).

The apparent contradiction with Phase 2 is explained by protocol differences: Phase 2 uses a fixed test endpoint, while Phase 3 estimates performance over fold hold-outs and explicitly models uncertainty with paired CI and FDR-corrected tests.

## 2) Protocol Sensitivity (Required Stress Test)

### Phase 2 vs Phase 3

- **Phase 2 endpoint:** fixed `FINAL_TEST.csv` (exploratory, fixed-split).
- **Phase 3 endpoint:** fold hold-out manifests with paired `(fold, seed)` analysis (confirmatory).

Observed effect:

- **Phase 2 (fixed endpoint):** manual > GPT over shared budgets in the reported MLP runs.
- **Phase 3 (fold hold-outs + inference):** GPT matches/exceeds manual for many budget-metric pairs, with statistically supported gains at multiple shared budgets.

Conclusion:

- Cross-source ranking is sensitive to evaluation protocol; therefore, final claims are anchored to Phase 3.

## 3) Control Result: Linear Probe (Phase 1)

Linear probe (single linear head, frozen features) was used as a head-capacity control at shared budgets 100/500/1191.

- Manual remained above GPT on the fixed endpoint in this control.
- Therefore, head nonlinearity is not the main confounder behind the fixed-endpoint ranking in Phase 2.

This supports the interpretation that protocol/endpoint effects and label-feature alignment are stronger drivers of cross-source conclusions than hidden-layer capacity alone.

Additional Phase 1 threshold check (5 seeds, `FINAL_TEST.csv`):

- Source: `outputs/aggregated_results/phase1_threshold_optimized_summary_all_metrics.csv`
- Manual: AUPRC $0.6550 \pm 0.0991$, AUROC $0.7746 \pm 0.0652$, F1-macro-optimized $0.6181 \pm 0.0538$
- GPT: AUPRC $0.5751 \pm 0.0090$, AUROC $0.7157 \pm 0.0044$, F1-macro-optimized $0.5934 \pm 0.0061$

Interpretation:

- Threshold tuning improves absolute test F1 for both sources under this fixed endpoint (larger gain for GPT), while preserving manual > GPT ordering in the fixed-endpoint setting.

## 4) Crossover and Uncertainty (Phase 3 Canonical)

Source for all inferential values:
- `outputs/aggregated_results/phase3_manual_vs_gpt_summary.csv`

Paired delta is defined as `Manual - GPT`.

### AUPRC (shared budgets)

| Budget | Delta Mean | 95% CI | FDR p-value | Significant | Direction |
| --- | --- | --- | --- | --- | --- |
| 20 | -0.0063 | [-0.0212, +0.0111] | 0.5255 | No | GPT slight (ns) |
| 50 | +0.0181 | [-0.0000, +0.0371] | 0.0977 | No | Manual slight (ns) |
| 100 | -0.0190 | [-0.0329, -0.0051] | 0.0254 | Yes | GPT |
| 250 | -0.0267 | [-0.0369, -0.0161] | 0.0024 | Yes | GPT |
| 500 | -0.0144 | [-0.0256, -0.0028] | 0.0330 | Yes | GPT |
| 800 | -0.0137 | [-0.0249, -0.0032] | 0.0330 | Yes | GPT |
| 1191 | -0.0168 | [-0.0277, -0.0063] | 0.0140 | Yes | GPT |
| 1520 | -0.0138 | [-0.0218, -0.0062] | 0.0132 | Yes | GPT |

### AUROC (shared budgets)

| Budget | Delta Mean | 95% CI | FDR p-value | Significant | Direction |
| --- | --- | --- | --- | --- | --- |
| 20 | -0.0074 | [-0.0242, +0.0126] | 0.5255 | No | GPT slight (ns) |
| 50 | +0.0245 | [+0.0108, +0.0376] | 0.0134 | Yes | Manual |
| 100 | -0.0054 | [-0.0172, +0.0071] | 0.5182 | No | GPT slight (ns) |
| 250 | +0.0030 | [-0.0080, +0.0136] | 0.6390 | No | Manual slight (ns) |
| 500 | -0.0148 | [-0.0257, -0.0045] | 0.0153 | Yes | GPT |
| 800 | -0.0162 | [-0.0232, -0.0091] | 0.0024 | Yes | GPT |
| 1191 | -0.0123 | [-0.0199, -0.0049] | 0.0134 | Yes | GPT |
| 1520 | -0.0179 | [-0.0263, -0.0102] | 0.0024 | Yes | GPT |

### F1-macro (shared budgets)

| Budget | Delta Mean | 95% CI | FDR p-value | Significant | Direction |
| --- | --- | --- | --- | --- | --- |
| 20 | -0.0989 | [-0.1566, -0.0444] | 0.0134 | Yes | GPT |
| 50 | +0.0412 | [+0.0154, +0.0663] | 0.0140 | Yes | Manual |
| 100 | -0.0497 | [-0.0834, -0.0193] | 0.0134 | Yes | GPT |
| 250 | -0.0021 | [-0.0245, +0.0191] | 0.8580 | No | Near parity |
| 500 | -0.0044 | [-0.0172, +0.0083] | 0.5704 | No | Near parity |
| 800 | -0.0182 | [-0.0305, -0.0061] | 0.0171 | Yes | GPT |
| 1191 | -0.0153 | [-0.0245, -0.0061] | 0.0134 | Yes | GPT |
| 1520 | -0.0194 | [-0.0316, -0.0075] | 0.0134 | Yes | GPT |

Interpretation:

- AUPRC: GPT advantage is significant from $N=100$ onward.
- AUROC: manual only leads significantly at $N=50$; GPT leads significantly from $N=500$ onward.
- F1-macro: mixed at low budgets, then GPT-leaning at higher shared budgets.

## 5) Bottleneck Attribution (Ranked by Impact)

Impact categories follow the study plan decision framework.

1. **Data/split protocol bottleneck (C): High impact**
- Evidence: ranking differs between fixed-endpoint Phase 2 and fold-holdout Phase 3.
- Effect: changes qualitative cross-source conclusion.

2. **Label bottleneck (B): Medium-to-High impact**
- Evidence: paired Phase 3 deltas favor GPT in many shared-budget metric pairs; per-class trends show GPT recall benefits at higher budgets.
- Effect: materially shifts crossover location and asymptotic relative standing.

3. **Representation bottleneck (A): Medium impact**
- Evidence: absolute ceilings remain moderate even at high budgets (AUPRC ~0.57, AUROC ~0.73), indicating frozen-feature limits.
- Effect: constrains both sources; does not alone determine GPT-vs-manual ranking.

4. **Optimization/threshold bottleneck (D): Low-to-Medium impact (not primary)**
- Evidence: under five-seed `FINAL_TEST.csv` recalculation, threshold tuning improves absolute test metrics (especially GPT F1), but does not overturn fixed-endpoint source ordering and does not reconcile fixed-endpoint vs fold-holdout protocol divergence.
- Effect: relevant for calibration-dependent absolute performance; limited for the headline cross-source claim anchored to Phase 3 paired inference.

5. **Head capacity bottleneck (E): Low-to-Medium impact for ranking**
- Evidence: linear-probe control preserved fixed-endpoint ranking; larger-MLP variation already explored in HPO.
- Effect: affects absolute scores, but not main ranking direction in fixed-endpoint setting.

## 6) Final Recommendation (Next-Stage Strategy)

1. **Use Phase 3 as canonical evidence for cross-source claims.**
- Final statements should cite fold+seed paired inference, not Phase 2 fixed-endpoint alone.

2. **For scaling under current frozen-feature pipeline, prioritize GPT-label expansion once above low-budget regime.**
- Supported by significant AUPRC/AUROC advantages across many shared high budgets and superior GPT-only asymptotic scaling.

3. **Retain manual labels strategically for low-budget calibration and quality checks.**
- Manual remains strong at selected low-budget operating points (for example AUROC/F1 at $N=50$).

4. **Keep protocol consistency strict for future comparisons.**
- Report paired deltas, 95% CI, and FDR-adjusted p-values by shared budget as standard output.

5. **Representation improvement is the most valuable external follow-up.**
- Within this repo, training remains frozen-feature only; any backbone/feature-family upgrade should be treated as a separate external study and compared using the same Phase 3 inferential protocol.

## 7) Exit-Criteria Check

- Dominant bottlenecks identified with evidence: **Yes**.
- GPT vs expert scaling curves estimated with uncertainty: **Yes**.
- Crossover claims supported by fixed-protocol experiments: **Yes** (Phase 2 + Phase 3, with protocol-conditional interpretation).
- Final recommendation produced: **Yes**.

The study plan is considered complete under the revised, repository-constrained scope.
