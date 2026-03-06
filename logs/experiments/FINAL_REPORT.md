## Bottleneck Attribution Summary

Based on the experimental design and the established hypotheses, the following bottlenecks have been identified and ranked by their impact on the system's performance:

1. **Representation Bottleneck (High Impact / Dominant):**
   - **Evidence:** During Phase 0 (Hyperparameter Optimization), the search space systematically explored higher-capacity multi-layer perceptrons (increased depth and width). The optimization process consistently converged on shallower, lower-capacity architectures.
   - **Conclusion:** Within the frozen-feature setup, the CT-CLIP visual representation behaves as the primary performance ceiling. Increasing head capacity did not yield robust generalization gains, supporting Hypothesis A (Representation Bottleneck) as the dominant constraint under the evaluated protocol. This is strong indirect evidence, but not a formal substitute for dedicated linear-probe or partial-unfreezing experiments.

2. **Label Bottleneck (Protocol-Dependent / Mixed):**
   - **Evidence:** Phase 2 showed clear manual-label advantage in sample efficiency and absolute performance (manual at low budgets exceeded the GPT asymptotic ceiling). Phase 3 per-class paired analysis, however, showed GPT equal or better across all 5 evaluated classes at the selected checkpoints.
   - **Conclusion:** The label bottleneck is not unidirectional in this project. Results indicate protocol dependence: manual labels dominated in the original budget-scaling setup, while GPT labels matched or exceeded manual performance in the fold-matched per-class comparison. Therefore, we cannot claim a universal GPT-noise bottleneck, nor a universal GPT superiority claim.

3. **Optimization/Threshold Bottleneck (Secondary / Not Dominant):**
   - **Evidence:** In Phase 1, per-label threshold tuning increased validation F1, but these gains did not robustly transfer to held-out test performance (manual dropped; GPT improved only modestly).
   - **Conclusion:** Operating-point selection affects reported metrics but does not explain the primary performance ceiling in the scaling results. This supports rejecting Hypothesis D as the dominant bottleneck.

## Interpretation Notes

- Phase 3 includes a checkpoint-selection asymmetry (fixed asymptotic GPT checkpoint versus fold-trained manual models), so label-quality conclusions should be interpreted with care.
- Across all phases, the most stable claim is that frozen visual features are the main limiting factor under the current static-feature pipeline.