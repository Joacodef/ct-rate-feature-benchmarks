# Experiments Log

Manual registry of experiment actions and decisions.

## Logging Rules
- Add one entry per action you intentionally run.
- Include exact command, config, outputs path, and short rationale.
- Mark status as `planned`, `running`, `completed`, or `failed`.
- If a command fails, paste the key traceback/error line and resolution.

## Entry Template

### [YYYY-MM-DD HH:MM] <short-title>
- **Status:** planned | running | completed | failed
- **Plan phase:** Phase 0 | Phase 1 | Phase 2 | Phase 3
- **Plan item:** integer checklist item within phase (e.g., 1, 2, 3)
- **Stage:** baseline | optuna | budget-splits | training | evaluation | analysis
- **Source:** GPT labels | Manual labels
- **Command:**
  ```powershell
  <exact command>
  ```
- **Config:** <config path>
- **Inputs:** <manifests/splits/features>
- **Outputs:** <output folder/study name>
- **Reason:** <why this was run>
- **Result summary:** <main outcome, metrics, or artifact produced>
- **Notes:** <issues, follow-up actions>

---

## Plan Tracking Map (Canonical names from Bottleneck Investigation Plan)

- **Phase 0 — Hyperparameter Optimization and Protocol Selection**
  - Goal: establish stable GPT/manual training protocol and select frozen hyperparameters.
  - Status: completed (Optuna completed for both sources).
- **Phase 1 — Model-Side Bottleneck Checks**
  - Item 1: Threshold/metric checks (per-label threshold tuning, F1 recomputation).
  - Note: Label quality and distribution checks are performed in the data/label-generation repo, not here.
  - Status: not started.
- **Phase 2 — Scaling Law Study (Core Objective)**
  - Item 1: Label-budget ladder.
  - Item 2: Fit scaling trends.
  - Item 3: Crossover analysis.
  - Status: item 1 completed; items 2-3 pending final analysis outputs.
- **Phase 3 — Robustness and Sensitivity**
  - Goal: alternate split seed and stronger-backbone sensitivity checks.
  - Status: not started.

---

## Recorded Actions

## Phase 0 — Hyperparameter Optimization and Protocol Selection

### [2026-02-25] Optuna search (GPT labels) - run 1
- **Status:** completed
- **Stage:** optuna
- **Source:** GPT labels
- **Command:**
  ```powershell
  python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_gpt_labels_config.yaml --study-name gpt_mlp_search --n-trials 120
  ```
- **Config:** configs/optuna_gpt_labels_config.yaml
- **Outputs:** outputs/optuna_gpt_labels_studies/gpt_mlp_search
- **Reason:** First Optuna run for GPT-label baseline tuning.

### [2026-02-25] Optuna search (GPT labels) - run 2
- **Status:** completed
- **Stage:** optuna
- **Source:** GPT labels
- **Command:**
  ```powershell
  python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_gpt_labels_config.yaml --study-name gpt_mlp_search_2 --n-trials 120
  ```
- **Config:** configs/optuna_gpt_labels_config.yaml
- **Outputs:** outputs/optuna_gpt_labels_studies/gpt_mlp_search_2
- **Reason:** Second Optuna run with different seed/study instance.

### [2026-02-25] Optuna search (GPT labels) - run 3
- **Status:** completed
- **Stage:** optuna
- **Source:** GPT labels
- **Command:**
  ```powershell
  python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_gpt_labels_config.yaml --study-name gpt_mlp_search_3 --n-trials 120
  ```
- **Config:** configs/optuna_gpt_labels_config.yaml
- **Outputs:** outputs/optuna_gpt_labels_studies/gpt_mlp_search_3
- **Reason:** Third Optuna run with different seed/study instance.

### [2026-02-25] Optuna search (Manual labels) - run 1
- **Status:** completed
- **Stage:** optuna
- **Source:** Manual labels
- **Command:**
  ```powershell
  python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_manual_labels_config.yaml --study-name manual_mlp_search --n-trials 120
  ```
- **Config:** configs/optuna_manual_labels_config.yaml
- **Outputs:** outputs/optuna_manual_labels_studies/manual_mlp_search
- **Reason:** First Optuna run for manual-label tuning.

### [2026-02-25] Optuna search (Manual labels) - run 2
- **Status:** completed
- **Stage:** optuna
- **Source:** Manual labels
- **Command:**
  ```powershell
  python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_manual_labels_config.yaml --study-name manual_mlp_search_2 --n-trials 120
  ```
- **Config:** configs/optuna_manual_labels_config.yaml
- **Outputs:** outputs/optuna_manual_labels_studies/manual_mlp_search_2
- **Reason:** Second Optuna run with different seed/study instance.

### [2026-02-25] Optuna search (Manual labels) - run 3
- **Status:** completed
- **Stage:** optuna
- **Source:** Manual labels
- **Command:**
  ```powershell
  python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_manual_labels_config.yaml --study-name manual_mlp_search_3 --n-trials 120
  ```
- **Config:** configs/optuna_manual_labels_config.yaml
- **Outputs:** outputs/optuna_manual_labels_studies/manual_mlp_search_3
- **Reason:** Third Optuna run with different seed/study instance.

### [2026-02-25] Optuna best summary (GPT studies)
- **Status:** completed
- **Stage:** analysis
- **Source:** GPT labels
- **Command:**
  ```powershell
  python .\scripts\optuna_best_summary.py .\outputs\optuna_gpt_labels_studies\gpt_mlp_search
  python .\scripts\optuna_best_summary.py .\outputs\optuna_gpt_labels_studies\gpt_mlp_search_2
  python .\scripts\optuna_best_summary.py .\outputs\optuna_gpt_labels_studies\gpt_mlp_search_3
  ```
- **Outputs:**
  - outputs/optuna_gpt_labels_studies/gpt_mlp_search
  - outputs/optuna_gpt_labels_studies/gpt_mlp_search_2
  - outputs/optuna_gpt_labels_studies/gpt_mlp_search_3
- **Result summary:**
  - `gpt_mlp_search`: best AUPRC `0.544412`, top_10 std `0.000053`, all-trial std `0.010757`
  - `gpt_mlp_search_2`: best AUPRC `0.545538`, top_10 std `0.000072`, all-trial std `0.009402`
  - `gpt_mlp_search_3`: best AUPRC `0.544929`, top_10 std `0.000026`, all-trial std `0.006701`
- **Notes:**
  - Stability favors `gpt_mlp_search_3` (lowest global std, lowest top-k std, smallest peak gap).
  - Peak AUPRC is highest in `gpt_mlp_search_2`, but gain vs `_3` is small (`+0.000609`).
  - Candidate frozen GPT hyperparameters for scaling study: `hidden_dims=[4096,4096]`, `learning_rate=1.078345968032944e-05`, `weight_decay=1.4532294467319546e-05`, `dropout=0.2903271520872891`, `batch_size=64`.

### [2026-02-25] Optuna best summary (Manual studies)
- **Status:** completed
- **Stage:** analysis
- **Source:** Manual labels
- **Command:**
  ```powershell
  python .\scripts\optuna_best_summary.py .\outputs\optuna_manual_labels_studies\manual_mlp_search
  python .\scripts\optuna_best_summary.py .\outputs\optuna_manual_labels_studies\manual_mlp_search_2
  python .\scripts\optuna_best_summary.py .\outputs\optuna_manual_labels_studies\manual_mlp_search_3
  ```
- **Outputs:**
  - outputs/optuna_manual_labels_studies/manual_mlp_search
  - outputs/optuna_manual_labels_studies/manual_mlp_search_2
  - outputs/optuna_manual_labels_studies/manual_mlp_search_3
- **Result summary:**
  - `manual_mlp_search`: best AUPRC `0.649525`, top_10 std `0.001991`, all-trial std `0.033037`, best F1 `0.321510`
  - `manual_mlp_search_2`: best AUPRC `0.652822`, top_10 std `0.001607`, all-trial std `0.036559`, best F1 `0.094118`
  - `manual_mlp_search_3`: best AUPRC `0.646614`, top_10 std `0.000529`, all-trial std `0.034094`, best F1 `0.222898`
- **Notes:**
  - Peak AUPRC is highest in `_2`, but with very low best F1-macro in this summary.
  - Stability concentration is strongest in `_3` (lowest top-k std, smallest peak gap, largest near-best set).
  - Balanced candidate frozen manual hyperparameters for scaling study: from `_1` (`hidden_dims=[4096,64]`, `learning_rate=7.96315959107497e-05`, `weight_decay=1.5712861337496863e-05`, `dropout=0.048751226478266366`, `batch_size=64`).
  - Alternative strict-stability candidate: from `_3` (`hidden_dims=[512,128]`, `learning_rate=5.462038715569678e-05`, `weight_decay=0.0006147167109835951`, `dropout=0.28605355806129484`, `batch_size=64`).

### [2026-02-25] Phase 1 Item 1: Representation checks (HPO MLP size)
- **Status:** completed
- **Plan phase:** Phase 1
- **Plan item:** 1
- **Stage:** analysis
- **Source:** GPT labels, Manual labels
- **Command:**
  HPO runs with Optuna, searching over MLP depth (up to 5 layers) and hidden sizes.
- **Config:** configs/optuna_gpt_labels_config.yaml, configs/optuna_manual_labels_config.yaml
- **Inputs:** data/manifests/gpt/budget_splits, data/manifests/manual/budget_splits
- **Outputs:** outputs/optuna_gpt_labels_studies/*, outputs/optuna_manual_labels_studies/*
- **Reason:** Test if larger MLP heads improve performance, to check for feature bottleneck.
- **Result summary:** Larger MLPs (3–5 layers, bigger hidden sizes) did not outperform 2-layer MLPs; HPO consistently selected smaller heads as optimal.
- **Conclusion:** No significant gain from increasing MLP size—representation bottleneck is likely, not classifier capacity.
### [2026-02-25] Phase 0: Representation check (HPO MLP size)
- **Status:** completed
- **Plan phase:** Phase 0
- **Plan item:** (embedded)
- **Stage:** analysis
- **Source:** GPT labels, Manual labels
- **Command:**
  HPO runs with Optuna, searching over MLP depth (up to 5 layers) and hidden sizes.
- **Config:** configs/optuna_gpt_labels_config.yaml, configs/optuna_manual_labels_config.yaml
- **Inputs:** data/manifests/gpt/budget_splits, data/manifests/manual/budget_splits
- **Outputs:** outputs/optuna_gpt_labels_studies/*, outputs/optuna_manual_labels_studies/*
- **Reason:** Test if larger MLP heads improve performance, to check for feature bottleneck (representation check is part of HPO).
- **Result summary:** Larger MLPs (3–5 layers, bigger hidden sizes) did not outperform 2-layer MLPs; HPO consistently selected smaller heads as optimal.
- **Conclusion:** No significant gain from increasing MLP size—representation bottleneck is likely, not classifier capacity.


## Phase 2 — Scaling Law Study (Core Objective)

### Phase 2 - Item 1: Label-budget ladder

### [2026-02-25] Budget manifests generation (GPT labels)
- **Status:** completed
- **Stage:** budget-splits
- **Source:** GPT labels
- **Command:**
  ```powershell
  python .\scripts\generate_budget_manifests.py --config-name .\configs\optuna_gpt_labels_config.yaml --budgets 50,100,200,500,1000,5000,10000,20000 --seeds 42,1234,2025 --copy-validation
  ```
- **Config:** configs/optuna_gpt_labels_config.yaml
- **Outputs:** data/manifests/gpt/budget_splits
- **Reason:** Create scaling-law train subsets with fixed validation.
- **Notes:** Stratified sampling default enabled in script.

### [2026-02-25] Budget manifests generation (Manual labels)
- **Status:** completed
- **Stage:** budget-splits
- **Source:** Manual labels
- **Command:**
  ```powershell
  python .\scripts\generate_budget_manifests.py --config-name .\configs\optuna_manual_labels_config.yaml --budgets 10,20,50,100,200,300,500,800 --seeds 42,1234,2025 --copy-validation
  ```
- **Config:** configs/optuna_manual_labels_config.yaml
- **Outputs:** data/manifests/manual/budget_splits
- **Reason:** Create scaling-law train subsets for manual labels with fixed validation.
- **Notes:** Stratified sampling default enabled in script.

### Phase 2 - Item 1: Label-budget ladder (training execution)

### [2026-02-25] Budget training sweep completed (GPT labels)
- **Status:** completed
- **Stage:** training
- **Source:** GPT labels
- **Command (pattern):**
  ```powershell
  python -m classification.train --config-name gpt_labels_config.yaml data.train_manifest=train_n<BUDGET>_s<SEED>.csv hydra.job.name=gpt_budget hydra.run.dir=outputs/gpt_budget/train_n<BUDGET>_s<SEED>
  ```
- **Config:** configs/gpt_labels_config.yaml
- **Inputs:** data/manifests/gpt/budget_splits
- **Outputs:** outputs/gpt_budget/*
- **Reason:** Run full GPT scaling-law training sweep across all generated budgets and seeds.
- **Result summary:** Completed all GPT budget split training runs.

### [2026-02-25] Budget training sweep completed (Manual labels)
- **Status:** completed
- **Stage:** training
- **Source:** Manual labels
- **Command (example):**
  ```powershell
  python -m classification.train --config-name manual_labels_config.yaml data.train_manifest=test_manual_train_n800_s1234.csv hydra.job.name=manual_budget hydra.run.dir=outputs/manual_budget/train_n800_s1234
  ```
- **Config:** configs/manual_labels_config.yaml
- **Inputs:** data/manifests/manual/budget_splits
- **Outputs:** outputs/manual_budget/*
- **Reason:** Run full manual-label scaling-law training sweep across all generated budgets and seeds.
- **Result summary:** Completed all manual budget split training runs.

### Phase 2 - Item 2: Fit scaling trends (evaluation + aggregation artifacts)

### [2026-02-25] Evaluation + aggregation completed (GPT labels)
- **Status:** completed
- **Stage:** evaluation
- **Source:** GPT labels
- **Command (pattern):**
  ```powershell
  python .\scripts\evaluate_and_aggregate_runs.py --runs-root .\outputs\gpt_budget --test-manifest-dir .\data\manifests\manual\ --source gpt --output-prefix gpt_labels
  ```
- **Outputs:** outputs/aggregated_results/gpt_labels_per_run.csv, outputs/aggregated_results/gpt_labels_by_budget.csv
- **Reason:** Evaluate all GPT budget runs on test manifests and aggregate by budget across seeds.
- **Result summary:** Aggregation command executed for GPT runs.

### [2026-02-25] Evaluation + aggregation completed (Manual labels)
- **Status:** completed
- **Stage:** evaluation
- **Source:** Manual labels
- **Command (example):**
  ```powershell
  python .\scripts\evaluate_and_aggregate_runs.py --runs-root .\outputs\manual_budget --test-manifest-dir .\data\manifests\manual\ --source manual --output-prefix manual_labels
  ```
- **Outputs:** outputs/aggregated_results/manual_labels_per_run.csv, outputs/aggregated_results/manual_labels_by_budget.csv
- **Reason:** Evaluate all manual budget runs on test manifests and aggregate by budget across seeds.
- **Result summary:** Aggregation command executed for manual runs.

