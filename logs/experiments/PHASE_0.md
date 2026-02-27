# Phase 0 — Hyperparameter Optimization and Protocol Selection

## Optuna search (Manual labels)
- **Status:** completed
- **Stage:** optuna
- **Source:** Manual labels
- **Command:**
  ```powershell
  python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_manual_labels_config.yaml --study-name manual_mlp_search --n-trials 200
  ```
- **Config:** configs/optuna_manual_labels_config.yaml
- **Outputs:** outputs/optuna_manual_labels_studies/manual_mlp_search
- **Reason:** First Optuna run for manual-label tuning.

FIX NEEDED HERE:
python .\scripts\optuna_best_summary.py .\outputs\optuna_manual_labels_studies\manual_mlp_search                           
Study: manual_mlp_search
 Best trial: optuna_mlp_t0161
 Best val AUPRC: 0.637442
 Best val AUROC: 0.733408
 Best val F1-macro: 0.565374
 Hyperparameters:
  - learning_rate: 0.004423456917742595
  - weight_decay: 2.3893022530110874e-05
  - dropout: 0.4652714305458842
  - hidden_dims: [512, 64]
  - batch_size: 64
 Stability (objective across trials):
  - usable_trials: 200 | mean=0.608642 median=0.613212 std=0.018375
  - top_10: mean=0.632130 median=0.631608 std=0.003104 min=0.628257 max=0.637442
  - peak_gap(best-top_10_median): 0.005835
  - near_best_band: >= 0.631068 (5 trials)
  - near_best hidden_dims mode: [512, 64] (4 trials)
  - near_best dropout mode: 0.4652714305458842 (1 trials)

FIX NEEDED HERE:
Aggregated results:

test_test_manual_valid.csv_auprc_mean,test_test_manual_valid.csv_auprc_std,test_test_manual_valid.csv_auprc_count,test_test_manual_valid.csv_auroc_mean,test_test_manual_valid.csv_auroc_std,test_test_manual_valid.csv_auroc_count,test_test_manual_valid.csv_f1_macro_mean,test_test_manual_valid.csv_f1_macro_std,test_test_manual_valid.csv_f1_macro_count
0.5281711952467838,0.013173294455328133,5,0.6605344630596577,0.014045202081829522,5,0.5275507435879038,0.016346606764271784,5

## Optuna search (GPT labels)
- **Status:** completed
- **Stage:** optuna
- **Source:** GPT labels
- **Command:**
  ```powershell
  python .\scripts\optuna_mlp_search.py --config-name .\configs\optuna_gpt_labels_config.yaml --study-name gpt_mlp_search --n-trials 200
  ```
- **Config:** configs/optuna_gpt_labels_config.yaml
- **Outputs:** outputs/optuna_gpt_labels_studies/gpt_mlp_search
- **Reason:** First Optuna run for GPT-label baseline tuning.

FIX NEEDED HERE:
python .\scripts\optuna_best_summary.py .\outputs\optuna_gpt_labels_studies\gpt_mlp_search\                                
Study: gpt_mlp_search
 Best trial: optuna_mlp_t0181
 Best val AUPRC: 0.549311
 Best val AUROC: 0.722103
 Best val F1-macro: 0.560058
 Hyperparameters:
  - learning_rate: 0.0019342483580818265
  - weight_decay: 0.0003896930162075688
  - dropout: 0.2523768381159455
  - hidden_dims: [512, 512]
  - batch_size: 64
 Stability (objective across trials):
  - usable_trials: 200 | mean=0.544475 median=0.545407 std=0.006226
  - top_10: mean=0.548438 median=0.548333 std=0.000314 min=0.548182 max=0.549311
  - peak_gap(best-top_10_median): 0.000979
  - near_best_band: >= 0.543818 (158 trials)
  - near_best hidden_dims mode: [512, 512] (82 trials)
  - near_best dropout mode: 0.2523768381159455 (1 trials)

FIX NEEDED HERE:
Aggregated results:

test_test_manual_valid.csv_auprc_mean,test_test_manual_valid.csv_auprc_std,test_test_manual_valid.csv_auprc_count,test_test_manual_valid.csv_auroc_mean,test_test_manual_valid.csv_auroc_std,test_test_manual_valid.csv_auroc_count,test_test_manual_valid.csv_f1_macro_mean,test_test_manual_valid.csv_f1_macro_std,test_test_manual_valid.csv_f1_macro_count
0.6016768084803295,0.017814753728755763,5,0.7036012341511269,0.00815181972718619,5,0.5236536041047084,0.007536909258011782,5