# Implementation Plan

## Phase 0 — Setup (done / in progress)
- [x] Project folder structure created
- [x] Dataset downloaded from PhysioNet
- [x] `.gitignore` + `data/README.md` for reproducibility without committing raw data
- [x] Repo pushed to GitHub

## Phase 1 — Data organization & cleaning
- [x] `01_organize_raw.py` — sort raw files by cohort (Ga/Ju/Si) and group (Co/Pt)
- [x] `02_clean_signals.py` — drop NaNs/duplicates, trim start/end gait-cycle edge effects,
      clip noise, validate 19-column format
- [x] Sanity check: row counts, missing files, corrupt files logged and resolved (sanity check via check_cleaned_data.py, demographics parsed via 04_merge_labels.py)
- [x] Parse `demographics.xls` — subject ID, H&Y stage (`04_merge_labels.py` -> `subject_labels.csv`; age/sex/height/weight not yet used as features)

## Phase 2 — Feature extraction (reproducing baseline)
- [x] Statistical features: mean, median, SD per sensor (`03_extract_features.py`); v2 adds min/max/range/IQR/skew/kurtosis
- [x] Spatiotemporal/kinematic features: stance/swing/stride time, ratios, cadence, stride-time CV, symmetry, COP (`03_extract_features_v2.py`)
- [x] Speed from demographics (`Speed_01`); stride/step length as derived estimates (speed x stride time), labelled as estimates (`03_extract_features_v3.py`)
- [x] Feature selection: variance + correlation (|r|>0.95) filter (`07_train_multi_scheme.py`)
- [x] Merge features + H&Y labels (done in-memory in the training scripts; no separate `master_dataset.csv`)

## Phase 3 — Baseline reproduction
- [x] 10-fold cross-validation, subject-grouped (`05_train_classifiers.py`)
- [x] Train DT, SVM, Ensemble (Bagging), Bayes classifiers (`05`, `07`)
- [x] Repeat for kinematic features (v2 feature set)
- [x] Compute accuracy, sensitivity, specificity, PPV, NPV, F-score, confusion matrix, AUC (`08_full_metrics.py`); ROC plots not yet made
- [x] Compare against paper's reported numbers (Table 8-9) (`15_baseline_comparison.py`, `docs/baseline_comparison.md`) — we do NOT reproduce the paper's 99.4%; see result below

## Phase 4 — Novel contribution (locked in — see rules.md decision log)
- [x] Identify and lock in the specific gap/angle
- [x] Implement the new method/feature/approach
- [x] Run same evaluation protocol as Phase 3 for fair comparison
- [x] Document improvement (or honest null result) vs. baseline

- [x] Class imbalance experiments (`09_class_imbalance.py`), per-cohort and leave-one-cohort-out (`10`, `11`)
- [x] Leaky vs grouped comparison, 5 seeds, in-fold check, leak audit (`14_leakage_comparison.py`)
- [x] Review Ju cohort investigation output; fixed a demographics-merge bug in the script (`11_investigate_ju_cohort.py`)

## Phase 4b — Extensions (planned)
- [x] Gait features v3: demographics, speed, spectral, wavelet, regularity (autocorrelation, sample entropy), double support (`03_extract_features_v3.py`); DFA not implemented
- [x] More models: Random Forest, Extra Trees, HistGradientBoosting, logistic regression, soft voting; 5 feature sets x 9 models x 3 schemes x 3 seeds (`12_train_models_v3.py`; XGBoost/LightGBM skipped, need libomp)
- [ ] Optional sliding-window LSTM with augmentation, grouped CV (`13_lstm_windows.py`)
- [x] SHAP global + per-patient explanations, out-of-fold, clinical names (`40_shap_report.py`)
- [x] SHAP on the compact 5-feature set (binary) and SHAP for the 4-class model, per-stage top
      features (`41_shap_4class_and_compact.py`)
- [ ] PADS wrist model: BLOCKED -- in-sandbox download abandoned (network too throttled, ~4-16 KB/s measured); needs user to download locally (see data-README.md); then group counts, features, grouped CV, leaky-vs-grouped
- [x] Model-improvement follow-ups: nested selection (de-biased), subject-level aggregation,
      ordinal vs multiclass, compact feature set, XGBoost/LightGBM (`16_model_improvements.py`,
      `docs/model_improvements.md`)
- [x] Nested-CV selection extended to include XGBoost/LightGBM as candidates (`17_nested_with_boosting.py`)
- [x] UCI speech voice model: downloaded, loaded, grouped CV, leaky-vs-grouped, SHAP
      (`30_voice_load.py`, `31_voice_leakage_comparison.py`, `32_voice_shap.py`)
- [ ] Late-fusion demo + patient report (`50_fusion_demo.py`); demo only, no fused accuracy claim
- [x] Combined gait+voice headline leakage figure (`19_combined_leakage_figure.py`)
- [ ] Remaining figures: confusion matrix, ROC, VGRF/COP plots (leakage gap and all SHAP plots already done)

- [x] Statistical robustness: paired significance (Wilcoxon + Cohen's d, 20 seeds), permutation
      (chance-level) test, capacity-vs-gap table, for both gait and voice (`18_statistical_robustness.py`,
      `docs/statistical_robustness.md`)

- [x] Fold-count sensitivity (5 vs 10 folds) and combined gait+voice headline figure
      (`20_fold_sensitivity.py`, `19_combined_leakage_figure.py`, `docs/figures/combined_leakage_gap.png`)

## Phase 5 — Paper writing
- [x] Methods + Results drafted with full numeric/statistical backing (`docs/paper_draft_methods_results.md`)
- [ ] Abstract
- [ ] Introduction + motivation
- [ ] Related work (baseline paper + others from Table 9)
- [ ] Dataset description
- [ ] Methodology (feature extraction + models + novel contribution)
- [ ] Results (tables, figures, comparison to baseline and literature)
- [ ] Discussion + limitations
- [ ] Conclusion
- [ ] References

## Phase 6 — Submission prep
- [ ] Target journal/conference identified
- [ ] Formatting to target's template
- [ ] Plagiarism/similarity check
- [ ] Submit