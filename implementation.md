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
- [ ] Step length, stride length, speed: not possible (no spatial data); optional height-based proxies, labelled as proxies
- [x] Feature selection: variance + correlation (|r|>0.95) filter (`07_train_multi_scheme.py`)
- [x] Merge features + H&Y labels (done in-memory in the training scripts; no separate `master_dataset.csv`)

## Phase 3 — Baseline reproduction
- [x] 10-fold cross-validation, subject-grouped (`05_train_classifiers.py`)
- [x] Train DT, SVM, Ensemble (Bagging), Bayes classifiers (`05`, `07`)
- [x] Repeat for kinematic features (v2 feature set)
- [x] Compute accuracy, sensitivity, specificity, PPV, NPV, F-score, confusion matrix, AUC (`08_full_metrics.py`); ROC plots not yet made
- [ ] Compare against paper's reported numbers (Table 5–9) — confirm pipeline is correctly reproduced

## Phase 4 — Novel contribution (locked in — see rules.md decision log)
- [x] Identify and lock in the specific gap/angle
- [x] Implement the new method/feature/approach
- [x] Run same evaluation protocol as Phase 3 for fair comparison
- [x] Document improvement (or honest null result) vs. baseline

- [x] Class imbalance experiments (`09_class_imbalance.py`), per-cohort and leave-one-cohort-out (`10`, `11`)
- [x] Leaky vs grouped comparison, 5 seeds, in-fold check, leak audit (`14_leakage_comparison.py`)
- [ ] Review Ju cohort investigation output (`11_investigate_ju_cohort.py`)

## Phase 4b — Extensions (planned)
- [ ] Gait features v3: demographics, FFT/wavelet, regularity (autocorrelation, entropy, DFA), double support (`03_extract_features_v3.py`)
- [ ] More models: Random Forest, XGBoost/LightGBM, logistic regression, stacking (`12_train_models_v3.py`)
- [ ] Optional sliding-window LSTM with augmentation, grouped CV (`13_lstm_windows.py`)
- [ ] SHAP global + per-patient explanations, clinically named features (`40_shap_report.py`)
- [ ] PADS wrist model: download, check group counts, features, grouped CV, leaky-vs-grouped (`20`-`22`)
- [ ] UCI speech voice model: features as provided, grouped CV, leaky-vs-grouped (`30`-`32`)
- [ ] Late-fusion demo + patient report (`50_fusion_demo.py`); demo only, no fused accuracy claim
- [ ] Figures: leakage gap (done in 14), confusion matrix, ROC, VGRF/COP plots, SHAP plots

## Phase 5 — Paper writing
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