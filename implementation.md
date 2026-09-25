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
- [ ] Parse `demographics.xls` — extract subject ID, age, sex, weight, height, H&Y stage

## Phase 2 — Feature extraction (reproducing baseline)
- [ ] Statistical features: mean, median, SD per sensor (16 sensors × 3 stats)
- [ ] Spatiotemporal/kinematic features: step time, stance time, swing time, swing/stance
      ratio, stride time, cadence, speed, step length, stride length
- [ ] Feature selection: correlation-based (Pearson) — reduce to most discriminative subset
- [ ] Merge features + H&Y labels into `data/processed/master_dataset.csv`

## Phase 3 — Baseline reproduction
- [ ] Train/test split with 10-fold cross-validation
- [ ] Train DT, SVM, Ensemble, Bayes classifiers (statistical features)
- [ ] Repeat for kinematic features
- [ ] Compute accuracy, sensitivity, specificity, PPV, NPV, F-score, confusion matrix, ROC/AUC
- [ ] Compare against paper's reported numbers (Table 5–9) — confirm pipeline is correctly reproduced

## Phase 4 — Novel contribution (to be defined — see rules.md decision log)
- [x] Identify and lock in the specific gap/angle
- [x] Implement the new method/feature/approach
- [x] Run same evaluation protocol as Phase 3 for fair comparison
- [x] Document improvement (or honest null result) vs. baseline

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