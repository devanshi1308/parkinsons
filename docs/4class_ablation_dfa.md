# 4-class accuracy push and DFA feature (scripts 42, 43)

Everything below is subject-grouped (`StratifiedGroupKFold` by `subject_id`), 5 outer folds, 5 seeds (42-46).
Imputation and scaling are fit inside each training fold. Provenance: `scripts/42_add_dfa_feature.py`,
`scripts/43_4class_accuracy_push.py`; CSVs `ablation_4class_runs.csv`, `ablation_4class_results.csv`,
`nested_4class_results.csv`, `nested_4class_selection_counts.csv`.

## 1. DFA feature (script 42)

DFA scaling exponents were computed for all 306 trials (0 failures): `dfa_alpha_L`, `dfa_alpha_R`,
`dfa_alpha_combined`, written to `features_v3_dfa.csv` (`features_v3.csv` is untouched).

| Stage | n trials | alpha L | alpha R | alpha combined |
|---|---|---|---|---|
| Healthy | 92 | 1.135 | 1.133 | 0.630 |
| Mild | 109 | 1.141 | 1.143 | 0.654 |
| Moderate | 81 | 1.142 | 1.141 | 0.657 |
| Severe | 24 | 1.215 | 1.231 | 0.768 |

Interpretation caveats (these matter for how the paper describes it):
- **This is not the stride-interval DFA of the PD literature.** The classic result (Hausdorff et al.) applies DFA to the
  stride-interval time series with box sizes counted in strides. Script 42 applies it to the raw VGRF waveform with boxes
  of 16-400 samples (0.16-4 s), so it mostly measures within-stride waveform structure. The literature's expected
  direction (lower alpha in PD) should not be assumed to transfer, and here Severe trials actually show a higher alpha.
- `dfa_alpha_combined` (about 0.66) is much lower than the single-foot values (about 1.14) because TotalL + TotalR is nearly
  constant while one foot or the other is loaded, so the sum is anti-persistent. It is a different quantity, not an error.
- The Severe difference rests on 24 trials from 10 subjects, so it is suggestive at best.

## 2. Feature-family ablation (Part A)

Families were taken from the real per-version feature files (v1 = 55 columns in the v3 table, v2 adds 106, v3 adds 41
including demographics and walking speed, DFA adds 3). Mean accuracy over 5 seeds x 5 folds:

| Classifier | demo+speed only | v1 | v1+v2 | v1+v2+v3 | +DFA |
|---|---|---|---|---|---|
| Random Forest | 0.367 | 0.415 | 0.467 | **0.483** | 0.475 |
| HistGradientBoosting | 0.314 | 0.402 | 0.445 | 0.479 | **0.485** |
| Bagging | 0.361 | 0.405 | 0.446 | 0.473 | 0.474 |
| Logistic Regression | 0.339 | 0.405 | 0.413 | 0.446 | 0.441 |
| Decision Tree | 0.376 | 0.353 | 0.394 | 0.429 | 0.436 |
| SVM (linear) | 0.367 | 0.394 | 0.411 | 0.409 | 0.409 |
| Naive Bayes | 0.364 | 0.405 | 0.386 | 0.372 | 0.371 |

Reference: always predicting the largest class (Mild) gives 0.356 trial-level accuracy.

Findings:
1. **Each family up to v3 helps the better models** (best-model accuracy 0.415 -> 0.467 -> 0.483), the same ordering the
   binary work found. Naive Bayes and the linear SVM do not benefit from the extra features.
2. **Demographics + speed alone is essentially at the majority baseline** (0.356-0.376) for 4-class, so unlike some binary
   cohorts the gait-signal features are doing the work here.
3. **DFA adds nothing.** Paired over classifier x seed, the mean accuracy change is -0.0001 and it is positive in only
   13 of 35 pairs. Report it as a negative result, as with v4.

## 3. Nested CV (Part B)

The inner loop picks the (feature set, classifier) pair by macro-F1 using only outer-training data; the outer fold is
scored once.

- **De-biased 4-class accuracy: 0.422 (SD across seeds 0.018); macro-F1: 0.327 (SD 0.018).**
- Per-seed accuracy: 0.435, 0.441, 0.405, 0.400, 0.428.
- The selected feature set varied from fold to fold (v1+v2+v3+dfa 10 times, v1+v2+v3 9, v1+v2 3, v1 2, demo+speed 1 of 25),
  and the selected classifier varied too, so there is no single stable winner. That instability is itself consistent with
  a sample-size ceiling.

Comparison with the earlier number (44.4%, script 17): the two are within noise of each other. Note that the earlier SD
(10.8) is across folds and the one here (1.8) is across seeds, so they are not comparable as spreads. The candidate pools
also differ (script 17 included LightGBM and XGBoost). We recommend citing a single figure in the paper, about 42-44%,
and saying plainly that it sits roughly 6-9 points above the 35.6% majority-class rate and that macro-F1 (about 0.33)
shows the minority classes are mostly missed.

Bottom line: the dedicated 4-class push confirms rather than breaks the sample-size ceiling. More feature families do
not rescue the Severe and Moderate classes.
