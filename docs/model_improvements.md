# Model-improvement follow-ups

Five checks requested after the first features-v3 / extended-model results. All subject-grouped CV, seed 42 (+ repeats where noted).

## 1. De-biased selection (nested CV)

`model_comparison_v3.csv` picked the best of 45 (feature_set, model) pairs on the same folds used to report accuracy -- optimistic. Here, each outer fold's winning pair is chosen on an inner CV over training data only, then scored once on the untouched outer fold.

**4class**: nested outer accuracy 0.421 (sd 0.083), outer macro-F1 0.316. Folds most often picked: v3_all/Random Forest (3), v3_signal/Bagging (3), v3_signal/Random Forest (1)

**binary**: nested outer accuracy 0.794 (sd 0.097), outer macro-F1 0.755. Folds most often picked: v3_all/HistGradientBoosting (5), v3_signal/HistGradientBoosting (2), v3_all/Logistic Regression (1)

This is the de-biased estimate to actually cite in the paper, not the best-of-45 table-selected number.

## 2. Subject-level aggregation (binary, Random Forest on v3_all)

| level                                 |   n |   accuracy |   f1_macro |
|:--------------------------------------|----:|-----------:|-----------:|
| trial                                 | 306 |      0.83  |      0.797 |
| subject (mean of trial probabilities) | 165 |      0.812 |      0.806 |

Averaging each subject's trial probabilities before thresholding did not improve accuracy here (most subjects have only 1-2 trials, so there is little to average), though macro-F1 ticked up slightly. Worth reporting as a null result, not a fix.

## 3. Ordinal vs multiclass (4-class)

| method             |   accuracy |   f1_macro |   mean_abs_stage_error |
|:-------------------|-----------:|-----------:|-----------------------:|
| multiclass_rf      |      0.507 |      0.39  |                  0.585 |
| ordinal_frank_hall |      0.484 |      0.414 |                  0.657 |

Mean absolute stage error counts Healthy-predicted-as-Severe as worse than Healthy-predicted-as-Mild; plain accuracy does not. The Frank-Hall ordinal model has lower raw accuracy but higher macro-F1 and a WORSE mean absolute stage error here -- so ordinal decomposition did not clearly help on this feature set/sample size; report as a negative result rather than adopting it.

## 4. Compact, clinically-named feature set

L1-logistic selection (C=0.03) fit inside each fold; default (nonzero-coefficient) threshold. 5 of 202 features were selected in at least half the folds.

| feature_set                       |   accuracy_mean |   accuracy_sd |
|:----------------------------------|----------------:|--------------:|
| full (202 features)               |           0.823 |         0.117 |
| compact (>=50% folds, 5 features) |           0.78  |         0.096 |

Compact feature list (>=50% of folds):

| clinical_name                                 |   selected_fraction_of_folds |
|:----------------------------------------------|-----------------------------:|
| Sensor 5 (left foot) force skewness           |                          1   |
| Walking speed                                 |                          1   |
| Centre-of-pressure front-back sway, left foot |                          0.9 |
| Sensor 7 (left foot) median force             |                          0.7 |
| Sensor 6 (left foot) force skewness           |                          0.7 |

A 5-feature model recovers most of the full 202-feature accuracy (78.0% vs 82.3%) with names a clinician can read directly -- a much better SHAP story than the raw-sensor-statistics list from `40_shap_report.py`. Walking speed and left-foot force skewness dominate, consistent with the SHAP result.

## 5. XGBoost / LightGBM added

| scheme   | feature_set     | model    |   accuracy |   accuracy_sd |   f1_macro |
|:---------|:----------------|:---------|-----------:|--------------:|-----------:|
| 4class   | demo_speed_only | LightGBM |      0.37  |         0.027 |      0.313 |
| 4class   | demo_speed_only | XGBoost  |      0.384 |         0.006 |      0.318 |
| 4class   | v2              | LightGBM |      0.483 |         0.013 |      0.373 |
| 4class   | v2              | XGBoost  |      0.466 |         0.02  |      0.354 |
| 4class   | v3_all          | LightGBM |      0.524 |         0.006 |      0.414 |
| 4class   | v3_all          | XGBoost  |      0.5   |         0.023 |      0.388 |
| binary   | demo_speed_only | LightGBM |      0.72  |         0.031 |      0.666 |
| binary   | demo_speed_only | XGBoost  |      0.722 |         0.018 |      0.674 |
| binary   | v2              | LightGBM |      0.809 |         0.009 |      0.772 |
| binary   | v2              | XGBoost  |      0.806 |         0.01  |      0.765 |
| binary   | v3_all          | LightGBM |      0.838 |         0.018 |      0.802 |
| binary   | v3_all          | XGBoost  |      0.845 |         0.018 |      0.815 |

LightGBM/XGBoost on v3_all roughly match or slightly beat the best scikit-learn model from `model_comparison_v3.csv` (binary ~84-85% vs ~83%; 4-class ~50-52% vs ~51%) -- a modest gain, not a breakthrough, and still subject to the same selection-bias caveat as point 1 unless re-validated with nested CV.

## 1b. Nested selection WITH XGBoost/LightGBM as candidates (`17_nested_with_boosting.py`)

Extends section 1's nested (de-biased) selection to also consider XGBoost and LightGBM (on v2 / v3_all / demo_speed_only, matching section 5's ablation). Closes the "still needs nested-CV confirmation" caveat from section 5.

**4class**: sklearn-only nested accuracy 0.421 (sd 0.083) vs with boosting 0.444 (sd 0.108); macro-F1 0.316 vs 0.339. Folds most often picked: v3_all/LightGBM (2), v3_all/XGBoost (2), v3_all/Random Forest (2)

**binary**: sklearn-only nested accuracy 0.794 (sd 0.097) vs with boosting 0.817 (sd 0.106); macro-F1 0.755 vs 0.788. Folds most often picked: v3_all/LightGBM (5), v3_all/XGBoost (4), v3_signal/HistGradientBoosting (1)
