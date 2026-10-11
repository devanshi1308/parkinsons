# Generalization, calibration and clinical metrics (gait, features v3)

Three checks left open by earlier scripts. Subject-grouped throughout. See the docstring in `scripts/23_generalization_calibration_clinical.py` for exact protocols.

## A. Leave-one-cohort-out

Train on two sub-cohorts, test on the third. `majority_baseline` is the accuracy of always predicting the held-out cohort's most common class, and `demo_speed_only` is a model with no gait-signal features at all.

### binary

| held_out_cohort   |   n_test | feature_set     | model         |   accuracy |   balanced_accuracy |   majority_baseline |   auc |
|:------------------|---------:|:----------------|:--------------|-----------:|--------------------:|--------------------:|------:|
| Ga                |      113 | v3_all          | Random Forest |      0.823 |               0.75  |               0.664 | 0.886 |
| Ga                |      113 | v3_all          | LightGBM      |      0.805 |               0.736 |               0.664 | 0.873 |
| Ga                |      113 | compact5        | Random Forest |      0.761 |               0.729 |               0.664 | 0.888 |
| Ga                |      113 | compact5        | LightGBM      |      0.796 |               0.769 |               0.664 | 0.854 |
| Ga                |      113 | demo_speed_only | Random Forest |      0.726 |               0.702 |               0.664 | 0.764 |
| Ga                |      113 | demo_speed_only | LightGBM      |      0.735 |               0.696 |               0.664 | 0.781 |
| Ju                |      129 | v3_all          | Random Forest |      0.829 |               0.742 |               0.806 | 0.893 |
| Ju                |      129 | v3_all          | LightGBM      |      0.899 |               0.846 |               0.806 | 0.914 |
| Ju                |      129 | compact5        | Random Forest |      0.775 |               0.724 |               0.806 | 0.827 |
| Ju                |      129 | compact5        | LightGBM      |      0.791 |               0.749 |               0.806 | 0.814 |
| Ju                |      129 | demo_speed_only | Random Forest |      0.899 |               0.77  |               0.806 | 0.899 |
| Ju                |      129 | demo_speed_only | LightGBM      |      0.829 |               0.666 |               0.806 | 0.867 |
| Si                |       64 | v3_all          | Random Forest |      0.703 |               0.708 |               0.547 | 0.765 |
| Si                |       64 | v3_all          | LightGBM      |      0.719 |               0.728 |               0.547 | 0.814 |
| Si                |       64 | compact5        | Random Forest |      0.625 |               0.631 |               0.547 | 0.757 |
| Si                |       64 | compact5        | LightGBM      |      0.688 |               0.697 |               0.547 | 0.765 |
| Si                |       64 | demo_speed_only | Random Forest |      0.625 |               0.631 |               0.547 | 0.715 |
| Si                |       64 | demo_speed_only | LightGBM      |      0.594 |               0.608 |               0.547 | 0.644 |

### 4class

| held_out_cohort   |   n_test | feature_set     | model         |   accuracy |   balanced_accuracy |   majority_baseline |
|:------------------|---------:|:----------------|:--------------|-----------:|--------------------:|--------------------:|
| Ga                |      113 | v3_all          | Random Forest |      0.469 |               0.374 |               0.363 |
| Ga                |      113 | v3_all          | LightGBM      |      0.425 |               0.336 |               0.363 |
| Ga                |      113 | compact5        | Random Forest |      0.531 |               0.411 |               0.363 |
| Ga                |      113 | compact5        | LightGBM      |      0.46  |               0.403 |               0.363 |
| Ga                |      113 | demo_speed_only | Random Forest |      0.363 |               0.301 |               0.363 |
| Ga                |      113 | demo_speed_only | LightGBM      |      0.372 |               0.293 |               0.363 |
| Ju                |      129 | v3_all          | Random Forest |      0.411 |               0.387 |               0.419 |
| Ju                |      129 | v3_all          | LightGBM      |      0.558 |               0.471 |               0.419 |
| Ju                |      129 | compact5        | Random Forest |      0.403 |               0.37  |               0.419 |
| Ju                |      129 | compact5        | LightGBM      |      0.388 |               0.352 |               0.419 |
| Ju                |      129 | demo_speed_only | Random Forest |      0.411 |               0.418 |               0.419 |
| Ju                |      129 | demo_speed_only | LightGBM      |      0.302 |               0.312 |               0.419 |
| Si                |       64 | v3_all          | Random Forest |      0.484 |               0.431 |               0.453 |
| Si                |       64 | v3_all          | LightGBM      |      0.547 |               0.442 |               0.453 |
| Si                |       64 | compact5        | Random Forest |      0.5   |               0.479 |               0.453 |
| Si                |       64 | compact5        | LightGBM      |      0.516 |               0.455 |               0.453 |
| Si                |       64 | demo_speed_only | Random Forest |      0.469 |               0.384 |               0.453 |
| Si                |       64 | demo_speed_only | LightGBM      |      0.531 |               0.431 |               0.453 |

## B. Calibration of the binary PD-risk output

| model         | calibration   |   brier |   log_loss |    ece |    auc |   accuracy |
|:--------------|:--------------|--------:|-----------:|-------:|-------:|-----------:|
| Random Forest | raw           |  0.1281 |     0.41   | 0.0791 | 0.8855 |     0.8301 |
| Random Forest | sigmoid       |  0.123  |     0.39   | 0.0518 | 0.8824 |     0.8203 |
| Random Forest | isotonic      |  0.1308 |     0.4505 | 0.0624 | 0.8656 |     0.8039 |
| LightGBM      | raw           |  0.1304 |     0.4423 | 0.101  | 0.8939 |     0.817  |
| LightGBM      | sigmoid       |  0.1148 |     0.3676 | 0.0361 | 0.8965 |     0.8366 |
| LightGBM      | isotonic      |  0.1168 |     0.3627 | 0.0503 | 0.8921 |     0.8268 |

Lower Brier / log-loss / ECE is better. Figure: `docs/figures/calibration_reliability.png`.

## C. Binary clinical metrics, mean and SD across 10 folds x 3 seeds (threshold 0.5)

| model         |   sensitivity_mean |   sensitivity_std |   specificity_mean |   specificity_std |   ppv_mean |   ppv_std |   npv_mean |   npv_std |   accuracy_mean |   accuracy_std |   auc_mean |   auc_std |
|:--------------|-------------------:|------------------:|-------------------:|------------------:|-----------:|----------:|-----------:|----------:|----------------:|---------------:|-----------:|----------:|
| LightGBM      |              0.897 |             0.116 |              0.699 |             0.186 |      0.878 |     0.069 |      0.792 |     0.193 |           0.838 |          0.094 |      0.911 |     0.073 |
| Random Forest |              0.89  |             0.13  |              0.688 |             0.159 |      0.872 |     0.059 |      0.787 |     0.2   |           0.829 |          0.095 |      0.894 |     0.076 |
