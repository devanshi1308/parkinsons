# Statistical robustness of the leakage finding

Three checks: (A) is the leaky-vs-grouped gap statistically significant, not just a 5-seed eyeball estimate; (B) is the honest (grouped) accuracy actually above chance; (C) does the gap track model capacity to memorize subjects (a mechanistic argument).

## A. Paired significance (20 seeds, Wilcoxon signed-rank + paired Cohen's d)

| modality   | scheme   | classifier         |   n_seeds |   leaky_mean |   grouped_mean |   gap_pts |   relative_inflation_pct |   wilcoxon_stat |   wilcoxon_p |   cohens_d_paired |
|:-----------|:---------|:-------------------|----------:|-------------:|---------------:|----------:|-------------------------:|----------------:|-------------:|------------------:|
| gait       | 4class   | Decision Tree      |        20 |       0.6494 |         0.4051 |   24.4283 |                  60.3062 |               0 |       0      |            5.8158 |
| gait       | 4class   | SVM                |        20 |       0.7203 |         0.4244 |   29.5867 |                  69.7165 |               0 |       0      |           12.5459 |
| gait       | 4class   | Ensemble (Bagging) |        20 |       0.7605 |         0.4607 |   29.9879 |                  65.0965 |               0 |       0      |            9.9423 |
| gait       | 4class   | Bayes (GaussianNB) |        20 |       0.5298 |         0.3899 |   13.9936 |                  35.8906 |               0 |       0      |            6.6989 |
| gait       | 3class   | Decision Tree      |        20 |       0.6766 |         0.4307 |   24.5935 |                  57.1015 |               0 |       0.0001 |            8.1293 |
| gait       | 3class   | SVM                |        20 |       0.7006 |         0.4362 |   26.4468 |                  60.6338 |               0 |       0      |           10.2641 |
| gait       | 3class   | Ensemble (Bagging) |        20 |       0.7866 |         0.4808 |   30.5812 |                  63.611  |               0 |       0      |            9.1741 |
| gait       | 3class   | Bayes (GaussianNB) |        20 |       0.5419 |         0.4612 |    8.0731 |                  17.5046 |               0 |       0      |            4.5209 |
| gait       | binary   | Decision Tree      |        20 |       0.7968 |         0.7168 |    7.9978 |                  11.1576 |               1 |       0      |            2.0135 |
| gait       | binary   | SVM                |        20 |       0.8395 |         0.7386 |   10.0898 |                  13.6602 |               0 |       0      |            3.4127 |
| gait       | binary   | Ensemble (Bagging) |        20 |       0.8657 |         0.7823 |    8.3392 |                  10.6603 |               0 |       0      |            3.4652 |
| gait       | binary   | Bayes (GaussianNB) |        20 |       0.7475 |         0.7257 |    2.1796 |                   3.0036 |               1 |       0.0001 |            1.3489 |
| voice      | binary   | Decision Tree      |        20 |       0.6282 |         0.5312 |    9.7019 |                  18.2657 |               0 |       0.0001 |            4.9173 |
| voice      | binary   | SVM                |        20 |       0.6453 |         0.5522 |    9.3125 |                  16.8655 |               0 |       0.0001 |            3.5441 |
| voice      | binary   | Ensemble (Bagging) |        20 |       0.6938 |         0.5481 |   14.5625 |                  26.5678 |               0 |       0      |            7.7537 |
| voice      | binary   | Bayes (GaussianNB) |        20 |       0.5864 |         0.5419 |    4.4471 |                   8.2062 |               0 |       0      |            2.4763 |

Cohen's d >0.8 is conventionally "large"; the leakage effect is large for every classifier/scheme/modality combination tested here.

## B. Permutation (chance-level) test on the honest protocol

| modality   | scheme   | classifier         |   n_permutations |   observed_accuracy |   null_mean |   null_sd |   null_max |   p_value |
|:-----------|:---------|:-------------------|-----------------:|--------------------:|------------:|----------:|-----------:|----------:|
| gait       | 4class   | Ensemble (Bagging) |              100 |              0.4884 |      0.3828 |    0.0461 |     0.5191 |    0.0198 |
| gait       | binary   | Ensemble (Bagging) |              100 |              0.771  |      0.512  |    0.0499 |     0.7056 |    0.0099 |
| voice      | binary   | Ensemble (Bagging) |              100 |              0.5452 |      0.4894 |    0.0569 |     0.65   |    0.1485 |

p-values are the fraction of subject-level label permutations that scored at least as well as the real labels; p<0.01 (i.e. 0/100 or close to it) means the grouped/honest accuracy is not explained by chance, even though it is far below the leaky numbers.

## C. Does the leakage gap track model capacity?

| scheme   | classifier         |   accuracy_gap_pts | modality   |   capacity_rank |
|:---------|:-------------------|-------------------:|:-----------|----------------:|
| 3class   | Bayes (GaussianNB) |               9.22 | gait       |               0 |
| 3class   | SVM                |              25.83 | gait       |               1 |
| 3class   | Decision Tree      |              26.2  | gait       |               2 |
| 3class   | Ensemble (Bagging) |              31.62 | gait       |               3 |
| 4class   | Bayes (GaussianNB) |              15.14 | gait       |               0 |
| 4class   | SVM                |              29.54 | gait       |               1 |
| 4class   | Decision Tree      |              26.57 | gait       |               2 |
| 4class   | Ensemble (Bagging) |              29.79 | gait       |               3 |
| binary   | Bayes (GaussianNB) |               1.39 | gait       |               0 |
| binary   | SVM                |              10.72 | gait       |               1 |
| binary   | Decision Tree      |               7.81 | gait       |               2 |
| binary   | Ensemble (Bagging) |               9.63 | gait       |               3 |
| binary   | Bayes (GaussianNB) |               3.81 | voice      |               0 |
| binary   | SVM                |               8.67 | voice      |               1 |
| binary   | Decision Tree      |              10.37 | voice      |               2 |
| binary   | Ensemble (Bagging) |              15.23 | voice      |               3 |

Classifiers are ordered by their ability to carve out subject-specific decision boundaries (Naive Bayes < linear SVM < single Decision Tree < Bagging of trees). If the gap grows roughly in this order within each scheme/modality, that supports a mechanistic story (higher-capacity models more easily memorize a subject's signature) rather than leakage being an unexplained statistical artifact.