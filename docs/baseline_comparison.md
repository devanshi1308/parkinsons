# Comparison against the baseline paper (Balaji et al. 2020)

Paper numbers are Table 8 (kinematic-feature cumulative performance, 4-class) and Table 9 (the proposed framework's headline accuracy), copied from `docs/paper.pdf`. Verify against the PDF before publishing. The paper states 10-fold CV repeated 10 times but does not say whether folds were grouped by subject; `trial_level_repeated10` below is our best-effort match to its description, not a confirmed replica, and `subject_grouped` is our honest protocol.

## Kinematic features, 4-class, mean of 10 repeats (%)

| Classifier | Paper (Table 8) | Ours, trial-level (closest match) | Ours, subject-grouped (honest) |
|---|---|---|---|
| Decision Tree | acc 99.4, sens 99.6, spec 99.8, PPV 99.0, F 99.2 | acc 59.5 (sd 2.1), sens 55.1, spec 85.7, PPV 56.7, F 54.4 | acc 41.1 (sd 3.6), sens 34.7, spec 79.2, PPV 34.7, F 33.3 |
| SVM | acc 99.4, sens 99.6, spec 99.8, PPV 99.0, F 99.2 | acc 60.6 (sd 0.9), sens 49.9, spec 85.5, PPV 47.9, F 48.1 | acc 51.9 (sd 1.0), sens 42.5, spec 82.5, PPV 41.1, F 40.2 |
| Bagging | acc 95.2, sens 91.6, spec 98.2, PPV 95.3, F 93.1 | acc 71.1 (sd 1.9), sens 65.1, spec 89.4, PPV 70.1, F 65.8 | acc 48.0 (sd 2.3), sens 38.8, spec 81.1, PPV 37.3, F 36.6 |
| GaussianNB | acc 69.9, sens 42.4, spec 89.1, PPV 65.7, F 72.1 | acc 44.0 (sd 1.2), sens 36.3, spec 80.0, PPV 33.5, F 34.0 | acc 40.6 (sd 1.6), sens 33.6, spec 78.8, PPV 31.4, F 30.1 |

## Statistical features, 4-class, mean of 10 repeats (%)

| Classifier | Ours, trial-level (closest match) | Ours, subject-grouped (honest) |
|---|---|---|
| Decision Tree | acc 62.8 (sd 1.9), sens 60.3, spec 86.7, PPV 61.9, F 59.4 | acc 36.2 (sd 2.6), sens 31.1, spec 77.3, PPV 29.9, F 28.8 |
| SVM | acc 70.4 (sd 1.2), sens 68.3, spec 89.3, PPV 73.2, F 69.0 | acc 41.1 (sd 1.2), sens 39.5, spec 78.9, PPV 33.9, F 34.6 |
| Bagging | acc 71.7 (sd 1.3), sens 67.6, spec 89.6, PPV 72.0, F 68.1 | acc 41.8 (sd 1.7), sens 33.9, spec 78.7, PPV 32.1, F 31.5 |
| GaussianNB | acc 55.8 (sd 1.0), sens 52.9, spec 84.2, PPV 53.3, F 51.2 | acc 40.4 (sd 1.2), sens 37.6, spec 78.8, PPV 32.0, F 32.1 |

## Headline comparison

- Paper's proposed framework (Table 9): **99.4%**
- Our best trial-level, closest-match protocol (kinematic features): **71.1%**
- Our best honest, subject-grouped protocol (kinematic features): **51.9%**

We do not match the paper's headline number under either protocol, and the gap between our two protocols is itself evidence for the leakage argument (see `14_leakage_comparison.py` for the controlled version of this comparison on the same splitter, same features).