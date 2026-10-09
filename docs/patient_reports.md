# Example patient reports (out-of-fold, real trials from the dataset)

Each report comes from a model that never saw that person. Risk is the Random Forest's out-of-fold probability of PD for that single walking trial. **Research screening estimate, not a diagnosis.**

## JuCo17 - recorded stage: Healthy

- Estimated chance of Parkinson's disease: **6%** (model baseline 70%)
- Raising the estimate: none
- Lowering the estimate: Walking speed (1.54 vs healthy median 1.25); Stride length (estimated from speed) (1.65 vs healthy median 1.36)

## GaPt20 - recorded stage: Mild

- Estimated chance of Parkinson's disease: **98%** (model baseline 70%)
- Raising the estimate: Stride length (estimated from speed) (0.835 vs healthy median 1.36); Sensor 5 (left foot) force skewness (0.462 vs healthy median 1.04); Double-support share of walking time (0.372 vs healthy median 0.3)
- Lowering the estimate: none

## JuPt13 - recorded stage: Severe

- Estimated chance of Parkinson's disease: **99%** (model baseline 70%)
- Raising the estimate: Sensor 5 (left foot) force skewness (0.398 vs healthy median 1.04); Stride length (estimated from speed) (0.726 vs healthy median 1.36); Sensor 5 (left foot) force kurtosis (-1.49 vs healthy median -0.0752)
- Lowering the estimate: none
