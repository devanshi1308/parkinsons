# Remaining paper sections (draft)

Companion to `docs/paper_draft_methods_results.md`. Every number is taken from the project's own results files or from
`progress_and_future_work.md`. Citations that already appear in the Methods draft are used as-is. Anything marked
**[CITE]** needs a real reference found and verified by the authors; none were invented here. Section numbers assume
Methods = 2 and Results = 3, with Related Work as Section 1.2 inside the Introduction or as its own section as the
target venue prefers.

---

## Abstract

Gait analysis is a promising low-cost route to objective Parkinson's disease (PD) screening, and published
classifiers on the public PhysioNet gait dataset report accuracies above 99%. We show that headline numbers of this kind
are inflated by subject-level leakage: each participant contributes several recordings, and trial-level
cross-validation places some of a person's recordings in both training and test data. Re-implementing the
baseline pipeline of Balaji et al. (2020), we find that trial-level evaluation yields 53-76% four-class accuracy
whereas subject-grouped evaluation yields 38-46%, a gap of 15-30 percentage points that is statistically
significant for every classifier and label scheme tested (20 seeds, Wilcoxon p < 0.001) and is unchanged when
preprocessing is fitted inside each fold. About 63% of trial-level test recordings belong to a subject also seen in
training. The same mechanism appears in an independent voice dataset, where every leaky-split test recording shares
a speaker with the training data. With richer spatiotemporal, centre-of-pressure, spectral and demographic features
and nested cross-validation to remove model-selection bias, honest accuracy is about 82-83% for healthy versus PD and
about 42-44% for four-class severity (majority-class rate 36%). The hardest class, Hoehn and Yahr stage 3, has only
10 subjects and is essentially unrecoverable. SHAP attributions highlight walking speed, centre-of-pressure sway and
stride variability, which are clinically plausible. We release the evaluation protocol and code, and recommend
subject-grouped evaluation as the minimum standard for sensor-based PD classification. Outputs are research screening
estimates, not diagnoses.

---

## 1. Introduction

Parkinson's disease is a progressive neurodegenerative disorder whose severity is conventionally rated with the
Hoehn and Yahr (H&Y) scale and the UPDRS. Both depend on an expert's observation and are therefore subjective, costly
and hard to repeat often. Gait is an attractive objective signal: it changes measurably in PD, can be captured by
inexpensive instrumented insoles or floor sensors, and can be recorded repeatedly. This has produced a substantial
literature applying machine learning to gait recordings, including the work of Balaji et al. (2020) on the PhysioNet
Gait in Parkinson's Disease dataset [Hausdorff; Goldberger et al. 2000], which reports accuracy near 99% for
stage classification. **[CITE: add 2-3 further gait/PD classification papers that report high accuracy on this dataset.]**

Such figures are hard to reconcile with the clinical difficulty of the task, and they invite a methodological
question: how were the models evaluated? The dataset contains several walking trials per person. If recordings are
shuffled into cross-validation folds without regard to who produced them, a classifier can be scored on trials from a
person it has already seen, and may succeed by recognising that individual's idiosyncratic gait rather than anything
associated with disease. This form of leakage is documented in other health-sensing settings [Saeb et al. 2017], but we
are not aware of it having been measured for the standard reporting practice on this dataset.

This paper makes four contributions.

1. **Quantified leakage.** We re-implement the baseline pipeline and evaluate it under trial-level and
   subject-grouped cross-validation with everything else held fixed. Subject-level leakage inflates four-class accuracy
   by 15-30 points, with significance testing, a fold-count sensitivity check, and an audit of how many test trials
   share a subject with training data (Section 3.2). Our re-implementation does not reproduce the original 99% figure
   under either protocol, and we report that plainly.
2. **Generalisation of the effect.** We repeat the comparison on an independent voice dataset, where leakage is even
   more direct, supporting the claim that this is a property of evaluation practice rather than of one sensor type.
3. **An honest benchmark.** Using richer features and nested cross-validation to remove model-selection bias, we report
   de-biased accuracy for three label schemes, including negative results (per-stride curve-shape features, a sliding-window
   LSTM, class-imbalance remedies and detrended fluctuation analysis all failed to improve honest accuracy).
4. **Explainable, calibrated outputs.** We provide SHAP explanations in clinically named features, a compact
   five-feature model, probability calibration, and sensitivity/specificity, with leave-one-cohort-out checks.

We do not claim a deployable diagnostic tool. The aim is to give researchers a corrected protocol and realistic
expectations.

### 1.2 Related work

**Gait-based PD classification.** Machine-learning classifiers on instrumented-gait data, using statistical summaries of
force signals with decision trees, support vector machines, ensembles and naive Bayes, are the approach reproduced
here [Balaji et al. 2020]. Deep-learning and signal-processing variants exist **[CITE]**; we tested a sliding-window LSTM
and found it underperformed tabular models on this sample size.

**Gait variability as a biomarker.** Stride-to-stride variability and long-range correlation of stride intervals have
been studied extensively as markers of PD and its progression, notably in the work associated with the PhysioNet gait
database [Hausdorff; Goldberger et al. 2000]. **[CITE: specific stride-variability and DFA papers.]** We include
stride-time variability and symmetry features for this reason. Detrended fluctuation analysis was introduced by
Peng et al. (1994) **[CITE: full reference]**; our attempt to apply it to the VGRF waveform, rather than to stride
intervals, added no accuracy (Section 3.x).

**Evaluation leakage in health sensing.** Saeb et al. (2017) showed that record-wise cross-validation can substantially
overstate performance when a person contributes multiple records, in the setting of smartphone-based health sensing.
Related concerns about subject-wise versus record-wise splitting have been raised in other biomedical signal tasks
**[CITE]**. Our contribution is to quantify the effect on a specific, widely used PD benchmark and to show it
generalises across modalities.

**Voice-based PD detection.** Acoustic features such as jitter, shimmer, pitch statistics and noise ratios are
standard in PD speech analysis [Sakar et al. 2013] **[CITE: further voice-PD work]**. We use this dataset not to claim a
voice diagnostic but to test whether the leakage mechanism is modality-specific.

---

## 4. Discussion

**Leakage explains most of the apparent performance.** Under subject-grouped evaluation, four-class accuracy falls from
53-76% to 38-46%, and the gap is largest for the highest-capacity models (Bagging and SVM, about 30 points). This is the
pattern one expects if those models partly memorise individuals. The gap is much smaller in the binary task (1-11 points),
which is consistent with a simpler decision boundary offering less room for identity to help. The voice dataset is the
cleanest demonstration, since every leaky-split test recording has a same-speaker recording in training, but its honest
accuracy cannot be distinguished from chance at 40 subjects (permutation p = 0.1485), so it supports the
mechanism claim and nothing more.

**What honest performance looks like.** The de-biased binary estimate is roughly 82-83%, and leave-one-cohort-out
accuracy averages 80.8%, so the binary result holds up on data collected by a different sub-study. Four-class severity
is a different story: about 42-44% accuracy against a majority-class rate of 36%, with macro-F1 near 0.33. A dedicated
four-class push (feature-family ablation plus nested selection of both feature set and classifier) confirmed rather than
broke this ceiling: later feature families improved the better models, but the selected configuration changed from fold to
fold, and DFA, per-stride curve shape, SMOTE, class weighting and an LSTM all failed to help. With only 10 subjects at
H&Y stage 3, the limit is the data, not the modelling.

**Clinical meaning.** At the operating point we calibrated, sensitivity is about 0.90 and specificity about 0.70, so
roughly 30% of healthy walkers would be flagged. That profile suits triage that is followed by clinical review, not
standalone diagnosis. Probability calibration was necessary: raw model probabilities were poorly calibrated, which would
make any patient-facing "chance of PD" misleading.

**Where the signal comes from.** SHAP attributions point to walking speed, centre-of-pressure sway and stride variability
for gait, and to pitch variability, jitter and shimmer for voice, in line with the established literature. Two cautions
apply. First, walking speed is a clinician-timed measurement from the demographics file, not derived from the force signal,
and stride and step lengths are estimates computed from it. Second, the added value of the gait signal over demographics
and speed alone is cohort-dependent: large in two sub-cohorts and near zero in the third. Claims about the signal's
independent contribution should be made per cohort, not globally.

**Limitations.**
- The dataset is small (165 subjects with gait data) and comes from three sub-studies with different class mixes, so
  cross-cohort generalisation is only partly tested.
- Severe PD (24 trials, 10 subjects) is effectively unclassifiable; we report this rather than merging or reweighting it
  away.
- The voice result is not statistically distinguishable from chance and is used only to illustrate leakage.
- Fusion of gait and voice is a mechanics demonstration only. No public dataset has both modalities on the same people,
  so we make no fused-accuracy claim.
- Our DFA was applied to the VGRF waveform rather than stride intervals and so does not test the stride-interval
  hypothesis from the literature.
- A third modality (wrist-worn tremor data, PADS) was planned but not completed; if it is added, the same
  leaky-versus-grouped comparison should be run on it.
- Outputs are research screening estimates, not medical diagnoses.

**Recommendations.** For sensor-based PD classification we recommend: (1) always split by participant; (2) report the
trial-level number alongside the grouped number if comparing with older work; (3) fit preprocessing and model selection
inside the training folds, with nested cross-validation when choosing among models or feature sets; (4) report
macro-averaged metrics and a majority-class baseline, particularly for imbalanced severity classes; and (5) report
calibration and sensitivity/specificity when probabilities are presented to users.

---

## 5. Conclusion

Reported accuracy for gait-based PD classification depends heavily on how the data are split. When recordings from the
same person can appear on both sides of a split, four-class accuracy is overstated by 15-30 points on the widely used
PhysioNet benchmark, and the same mechanism appears in voice data. Under a leakage-free, nested protocol, healthy-versus-PD
discrimination reaches roughly 82-83% with clinically plausible explanatory features, while four-class severity staging
remains limited to about 42-44% by the size of the rarest class. We hope the protocol and the negative results reported
here help set more realistic expectations for the field. Future work includes completing the wrist-sensor modality,
external validation on a further independent gait dataset, and prospective evaluation with the same participants
recorded in more than one modality.
