# Methods and Results (draft)

Working draft. Every number here traces to a script and a CSV in `data/processed/`; the script
name is given in brackets so a reviewer (or future us) can reproduce it. Replace bracketed
script-name provenance notes with proper citations/footnotes at final formatting. Still needed
before this is submission-ready: Introduction, Related Work (literature review material already
drafted in `docs/project_overview.pdf` section 3, needs trimming into a Related Work section),
Discussion, Limitations (partly drafted inline below), Conclusion, Abstract.

---

## 2. Methods

### 2.1 Datasets

**Gait.** We use the PhysioNet "Gait in Parkinson's Disease" dataset (gaitpdb v1.0.0)
[Hausdorff; Goldberger et al. 2000], comprising vertical ground reaction force (VGRF)
recordings from 16 foot-worn sensors (8 per foot) plus 2 total-force channels, sampled at
100 Hz, from three independently collected sub-cohorts (Ga, Ju, Si). The accompanying
demographics file documents 166 subjects (93 idiopathic PD patients, 73 healthy controls);
one subject (demographic record "Juc010", a Ju-cohort healthy control) has no corresponding
gait recording in the public distribution, leaving 165 subjects and 306 trials with usable
VGRF data [`04_merge_labels.py`, `03_extract_features_v2.py`]. PD severity is recorded on
the Hoehn & Yahr (H&Y) scale (2, 2.5, 3), which we map to four classes: Healthy (n=73),
Mild/H&Y 2 (n=55), Moderate/H&Y 2.5 (n=28), Severe/H&Y 3 (n=10 subjects, 24 trials). A
3-class scheme merges Moderate and Severe (the two smallest, most-confused classes), and a
binary scheme contrasts Healthy against all PD subjects, following the framing most prior
work on this dataset reports [`07_train_multi_scheme.py`].

**Voice.** To test whether the leakage effect we identify in gait (Section 3.2) is
specific to that modality, we additionally analyze the UCI "Parkinson's Speech with
Multiple Types of Sound Recordings" dataset (ID #301) [Sakar et al. 2013]. Its training
partition comprises 40 subjects (20 PD, 20 healthy), each contributing 26 voice recordings
(sustained vowels, numbers, words, short sentences) summarized by 26 standard acoustic
features (5 jitter measures, 6 shimmer measures, autocorrelation/noise-to-harmonics/
harmonics-to-noise ratios, 5 pitch statistics, 4 periodicity measures, 3 voicing-quality
measures); no raw audio is distributed. A separate test partition (28 subjects x 6
recordings) contains PD patients only and no UPDRS scores, and is therefore unusable for
accuracy estimation; we use it for completeness checks only and report all voice results on
the 40-subject, class-balanced training partition [`30_voice_load.py`].

Both datasets involve multiple recordings per subject and no subject appears in both
datasets. We never merge rows across the two datasets; the voice analysis is an independent
replication of the evaluation-methodology question, not a multimodal fusion of the same
people.

### 2.2 Feature extraction (gait)

Following the original gait-classification literature on this dataset [Balaji et al. 2020],
we first extract purely statistical per-sensor features (mean, median, SD; 16 sensors x 3
statistics) [`03_extract_features.py`], then a richer spatiotemporal/kinematic feature set
[`03_extract_features_v2.py`] comprising: (i) stance, swing and stride time per foot via
hysteresis thresholding of the total-force signal (two thresholds, to reduce spurious
stance/swing flips from sensor noise near a single cutoff); (ii) stride-time coefficient of
variation, an established PD gait-variability biomarker; (iii) a left-right symmetry index
for stance and stride time; (iv) centre-of-pressure (COP) trajectory features (sway in the
medial-lateral and anterior-posterior directions, and path length) computed from the
sensors' documented X-Y positions; and (v) extended statistical moments (skewness,
kurtosis, min, max, range, IQR) beyond the mean/median/SD the baseline paper used.

A third feature set [`03_extract_features_v3.py`] adds: spectral features (dominant
frequency, relative power in three gait-relevant bands, spectral entropy and centroid, via
Welch's method) and wavelet features (db4, level-5 relative sub-band energy) computed per
foot's total-force signal; stride-to-stride regularity (height of the first and second
autocorrelation peaks of the combined bilateral force signal) and stride-time sample
entropy; double-support time (the fraction of the gait cycle both feet are loaded) and
per-foot stance-time percentage; and five demographic/clinical features (age, sex, height,
weight, BMI) together with walking speed. Walking speed is not derived from the VGRF signal
-- it is a clinician-timed measurement (`Speed_01`, m/s) present in the dataset's
demographics file for 165 of 166 subjects, which we had initially (incorrectly) assumed was
unavailable; stride and step length are then computed as speed x stride time and reported
as **derived estimates**, not measured distances (the VGRF files contain no spatial/distance
information). UPDRS, UPDRS-motor, Timed-Up-and-Go and H&Y itself are deliberately excluded
as features, since they are clinical severity scores that would leak the label.

### 2.3 Evaluation protocol

**The leakage question.** Each subject in both datasets contributes multiple
trials/recordings. A cross-validation split performed at the trial/recording level can place
some of a subject's trials in the training fold and others in the test fold of the same
split. A classifier can then partly succeed by recognizing an individual's idiosyncratic
gait or voice signature, rather than by learning signal associated with PD status -- a form
of data leakage that is well documented in other health-sensing contexts [Saeb et al. 2017]
but, to our knowledge, not previously quantified for this dataset's standard reporting
practice. We compare two protocols that are otherwise identical (same features, same model
hyperparameters, same seeds):

- *Trial/recording-level ("leaky")*: `StratifiedKFold`, ignoring subject identity.
- *Subject-grouped ("honest")*: `StratifiedGroupKFold` by subject ID, so no subject's data
  ever appears in both the training and test fold of a given split.

Both use 10 folds (capped by the smallest class's size where needed), repeated over multiple
random seeds, with four classifier families chosen to mirror the baseline literature:
Decision Tree, linear-kernel SVM, a Bagging ensemble of 30 decision trees, and Gaussian Naive
Bayes [`14_leakage_comparison.py`, `31_voice_leakage_comparison.py`]. Preprocessing
(median imputation, a variance filter, and a |r|>0.95 correlation filter) is applied
identically under both protocols; a sensitivity check refitting these steps inside each
training fold rather than once on the whole dataset gives materially the same result
(Section 3.2), confirming the gap is attributable to label/subject leakage rather than
preprocessing information leakage.

**Statistical testing.** For each classifier and label scheme, we run 20 seeds and test
whether the leaky-protocol accuracy exceeds the grouped-protocol accuracy using a paired
Wilcoxon signed-rank test (paired by seed), report the paired effect size (Cohen's d on the
seed-wise accuracy differences), and report the gap as both raw percentage points and
relative inflation (leaky/grouped - 1). We additionally run a permutation (chance-level)
test on the honest protocol: subject-level label shuffling (each subject's label is
reassigned, but all of that subject's trials/recordings keep the shuffled label together,
preserving class balance and the one-subject-many-trials structure of the real data) over
100 permutations, with the Bagging classifier, yielding an empirical p-value for whether the
honest-protocol accuracy exceeds chance [`18_statistical_robustness.py`]. A fold-count
sensitivity check (5-fold vs 10-fold, 10 seeds) confirms the leakage gap is not an artifact
of the chosen fold count [`20_fold_sensitivity.py`].

**Model selection bias.** An initial feature/model ablation selected the best-performing
(feature set, classifier) combination by evaluating all combinations on the same
cross-validation folds used to report accuracy -- a form of selection bias that inflates the
reported number. We correct this with a nested cross-validation procedure: for each outer
subject-grouped fold, the winning (feature set, classifier) combination is chosen using only
an inner cross-validation over that fold's training data, then evaluated once on the
untouched outer test fold [`16_model_improvements.py`, `17_nested_with_boosting.py`]. We
report this de-biased estimate, not the best-of-ablation table, as our primary result.

### 2.4 Models

Beyond the four baseline-matching classifiers (Section 2.3), we evaluate Random Forest,
Extra Trees, HistGradientBoosting, L2-regularized Logistic Regression, a soft-voting
ensemble of the three strongest of these, and (after resolving a missing system dependency,
`libomp`) XGBoost and LightGBM [`12_train_models_v3.py`]. All models are evaluated under
subject-grouped cross-validation only, inside the nested-selection procedure described above.

### 2.5 Explainability

We compute SHAP (TreeExplainer) values for a Random Forest classifier, out-of-fold under
subject-grouped cross-validation, so every explained trial/recording comes from a model that
never saw that subject during training [`40_shap_report.py`, `32_voice_shap.py`,
`41_shap_4class_and_compact.py`]. Gait features are given clinically interpretable names
(e.g. "stride-time variability", "left-right asymmetry", "stance ratio") rather than raw
sensor-channel identifiers where possible. We additionally fit an L1-regularized logistic
regression inside each cross-validation fold to identify a small, stable, clinically-named
feature subset (features selected with non-zero coefficient in at least half of folds)
[`16_model_improvements.py` section 4], to give a more interpretable explanation than the
full ~200-feature model.

---

## 3. Results

### 3.1 Baseline comparison

The original gait-classification study on this dataset [Balaji et al. 2020] reports up to
99.4% accuracy (Decision Tree, kinematic features, 4-class H&Y staging), evaluated with
10-fold cross-validation repeated 10 times; the manuscript does not state whether folds were
grouped by subject. Reproducing its described protocol as closely as possible (same feature
families -- purely statistical vs. kinematic -- same four classifiers, same repeated 10-fold
design, but at the trial level rather than confirmed-subject-grouped, since this is the most
literal reading of "10-fold CV" without an explicit subject-grouping statement) yields at
most 71.1% accuracy (Bagging, kinematic features, mean of 10 repeats); under our honest,
subject-grouped protocol, the same features and classifiers yield at most 51.9% (SVM,
kinematic features) [`15_baseline_comparison.py`, Table in `docs/baseline_comparison.md`].
We do not reproduce the published 99.4% under either protocol, and we are therefore careful
not to attribute the full 47.5-point gap (99.4% vs. our honest 51.9%) to subject leakage
alone -- undocumented differences in the exact feature set, classifier hyperparameters, or
preprocessing may also contribute. What this comparison does establish independently is a
second demonstration of the leakage effect itself: on identical features and classifiers,
switching only the cross-validation protocol (trial-level to subject-grouped) drops
kinematic-feature accuracy from 71.1% to 51.9%.

### 3.2 Quantifying subject-level leakage

**Gait.** Across four classifiers and three label schemes (20 seeds each), trial-level
accuracy exceeds subject-grouped accuracy by 2.2 to 30.6 percentage points (Table 1). The
gap is statistically significant for every classifier/scheme combination (paired Wilcoxon
signed-rank test, all p<0.001, most p<0.0001) with large paired effect sizes (Cohen's d
1.35-12.55) [`18_statistical_robustness.py`]. On average across folds, 63% of trial-level
test-fold trials belong to a subject also present in that fold's training data
[`14_leakage_comparison.py`]. The gap is largest for 4-class and 3-class staging
(up to ~30 points for Bagging/SVM) and smallest for the binary Healthy-vs-PD framing (1.4 to
10.7 points), consistent with fine-grained staging offering more opportunity for a model to
substitute subject recognition for disease-stage recognition.

**Voice.** The same comparison on the independent voice dataset shows a gap of 3.8 to 15.2
points, significant for every classifier (all p<0.001) with large effect sizes (Cohen's d
2.48-7.75) [`31_voice_leakage_comparison.py`, `18_statistical_robustness.py`]. Notably, on
average **100%** of trial-level test-fold recordings in the voice dataset belong to a
subject also present in that fold's training data -- a stronger leakage mechanism than
gait's 63%, because each of only 40 subjects contributes 26 recordings spread across the
dataset, making subject overlap under random recording-level splitting nearly unavoidable.
This is, to our knowledge, the first demonstration that the trial/recording-level leakage
problem identified for gait classification is not specific to that modality or dataset.

**Robustness checks.** The gap is materially unchanged whether preprocessing is fit once
globally or inside each training fold (ruling out a preprocessing-leakage explanation)
[`14_leakage_comparison.py`], and is stable whether cross-validation uses 5 or 10 folds
(differences of 1-3 points across fold counts for every classifier/scheme tested)
[`20_fold_sensitivity.py`].

**Does the gap track model capacity?** Ranking classifiers by their capacity to fit
subject-specific decision boundaries (Naive Bayes < linear SVM /single Decision Tree <
Bagging of trees), the leakage gap follows this order in 7 of 8 scheme/modality
combinations we tested (the exception being gait binary, where SVM slightly exceeds
Bagging) [`18_statistical_robustness.py` section C]. This supports a mechanistic
explanation -- higher-capacity models more readily memorize a subject's individual signature
when given access to that subject's other trials -- rather than leakage being an unexplained
statistical artifact.

**Is the honest (subject-grouped) accuracy itself above chance?** A subject-level label-
permutation test (100 permutations, Bagging classifier) shows gait's honest accuracy is
significantly above chance for both binary (77.1% observed vs. 51.2% null mean, p=0.0099)
and 4-class (48.8% vs. 38.3% null mean, p=0.0198) staging. For voice, however, the honest
accuracy is **not** statistically distinguishable from chance with this test (54.5% observed
vs. 48.9% null mean, p=0.1485) [`18_statistical_robustness.py` section B]. We report this
plainly as a limitation (Section 4): the voice replication demonstrates that the leakage
*mechanism* generalizes beyond gait, but at only 40 subjects we cannot claim the
voice model has learned a statistically confident PD-vs-healthy signal, even though its
SHAP-identified top features (Section 3.4) are consistent with established dysphonia
markers in the PD-voice literature.

**Table 1. Leakage gap by classifier and scheme (20-seed means; gait n=165 subjects/306
trials, voice n=40 subjects/1040 recordings).**

| Modality | Scheme | Classifier | Leaky | Grouped | Gap (pts) | Wilcoxon p | Cohen's d |
|---|---|---|---|---|---|---|---|
| Gait | 4-class | Bagging | 76.1% | 46.1% | +30.0 | <0.0001 | 9.94 |
| Gait | 4-class | SVM | 72.0% | 42.4% | +29.6 | <0.0001 | 12.55 |
| Gait | 4-class | Decision Tree | 64.9% | 40.5% | +24.4 | <0.0001 | 5.82 |
| Gait | 4-class | Naive Bayes | 53.0% | 39.0% | +14.0 | <0.0001 | 6.70 |
| Gait | Binary | SVM | 84.0% | 73.9% | +10.1 | <0.0001 | 3.41 |
| Gait | Binary | Bagging | 86.6% | 78.2% | +8.3 | <0.0001 | 3.47 |
| Gait | Binary | Decision Tree | 79.7% | 71.7% | +8.0 | <0.0001 | 2.01 |
| Gait | Binary | Naive Bayes | 74.7% | 72.6% | +2.2 | 0.0001 | 1.35 |
| Voice | Binary | Bagging | 69.4% | 54.8% | +14.6 | <0.0001 | 7.75 |
| Voice | Binary | Decision Tree | 62.8% | 53.1% | +9.7 | 0.0001 | 4.92 |
| Voice | Binary | SVM | 64.5% | 55.2% | +9.3 | 0.0001 | 3.54 |
| Voice | Binary | Naive Bayes | 58.6% | 54.2% | +4.4 | <0.0001 | 2.48 |

(3-class gait omitted from the table for space; see `data/processed/leakage_significance.csv`
for the complete results, which also include 3-class.)

### 3.3 Corrected model performance

Under the de-biased, nested-selection protocol (Section 2.3), honest accuracy is **44.4%
(SD 10.8) for 4-class H&Y staging** and **81.7% (SD 10.6) for binary Healthy-vs-PD
classification**, averaged over 10 outer folds with XGBoost/LightGBM included as candidates
[`17_nested_with_boosting.py`]. These figures are lower than an earlier, uncorrected
best-of-ablation table (51% and 83% respectively), illustrating the same model-selection-bias
principle as Section 3.2's leakage finding, applied to hyperparameter/architecture choice
rather than cross-validation protocol. No single (feature set, classifier) combination
dominated the inner-fold selection for 4-class staging -- Random Forest, Bagging,
HistGradientBoosting, LightGBM, XGBoost, and even plain Logistic Regression each won at
least one outer fold -- indicating that "the best model" is not a stable concept at this
sample size (165 subjects); for binary classification, LightGBM/XGBoost on the full v3
feature set won 8 of 10 outer folds, a more consistent (though still modest) advantage.

Ablating feature families shows most of the honest-protocol accuracy gain comes from model
choice rather than new features: Random Forest on the original (v2) feature set already
reaches 81% binary accuracy (non-nested); adding the new spectral, wavelet, regularity and
double-support features (Section 2.2) adds approximately nothing (80.9%); adding walking
speed and derived stride/step-length estimates adds 1.5-2 points
[`12_train_models_v3.py`, `data/processed/model_comparison_v3.csv`]. A demographics-and-speed-
only baseline (no gait-signal features at all) already reaches 73-78% binary accuracy,
underscoring that any claimed gait-signal contribution should be reported relative to this
baseline, not to zero.

Subject-level aggregation of trial probabilities (averaging each subject's out-of-fold
probabilities before thresholding, rather than evaluating per trial) did not improve
accuracy (83.0% trial-level vs. 81.2% subject-aggregated; macro-F1 approximately unchanged),
because most subjects contribute only 1-2 trials. An ordinal (Frank-Hall) decomposition of
the 4-class H&Y staging problem, which penalizes "off by one stage" predictions less than
"off by two", achieved lower raw accuracy and a *worse* mean absolute stage error than plain
multiclass classification (0.657 vs. 0.585), so we do not adopt it as the primary 4-class
model [`16_model_improvements.py` sections 2-3].

### 3.4 Explainability

Out-of-fold SHAP analysis of the binary Random Forest (v3 feature set, 82.3% non-nested
accuracy) ranks walking speed, left-foot force-distribution shape (skewness/kurtosis at two
specific sensor locations), centre-of-pressure anterior-posterior sway, and double-support
time among the top contributors to PD-risk predictions, directions consistent with known PD
gait signatures (reduced speed, altered loading, increased double support). An L1-selected
compact feature set of five features (walking speed, left-foot sensor-5 force skewness,
left-foot COP anterior-posterior sway, left-foot sensor-7 median force, left-foot sensor-6
force skewness) recovers 78.0% accuracy, a modest reduction from the full ~200-feature
model's 82.3%, with a substantially more interpretable, clinician-presentable feature list
[`16_model_improvements.py` section 4, `41_shap_4class_and_compact.py`]. For the 4-class
model, top per-stage features are dominated by the derived stride/step-length estimates in
every H&Y stage; because walking speed, stride length, step length, and stride-length/height
are all derived from the same underlying measurement and are highly correlated, we interpret
this as "speed/stride-length-type information matters most" rather than as four independent
findings.

For voice, out-of-fold SHAP (55.6% accuracy, consistent with Section 3.2's grouped-protocol
result) ranks pitch variability (SD), jitter (PPQ5), maximum pitch, fraction of unvoiced
frames, and shimmer (APQ11) as the top contributors -- established dysphonia measures in the
PD-voice literature, lending the explanation face validity despite the modest, not
statistically confident (Section 3.2) classification accuracy at this sample size.

### 3.5 Per-cohort analysis

Gait data were collected across three independently recruited sub-cohorts (Ga, Ju, Si). A
pooled model evaluated per cohort shows markedly uneven performance: 46.9% accuracy (Ga),
31.8% (Ju), 62.5% (Si) [`10_per_cohort_analysis.py`]. Investigating this gap
[`11_investigate_ju_cohort.py`] rules out a data-quality or equipment explanation (raw
sensor force statistics and the kinematic-feature missingness rate are flat across cohorts)
and instead finds two substantive contributors: (i) class balance differs sharply by cohort
-- Ju's trials are 41.9% Moderate-stage (the hardest class) versus 17.7% (Ga) and 10.9%
(Si), with correspondingly fewer easy Healthy trials (19.4% vs. 33.6%/45.3%); and (ii) trial
duration is far more variable in Ju (mean 87.5s, SD 32.5, range 36.3-259.6s) than in Ga
(116.5s, SD 6.6) or Si (fixed at 117.2s), suggesting a less standardized recording protocol.
Leave-one-cohort-out evaluation (44.2% accuracy when Ju is entirely held out, versus 31.8%
when pooled) supports the class-mix explanation over a "Ju subjects are inherently harder to
classify" explanation.

---

## 4. Limitations (draft notes for the full Discussion/Limitations section)

- Gait: 165 usable subjects; the Severe H&Y stage (10 subjects, 24 trials) is close to
  unlearnable under any protocol and inflates apparent 4-class accuracy's informativeness if
  reported without macro-F1 or per-class metrics alongside it.
- Voice: 40 subjects; honest-protocol accuracy, while real (Section 3.2's leakage gap is
  highly significant), is not statistically distinguishable from chance by our permutation
  test. Voice results should be read as evidence the *leakage mechanism* generalizes, not as
  a validated voice-based PD classifier.
- Step/stride length are estimated from clinician-timed walking speed and measured stride
  time, not from direct spatial/distance sensing; they should not be treated as equivalent
  to instrumented gait-lab measurements.
- The baseline-paper comparison (Section 3.1) cannot isolate leakage as the sole cause of the
  47.5-point gap to the published 99.4%, since several methodological details (exact feature
  list, hyperparameters) are not fully specified in the original publication.
- Wrist-sensor (PADS) replication was planned as a third modality but [status as of this
  draft: data download/processing pending -- update before finalizing].
- Multimodal fusion (combining gait, voice, and wrist predictions into one risk score) is
  not attempted in the current results, since no public dataset provides all modalities on
  the same subjects; any future fusion demonstration should be presented as exploratory, not
  as a validated combined-accuracy claim.
