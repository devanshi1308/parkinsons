# Decision log

Record of project decisions. Add new entries at the bottom with the date and the reason.

## Evaluation protocol
- **2026-09 | Subject-grouped CV is the primary protocol.** `StratifiedGroupKFold`, `subject_id`
  as groups, seed 42 (10 folds; folds capped by the smallest class). Why: trial-level CV lets a
  subject's trials sit in both train and test.
- **2026-09 | Leakage is quantified by `scripts/14_leakage_comparison.py`** (5 seeds, leaky vs
  grouped, global vs in-fold preprocessing, leak audit). Results: `data/processed/leakage_comparison.csv`,
  `data/processed/leakage_audit.csv`, `docs/figures/leakage_gap.png`. The earlier "~67-70% vs ~40%"
  figure had no script behind it and was replaced by these measured numbers.
- **Feature/augmentation/selection steps must fit inside training folds** for any new experiment.
  (Existing `07` preprocessing is label-free and global; script 14 shows in-fold gives the same result.)

## Labels
- **4-class** Healthy / Mild (H&Y 2) / Moderate (2.5) / Severe (3); **3-class** merges Moderate+Severe;
  **binary** Healthy vs PD. Why: Severe has only 10 subjects (24 trials) and is effectively unlearnable.

## Features
- **Correction (2026-09): gait speed is available.** `demographics.xls` has `Speed_01` (m/s, 165 of
  166 subjects), so speed is a subject-level feature and stride/step length are DERIVED ESTIMATES
  (speed x stride time), always labelled as estimates. The VGRF files themselves still have no
  distance data, so measured step/stride length remains impossible. This replaces the earlier
  entry that excluded speed and stride length entirely.
- **UPDRS, UPDRSM, TUAG and H&Y are never used as features** (clinical severity scores = label
  proxies; NaN for controls). Height in `demographics.xls` mixes metres (Ga) and cm (Ju/Si);
  values > 3 are divided by 100.
- **Features v3 (`03_extract_features_v3.py`)** adds spectral, wavelet, regularity, double-support,
  demographics and speed families, tagged in `feature_groups_v3.csv` for ablation. Ablations always
  include a demographics+speed-only baseline to show what the gait signal adds.
- Features v2 adds COP, stride-time CV, symmetry, hysteresis stance/swing, skew/kurtosis/IQR.

## Scope
- **Datasets:** gaitpdb (gait), PADS (wrist tremor/hand movement), UCI Parkinson's Speech #301 (voice).
  Different people in each; never merge rows across them.
- **Dropped modalities:** handwriting/spiral drawing, eyes/pupils. Face deferred (no large public data).
- **Fusion is a demo only.** No public dataset has all modalities on the same people, so fused
  accuracy is not claimed.
- **Alpha-synuclein / aggregation-prediction idea is out of scope** for this paper (possible separate
  project; the tool "Kanya" could not be verified).
- **Paper thesis stays leakage-first.** Voice and wrist results are generalization evidence, SHAP is
  interpretability; neither changes the main claim.

## Open items
- Voice/wrist objectives depend on downloading PADS and UCI speech; PADS group counts unchecked.
- Confirm whether to keep `docs/paper.pdf` in the public repo (baseline paper, possibly copyrighted).
- 165 unique subjects appear in `features_v2.csv` vs 166 documented; investigate the missing one.

## Results log (features v3, models; script 12 and 40, subject-grouped, 3 seeds)
- **2026-09 | Binary:** best 83% accuracy, AUC ~0.90 (Random Forest / Voting / HistGradientBoosting on
  `v3_all`); the baseline four models on v2 reach up to 77% (Bagging). Random Forest on v2 alone is 81%,
  so most of the gain comes from better models (+4 pts); speed adds ~1.5-2 pts; the new spectral,
  wavelet, regularity and double-support families add ~0 (81.0% -> 80.9%).
- **4-class:** best ~51% (HGB 50.9, RF 50.7 on `v3_all`) vs 46-49% before. The 2-4 point gain is small
  relative to fold variance (SD ~0.15), so it is not established.
- **Demographics + speed only** (no gait signal) already give 73% (RF) to 78% (SVM) in binary, so the
  gait-signal contribution must be reported next to this baseline (RF: 73% -> 81% with signal).
- **Selection caveat:** the best combination was picked from 45 candidates on the same CV folds, so
  the best numbers are optimistic. Treat as exploratory; confirm with nested CV or leave-one-cohort-out.
- **SHAP (binary RF, out-of-fold):** top drivers are estimated stride length, walking speed
  (higher lowers risk), double-support share (higher raises risk), centre-of-pressure sway and
  swing time; many other top features are raw sensor skewness/kurtosis that are hard to read
  clinically. By family, per-feature importance is highest for speed, double-support and COP.

## Baseline comparison (script 15, closes Phase 3)
- Paper's Table 8 (kinematic, 4-class) claims DT/SVM 99.4%. Our best match of its described protocol
  (trial-level 10-fold, repeated 10x, NOT subject-grouped) gives at most 71.1% (Bagging, kinematic
  features). Subject-grouped (honest) gives at most 51.9% (SVM, kinematic). We do not reproduce the
  paper's headline number under either protocol.
- The paper does not state whether its folds were grouped by subject, so `trial_level_repeated10` is
  our best-effort match to the paper's description, not a confirmed replica. The large remaining gap
  (71.1% vs 99.4%) likely reflects other undocumented differences (exact feature list, hyperparameters,
  preprocessing), not only grouping. State this honestly in the paper; do not claim to have found the
  single cause of the paper's 99.4%.
- This independently corroborates the leakage argument: the same trial-level-vs-grouped split causes a
  large accuracy drop (71.1% -> 51.9%, kinematic) on a second, independent script (15 vs 14).

## Model-improvement follow-ups (script 16, closes the 2026-10 improvement discussion)
- **De-biased ("nested") estimate -- cite THIS, not model_comparison_v3's best-of-45 table:**
  4-class outer accuracy 42.1% (sd 8.3), macro-F1 31.6%. Binary outer accuracy 79.4% (sd 9.7),
  macro-F1 75.5%. Both are lower than the best-of-45 numbers (51%, 83%), confirming those were
  optimistic from selection bias. Most-picked pair: v3_all/HistGradientBoosting (binary),
  v3_all/Random Forest and v3_signal/Bagging (4-class).
- **Subject-level probability aggregation:** did not help (trial 83.0% vs subject-mean 81.2%
  accuracy; macro-F1 ~flat, 0.797 vs 0.806). Most subjects have only 1-2 trials. Report as a
  null result, not an adopted method.
- **Ordinal (Frank-Hall) vs multiclass, 4-class:** ordinal gave WORSE mean-absolute-stage-error
  (0.657 vs 0.585) and lower accuracy (48.4% vs 50.7%) than plain multiclass RF, despite higher
  macro-F1 (0.414 vs 0.390). Ordinal decomposition did not clearly help here; report as negative,
  do not adopt as the primary 4-class model.
- **Compact feature set:** L1-logistic (C=0.03) selected inside each fold -> 5 features
  (walking speed, left-foot sensor-5 force skewness, left-foot COP front-back sway, left-foot
  sensor-7 median force, left-foot sensor-6 force skewness) recover 78.0% vs the full 202-feature
  82.3% (binary, non-nested). Use this 5-feature list for the clinician-facing SHAP story instead
  of the raw ~200-feature set.
- **XGBoost/LightGBM now installed** (`brew install libomp`). On v3_all: binary 83.8-84.5%,
  4-class 50.0-52.4% -- matches or slightly beats the earlier best scikit-learn models, not a
  breakthrough, and not yet nested-CV-corrected.
- **Net takeaway for the paper:** report the de-biased 4-class/binary numbers as primary;
  mention subject-aggregation and ordinal-decomposition as tried-and-didn't-help; use the
  5-feature compact model for interpretability; note XGBoost/LightGBM as a minor, unconfirmed gain.

## Ju-cohort investigation, reviewed (2026-10) + bug fix + missing-subject explanation
- **Bug found and fixed in `11_investigate_ju_cohort.py`:** it built subject IDs as
  `{Study}{Group}{Subjnum}` using `demo['Group']` ('PD'/'CO'), but real subject_ids use 'Pt'/'Co'
  (e.g. "GaPt03"), so every row became e.g. "GaPD03" and the demographics merge silently matched
  nothing (all-NaN output). Fixed to use `demo['ID']` directly, which is already correctly formatted.
- **Missing 166th subject, explained:** `demographics.xls` row `Juc010` (note lowercase/typo'd
  "c0" instead of "Co") is a demographic entry with NO corresponding gait trial file anywhere in
  `data/raw/` (no `JuCo10_*.txt` exists). This is a data-entry/distribution gap in PhysioNet's own
  file, not a bug in our pipeline. 165 of 166 demographic subjects have gait data; this one does not.
- **Ju's low pooled accuracy is substantially explained by class mix, not equipment/noise:**
  class balance differs sharply by cohort (Healthy/Mild/Moderate/Severe %): Ga 33.6/36.3/17.7/12.4,
  Ju 19.4/31.0/41.9/7.8, Si 45.3/43.8/10.9/0.0. Ju is loaded with the hardest class (Moderate,
  41.9% of its trials) and has the fewest easy Healthy trials — a harder class mix than Ga or Si,
  independent of any Ju-specific signal quality issue.
- **Trial duration is much more variable in Ju** (mean 87.5s, sd 32.5, range 36.3-259.6s) than Ga
  (116.5s, sd 6.6) or Si (fixed at 117.2s). Ga/Si trials are essentially a fixed protocol length;
  Ju is not. This is a plausible secondary contributor to noisier per-trial features in Ju (more or
  less data to average per trial) and is worth a sentence in the paper's limitations.
- **Demographics differ by cohort too:** Si is youngest (59.9 vs Ga 71.3, Ju 66.0) — a potential
  age confound on top of the class-mix difference; height units already fixed per the 2026-09
  correction (Ga metres, Ju/Si cm).
- **No evidence of a stance-detection / equipment artefact:** raw sensor mean/SD and the kinematic
  NaN rate are flat across cohorts (0% missing everywhere), ruling out a feature-extraction bug as
  the explanation.
- **Net:** report Ju's weak pooled score as "harder class mix + protocol-duration variability",
  not as an unexplained anomaly; the leave-one-cohort-out result (44.2% when Ju is held out
  entirely, vs 31.8% pooled) already supports this — Ju subjects are not simply harder to classify
  in isolation, they are harder mainly when lumped into one pooled model with a skewed class mix.

## SHAP follow-ups: compact feature set + 4-class (script 41)
- **Compact 5-feature binary SHAP:** out-of-fold accuracy 79.4% (close to the nested de-biased
  79.4% from script 16 -- consistent). Ranked drivers: left-foot sensor-5 force skewness >
  walking speed > left-foot COP front-back sway > left-foot sensor-7 median force > left-foot
  sensor-6 force skewness. This is the feature list and chart to use for the clinician-facing
  explanation (`docs/figures/shap_compact_top_features.png`, `shap_compact_beeswarm.png`).
- **4-class SHAP (out-of-fold multiclass RF, v3_all, 50.7% accuracy):** per-stage top features
  are dominated by the derived stride/step-length estimates (speed x stride time) in every stage,
  with sensor-5 force skewness, COP sway, and stance time also appearing. CAVEAT: stride/step
  length, stride-length/height, and speed are all derived from the same underlying measurement
  (speed x stride time) and are highly correlated, so SHAP is likely splitting/sharing importance
  across near-duplicate features here rather than revealing 4 independent signals -- read the
  4-class ranking as "speed- and stride-length-type information matters most", not as evidence
  for each individual derived column. Figures: `shap_4class_top_features.png` (per stage),
  `shap_4class_global_importance.png`.

## Nested CV including XGBoost/LightGBM as candidates (script 17) -- FINAL numbers to cite
- **4-class:** outer accuracy 44.4% (sd 10.8), outer macro-F1 33.9% (sd 9.4). Barely above the
  sklearn-only nested result (42.1%/31.6% from script 16) -- boosting adds ~2 points, within
  noise. No single (feature_set, model) dominates fold selection (LightGBM, XGBoost, Random
  Forest, Bagging, HistGradientBoosting and even Logistic Regression each won at least one outer
  fold) -- a sign the "best model" is unstable at this sample size, not a real winner.
- **Binary:** outer accuracy 81.7% (sd 10.6), outer macro-F1 78.8% (sd 11.0) -- a real but modest
  improvement over the sklearn-only nested 79.4%/75.5%. LightGBM and XGBoost on v3_all won 8 of
  10 outer folds, so boosting's binary gain looks more consistent than the 4-class one.
- **These are now the final, de-biased numbers to cite in the paper** (supersede both
  model_comparison_v3.csv's best-of-45 table AND script 16's sklearn-only nested numbers):
  4-class 44.4% (sd 10.8), binary 81.7% (sd 10.6).
- Fold-level sd is large (~11 points) on both schemes -- with 165 subjects, any point estimate
  needs that uncertainty reported alongside it, not just the mean.

## Voice (UCI #301) downloaded and analyzed (2026-10) -- second leakage demonstration done
- **Downloaded** via `curl` + `unar` (the archive is a RAR inside a ZIP) into `data/raw_voice/`.
  40 train subjects (20 healthy, 20 PD) x 26 recordings = 1040 rows; 28 test subjects x 6
  recordings = 168 rows, ALL labeled PD (no healthy controls, no UPDRS column) -- confirmed by
  inspecting class values directly, matches prd.md. Feature name order taken from the dataset's
  own UCI page (5 Jitter, 6 Shimmer, AC/NTH/HTN, 5 pitch, 4 periodicity, 3 voice-quality).
  Loader: `30_voice_load.py` -> `data/processed/voice_features.csv`.
- **Leakage comparison** (`31_voice_leakage_comparison.py`, 5 seeds, train partition only, same 4
  classifiers as gait): recording-level (leaky) accuracy 58.4-69.2% vs subject-grouped 52.2-55.7%
  -- gap 3.8-15.2 points (largest for Bagging). On average **100% of leaky-fold test recordings**
  belong to a subject also in that fold's training set (vs ~63% for gait) -- because every subject
  contributes 26 recordings spread across the whole dataset with only 40 subjects total, almost
  guaranteeing subject overlap under any random recording-level split. This is a STRONGER and
  cleaner demonstration of the leakage mechanism than gait, even though the absolute point-gap is
  smaller (lower baseline accuracy ceiling here). THIS IS THE KEY GENERALIZATION RESULT for the
  paper's thesis: the leakage effect is not a gait-specific artifact.
- **SHAP** (`32_voice_shap.py`, out-of-fold RF, grouped CV, 55.6% out-of-fold accuracy, consistent
  with the leakage comparison's grouped numbers): top drivers are pitch variability (SD), jitter
  (PPQ5), maximum pitch, fraction of unvoiced frames, shimmer (APQ11) -- all established dysphonia
  measures in the PD-voice literature (Little et al. 2009 and similar), so the ranking is
  clinically plausible.
- **Caveat to keep in the paper:** 40 subjects is small; report voice results as indicative /
  generalization evidence, not a competitive voice-PD benchmark result.

## Statistical robustness of the leakage finding (script 18) -- IMPORTANT, includes a caveat
- **A. Paired significance (20 seeds, Wilcoxon + paired Cohen's d):** the leaky-vs-grouped gap is
  significant (p<0.001, most p<0.0001) for EVERY classifier/scheme in BOTH gait and voice, with
  large effect sizes (Cohen's d 1.35-12.55). The 20-seed means refine the earlier 5-seed figures
  slightly (e.g. gait 4-class Bagging gap 30.0pts here vs ~29.8pts before -- consistent). Use the
  20-seed numbers (`data/processed/leakage_significance.csv`) as the final cited gap sizes, since
  they have more statistical power than the original 5-seed run.
- **B. Permutation (chance-level) test, grouped protocol, Bagging, 100 permutations -- MIXED
  RESULT, report honestly:**
  - Gait binary: observed 77.1% vs null mean 51.2% (sd 5.0), **p=0.0099** -- clearly above chance.
  - Gait 4-class: observed 48.8% vs null mean 38.3% (sd 4.6), **p=0.0198** -- above chance.
  - **Voice binary: observed 54.5% vs null mean 48.9% (sd 5.7), p=0.1485 -- NOT significant at
    p<0.05.** The honest (subject-grouped) voice-Bagging accuracy is NOT statistically
    distinguishable from chance with this permutation test. This does not invalidate the voice
    LEAKAGE result (section A's gap is still highly significant, p=0.0001, i.e. the leaky number
    is reliably inflated relative to the grouped number) -- but it means we should NOT claim the
    grouped voice model has learned a strong, statistically confident PD-vs-healthy signal. State
    this plainly in the paper: voice's generalization value is in replicating the LEAKAGE
    MECHANISM (section A, and the 100%-subject-overlap leak-audit finding), not in claiming a
    validated voice diagnostic result. This is consistent with voice's known limitation (40
    subjects, small).
- **C. Capacity vs leakage-gap table:** gap size follows GaussianNB < {SVM, Decision Tree} <
  Bagging in every scheme/modality except gait binary (where DT < SVM < Bagging, SVM highest) --
  overall a consistent pattern that higher-capacity/memorization-prone models (Bagging especially)
  show larger leakage gaps than Naive Bayes. Supports a mechanistic explanation (models that can
  fit subject-specific patterns benefit most from seeing a subject's other trials in training),
  not just an unexplained statistical artifact.
- **Net effect on the paper's claims:** the leakage INFLATION finding is now statistically
  bulletproof for both modalities (part A). The claim that grouped/honest models "work" should be
  stated with the permutation evidence attached per modality: gait grouped accuracy is
  significantly above chance; voice grouped accuracy, while real and consistent with the acoustic
  literature (SHAP directions match known dysphonia markers), is not statistically distinguishable
  from chance at n=40 subjects with this test and classifier -- report as a genuine limitation.
