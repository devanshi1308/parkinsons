# PRD — Parkinson's Disease Gait Classification (Research Paper Project)

## Problem
Clinical PD severity rating (H&Y scale, UPDRS) is subjective, expert-dependent, and inefficient.
Gait abnormality is an early, quantifiable biomarker of PD. This project builds a supervised
ML pipeline that classifies gait signals into PD stages using vertical ground reaction force
(VGRF) data, reproducing Balaji et al. (2020, Applied Soft Computing) as a baseline — and
identifies a methodological flaw in how that baseline (and likely similar prior work) was
evaluated.

## Goal
Produce a publishable research paper with:
1. A reproduced baseline (statistical + kinematic features, DT/SVM/EC/Bayes classifiers)
2. A demonstrated, quantified novel contribution (see below — LOCKED IN)
3. Honest, leakage-free results, reported alongside the inflated baseline numbers for contrast

## Objective (LOCKED IN)
**Working title:** Leakage-free, explainable machine-learning screening of Parkinson's disease
from gait, with cross-modality validation on wrist-sensor and voice data.

**Main objective:** show that reported accuracy for PD gait classification is inflated when the
same person's trials appear in both train and test data, and provide a corrected, honest
evaluation protocol with interpretable models.

**Specific objectives**
1. Re-implement the baseline pipeline (Balaji et al. 2020) on PhysioNet gaitpdb and evaluate it
   under both trial-level and subject-grouped CV. Our trial-level numbers do not match the
   baseline's ~99% headline; report that plainly.
2. Quantify the leakage (`scripts/14_leakage_comparison.py`, 5 seeds, 4-class): trial-level
   65-76% vs subject-grouped 38-46% accuracy, a 15-30 point inflation depending on classifier
   (~30 points for Bagging and SVM). Binary inflation is smaller (1-11 points).
3. Improve features (COP, variability, symmetry, frequency/regularity, demographics) and
   measure the gain under honest evaluation.
4. Show the leakage effect generalizes: repeat leaky-vs-grouped on voice (UCI speech) and
   wrist-sensor (PADS) data.
5. Make predictions explainable with SHAP, in clinically named features.
6. Demo a late-fusion pipeline combining the three models into one risk score with an
   explanation (exploratory, not validated).

**Who this helps**
- ML/biomedical researchers: warning + fix for inflated PD accuracy numbers (main contribution).
- Neurologists/clinicians: realistic performance expectations and feature-level explanations
  they can check against clinical knowledge.
- Patients/screening programs (indirectly): honest models are the ones that could support
  low-cost, objective, repeatable screening where specialists are scarce.

**One-sentence pitch:** Published gait-based PD classifiers overstate their accuracy because of
subject leakage; we measure the inflation, correct it, and show honest models can still be
informative and explainable.

## Novel contribution (LOCKED IN)
**Subject-level data leakage in prior gait-PD classification work, and a corrected,
leakage-free evaluation with richer spatiotemporal features.**

Specifically:
1. **Diagnosis**: Showed that trial-level cross-validation (used in the baseline paper and
   likely similar prior work) allows the same subject's trials to appear in both train and
   test folds, inflating reported accuracy. In our pipeline (4-class, 5 seeds), trial-level CV
   gave 65-76% accuracy across classifiers; switching to subject-grouped CV
   (StratifiedGroupKFold, no subject ever split across train/test) gave 38-46%. The gap is
   15-30 points depending on classifier (~30 for Bagging and SVM), attributable to leakage
   alone, not model quality. On average ~63% of trial-level test trials belong to a subject
   also present in training. Gaps are smaller in the 3-class (9-32 points) and binary (1-11
   points) schemes. Results are unchanged when preprocessing is fit inside each fold.
2. **Improved feature engineering**: Added center-of-pressure (COP) trajectory features
   (computed from the paper's own sensor X-Y geometry — spatial information the original
   VGRF-only approach didn't use), stride-time coefficient of variation (established PD
   variability biomarker), left-right symmetry index, hysteresis-based stance/swing
   detection, and extended statistical moments (skew/kurtosis/IQR). This improved leakage-free
   accuracy modestly (4-class: ~40% → ~49%).
3. **Multi-scheme evaluation**: Reported results under 4-class (Healthy/Mild/Moderate/Severe),
   3-class (merging Moderate+Severe due to small sample size and high confusion), and binary
   (Healthy/PD) framings — all under the same rigorous leakage-free protocol. Binary reaches
   77.1% (Ensemble), a defensible, literature-comparable number. Showed that 4-class accuracy
   can appear misleadingly close to 3-class accuracy due to an unlearnable minority class
   (Severe, n=24) rarely being predicted, while macro-F1 correctly reveals the 3-class result
   as more balanced/honest.

## Dataset
PhysioNet "Gait in Parkinson's Disease" (gaitpdb v1.0.0)
- 93 PD patients + 73 healthy controls (166 subjects), 306 trials across 3 sub-cohorts (Ga/Ju/Si)
- 16 VGRF foot sensors (8 per foot) + 2 total-force channels, 100 Hz
- Labels: Hoehn & Yahr scale (2, 2.5, 3) + healthy control — confirmed matches paper exactly
  (Healthy 73 / Mild 55 / Moderate 28 / Severe 10 at subject level)

## Known limitation (disclosed, not hidden)
Step length, stride length, and gait speed (reported in the original paper) require spatial/
distance capture not present in this VGRF-only public dataset — left as NaN / excluded.
COP features partially substitute with real spatial information derived from sensor geometry.

## Datasets (final set)
| Model | Dataset | Subjects | Signal | Access |
|---|---|---|---|---|
| A: Gait | PhysioNet gaitpdb v1.0.0 | 166 (93 PD, 73 healthy) | Foot VGRF, 100 Hz | Open |
| B: Wrist tremor / hand movement | PADS (PhysioNet Parkinson's Disease Smartwatch) v1.0.0 | 469 (PD, differential dx, healthy) | Wrist accel + gyro, 100 Hz, 11 movement tasks | Open, CC BY-NC-SA 4.0, ~1.4 GB |
| C: Voice | UCI Parkinson's Speech with Multiple Types of Sound Recordings (#301) | 40 train (20 PD, 20 healthy) + 28 PD-only test | 26 recordings/subject, 26 extracted acoustic features (no raw audio) | Open, CC BY 4.0 |

Notes:
- PADS documented tasks are rest, postural and kinetic (no confirmed finger-tapping), so Model B
  is a wrist tremor/hand-movement model. Main task PD vs healthy; PD vs differential dx optional.
  Group counts to be confirmed from PADS metadata after download.
- UCI test set has PD patients only (no controls), so use the 40-subject training set with
  subject-grouped CV. Voice is small; disclose this.
- Only gaitpdb has H&Y staging; B and C are binary. Optional: a raw-audio voice dataset
  (e.g. a Figshare vowel /a/ set, unverified) for richer features.
- The three datasets contain different people, so fusion is a demo, not a validated result.

## Expected output (per patient)
For a patient's gait / wrist / voice recordings the system reports:
1. **PD risk:** estimated probability (chance) of Parkinson's disease from each model, plus a
   fused overall risk score (calibrated probabilities, simple or validation-weighted average).
2. **Feature attribution:** which features most increased or decreased the risk, via SHAP,
   named clinically (e.g. stride-time variability, left-right asymmetry, stance ratio, COP sway,
   tremor band power, jitter/shimmer).
3. **Plain-language summary,** e.g. "Elevated PD risk (0.72), driven mainly by high stride
   variability and asymmetric stance."
4. **Global view:** ranked feature importance per model, plus per-patient waterfall plots.

Caveat to state in the paper: outputs are research screening estimates, not a diagnosis;
per-modality accuracy is validated, fused accuracy is not (no paired data); SHAP shows what the
model used, not disease mechanism.

## Success criteria
- [x] Clean, reproducible pipeline (raw → features → model → metrics)
- [x] Leakage identified, quantified, and fixed
- [x] Measurable improvement from feature engineering under honest evaluation
- [ ] Leakage comparison repeated on PADS and UCI speech
- [ ] SHAP explanations + per-patient PD-risk report
- [ ] Late-fusion demo
- [ ] Paper draft with intro, related work, methodology, results, discussion, conclusion
- [x] All results reproducible from scripts + fixed random seed (42)

## Non-goals
- Clinical deployment or regulatory validation
- Real-time gait monitoring system
- Building a UI/app
- Beating the baseline's (likely leakage-inflated) 99.4% headline number — the point is
  correcting the evaluation methodology, not chasing a bigger number

## Stakeholders
- Devanshi (author, implementer)
- Kamakshi (author, implementer)
- Shreya (author, implementer)