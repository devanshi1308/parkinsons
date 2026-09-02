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

## Novel contribution (LOCKED IN)
**Subject-level data leakage in prior gait-PD classification work, and a corrected,
leakage-free evaluation with richer spatiotemporal features.**

Specifically:
1. **Diagnosis**: Showed that trial-level cross-validation (used in the baseline paper and
   likely similar prior work) allows the same subject's trials to appear in both train and
   test folds, inflating reported accuracy. Reproducing the baseline's trial-level CV gave
   ~67-70% on our pipeline; switching to subject-grouped CV (StratifiedGroupKFold, no subject
   ever split across train/test) dropped this to ~40% — a ~30 percentage point gap attributable
   to leakage alone, not model quality.
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

## Success criteria
- [x] Clean, reproducible pipeline (raw → features → model → metrics)
- [x] Leakage identified, quantified, and fixed
- [x] Measurable improvement from feature engineering under honest evaluation
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