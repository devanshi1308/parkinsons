# CLAUDE.md

Guidance for Claude Code in this repo. Read `prd.md`, `architecture.md`, `implementation.md`
and `rules.md` (decision log) for detail; this file summarizes the project and every decision
made so far.

## Project
Research paper on Parkinson's disease (PD) screening from gait, wrist-sensor and voice data.
Authors: Devanshi Kambli, Kamakshi, Shreya.

**Thesis (main claim):** published gait-based PD classifiers overstate accuracy because of
subject-level leakage (same person's trials in both train and test). We measure the inflation,
correct it with subject-grouped CV, and show honest models can still be informative and
explainable.

**Baseline:** Balaji et al. 2020, Applied Soft Computing (`docs/paper.pdf`). It reports ~99%; our
re-implementation does not match that, and the paper must say so plainly (we "re-implement and
evaluate under both protocols", not "reproduce").

**Not this paper:** clinical deployment, an app/UI, chasing the baseline's number. Outputs are
research screening estimates, not diagnoses.

## Datasets
| Model | Dataset | Notes |
|---|---|---|
| A: Gait | PhysioNet gaitpdb v1.0.0 | 166 documented subjects (165 unique IDs in `features_v2.csv`), 306 trials, cohorts Ga/Ju/Si, 16 VGRF sensors + 2 totals, 100 Hz. H&Y: Healthy 73 / Mild 55 / Moderate 28 / Severe 10 subjects (Severe = 24 trials) |
| B: Wrist | PADS (PhysioNet smartwatch) | 469 participants, accel+gyro 100 Hz, 11 tasks, open CC BY-NC-SA 4.0. Not downloaded; group counts unchecked. No confirmed finger-tapping task, so this is a wrist tremor / hand-movement model |
| C: Voice | UCI Parkinson's Speech #301 | 40 training subjects (20 PD/20 healthy) x 26 recordings, 26 provided acoustic features, no raw audio; test set is 28 PD only (unusable for accuracy). Not downloaded |

Different people in each dataset: never merge rows across them. Raw data is git-ignored; see
`data-README.md` for download steps.

## Decisions (chronological summary)
**Scope and framing**
- Gap locked in: subject-level leakage + corrected leakage-free evaluation + richer features.
- Paper stays leakage-first. Voice and wrist results are generalization evidence (repeat
  leaky-vs-grouped); SHAP is interpretability; late fusion is a demo only.
- Multimodal design: three separate models, one per dataset, combined only at prediction time by
  calibrated late fusion (simple average or validation-AUC weights, never tuned on test data).
  Fused accuracy is NOT claimed (no public dataset has all modalities on the same people).
- Dropped modalities: handwriting/spiral drawing, eyes/pupils. Face deferred (no large public
  data). Voice chosen over face because far more public data exists.
- Alpha-synuclein aggregation idea ("Kanya" tool from "Vercula") is out of scope for this paper.
  The tool could not be verified (no search results). If pursued, make it a separate project
  using established predictors (TANGO, AGGRESCAN, Zyggregator, CamSol, WALTZ, PASTA 2.0,
  FoldAmyloid), UniProt P37840, literature-based mutants (A30P, E46K, H50Q, G51D, A53E, A53T),
  and validation against published experiments with primary citations.
- Expected per-patient output: PD risk per modality + fused score, top contributing features
  (SHAP, clinically named), one plain-language sentence. Only gaitpdb supports H&Y staging; B
  and C are binary.

**Evaluation protocol (hard rules)**
1. Primary protocol: `StratifiedGroupKFold` by `subject_id`, seed 42, 10 folds (capped by
   smallest class).
2. Feature selection, scaling, SMOTE and augmentation fit inside training folds only.
3. Windows/recordings from one subject stay in one fold.
4. Report accuracy, macro-F1, sensitivity, specificity, AUC with fold variance.
5. Never claim fused accuracy; never fabricate numbers, citations or dataset facts; mark
   anything unverified.

**Labels**
- 4-class Healthy/Mild(H&Y 2)/Moderate(2.5)/Severe(3); 3-class merges Moderate+Severe;
  binary Healthy vs PD. Severe (10 subjects) is effectively unlearnable (F1 ~0-0.13); state this.

**Features**
- Step length, stride length, gait speed excluded (gaitpdb has no spatial data; cannot be filled
  from other datasets because they are different people). Height-based proxies are allowed only
  if labelled as proxies.
- v2 features: statistics (+min/max/range/IQR/skew/kurtosis), COP from sensor geometry,
  stride-time CV, left-right symmetry, hysteresis stance/swing.
- Planned v3: demographics (age confound: report with/without), FFT/wavelet, regularity
  (autocorrelation, sample entropy, DFA), double-support time.

**Models**
- Used: Decision Tree, linear SVM, Bagging (30 trees), GaussianNB (mirrors the baseline).
- Planned: Random Forest/Extra Trees, XGBoost/LightGBM, logistic regression/LDA, stacking;
  optional sliding-window LSTM with augmentation (expect it to underperform at 165 subjects;
  report honestly). Ordinal models are a possible fit for H&Y.

**Making the data "bigger"**
- Real limit is subjects (166), not rows. Windows/augmentation add rows, not subjects. Only
  external datasets add subjects, and they need shared features and binary labels.
- Class imbalance: class weights and SMOTE tried; no meaningful gain.

**Repo/process**
- `.gitignore` (was misnamed `gitignore`) fixed; data folders and processed CSVs are ignored.
  `docs/paper.pdf` was committed; confirm it is fine in a public repo (possibly copyrighted).
- Dr. Minirani removed from all files (user request).
- I (Claude) make no commits or pushes; give the user the commands.

## Measured results (keep in sync with `data/processed/*.csv`)
Grouped 10-fold, seed 42 (`classifier_results_all_schemes.csv`):
- 4-class: DT 43.2%, SVM 42.8%, Bagging 48.8%, NB 38.8%
- 3-class: best Bagging 46.4%; binary: best Bagging 77.1%
- Per-cohort (pooled model): Ga 46.9%, Ju 31.8% (weakest), Si 62.5%; leave-one-cohort-out 39-48%

Leakage comparison, `scripts/14_leakage_comparison.py`, 5 seeds (`leakage_comparison.csv`,
`leakage_audit.csv`, `docs/figures/leakage_gap.png`):
- 4-class leaky 53-76% vs grouped 38-46%; gap 15-30 points (~30 for Bagging and SVM)
- 3-class gap 9-32 points; binary gap only 1-11 points
- ~63% of leaky-fold test trials belong to a subject also in training
- In-fold preprocessing gives the same result
- Use multi-seed Bagging 4-class grouped = 46% in the paper (48.8% is a single seed)
- The old "~67-70% vs ~40%" claim had no script behind it; it was replaced by these numbers.
  Any figure in a doc must trace to a script and a CSV.

## Repo layout
```
scripts/   numbered pipeline (01-11, 14 done; 12, 13, 20-22, 30-32, 40, 50 planned)
data/      raw/ raw_sorted/ interim/ processed/ (git-ignored); raw_pads/, raw_voice/ planned
docs/      paper.pdf, figures/
prd.md architecture.md implementation.md rules.md prompt.md data-README.md README.md
```
Run scripts from the project root with `venv/bin/python scripts/NN_name.py`. New scripts follow
the numbering and style of existing ones (docstring header, pandas/sklearn, pathlib). Reuse
`07_train_multi_scheme.py` loaders instead of duplicating them.

## Open items
- Leaky-vs-grouped comparison for PADS and UCI speech; download PADS and check group counts.
- Compare against baseline Tables 5-9 (no script yet); ROC plots; review Ju investigation
  output (`11_investigate_ju_cohort.py`); find the missing 166th subject.
- Features v3, extra models, SHAP, fusion demo, paper writing, venue selection.
- Update `implementation.md` checkboxes and `rules.md` whenever a decision or milestone changes.

## Style for answers
Concise, recommendation first, no exhaustive option surveys. Use they/them for people whose
pronouns are unknown.
