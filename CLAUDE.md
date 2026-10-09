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
- Correction: gait speed IS available (`Speed_01` in demographics.xls, m/s, 165/166 subjects, timed by
  clinicians in trial 01). Stride/step length are DERIVED ESTIMATES (speed x stride time), always
  labelled as estimates; measured distance is impossible from VGRF. Values cannot be borrowed from
  other datasets (different people). UPDRS/UPDRSM/TUAG/H&Y are never features (label proxies).
  Height unit mix (m for Ga, cm for Ju/Si) is normalised in script 03 v3.
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

Features v3 and models (`scripts/12_train_models_v3.py`, 3 seeds, grouped CV; `model_comparison_v3.csv`):
- Binary best ~83% accuracy, AUC ~0.90 (RF/Voting/HGB, `v3_all`); baseline four models up to 77%
- RF on v2 alone is 81%: most gain is from better models; speed +1.5-2 pts; new signal families ~0
- Demographics+speed only: 73% (RF) to 78% (SVM) binary; always report this baseline beside gait results
- 4-class best ~51% vs 46-49% before; small vs fold SD ~0.15, not established
- Best-of-45 selection on the same folds is optimistic: exploratory until nested CV or LOCO confirms
- SHAP (`40_shap_report.py`): out-of-fold; sensible directions (higher speed/stride length lowers risk,
  higher double-support raises it); outputs in docs/figures/shap_*.png and docs/patient_reports.md

Model-improvement follow-ups (`scripts/16_model_improvements.py`, `docs/model_improvements.md`):
- De-biased (nested CV) numbers to actually cite: 4-class 42.1% (sd 8.3), binary 79.4% (sd 9.7) --
  lower than model_comparison_v3's best-of-45 table (51%, 83%), confirming selection bias there
- Subject-level probability aggregation: no help (null result)
- Ordinal (Frank-Hall) vs multiclass 4-class: ordinal WORSE on mean-abs-stage-error; don't adopt
- Compact 5-feature set (speed, 2x sensor-5/6/7 force shape, COP sway) recovers 78.0% vs full 82.3%
  binary -- use this for the clinician-facing SHAP story
- XGBoost/LightGBM now work (`brew install libomp` fixed it)
- Nested CV extended to include them (`17_nested_with_boosting.py`) -- FINAL numbers to cite:
  4-class 44.4% (sd 10.8), binary 81.7% (sd 10.6). Binary gain over sklearn-only-nested (79.4%) is
  real but modest; 4-class gain (vs 42.1%) is within noise and no model dominates fold selection.

Voice (UCI #301), done (`30_voice_load.py`, `31_voice_leakage_comparison.py`, `32_voice_shap.py`):
- Downloaded via curl+unar into data/raw_voice/; 40 train subj (20/20) x 26 rec, 28 test subj
  (PD-only, unusable for accuracy)
- Leakage: recording-level 58.4-69.2% vs subject-grouped 52.2-55.7% (gap 3.8-15.2 pts); 100% of
  leaky-fold test recordings share a subject with training (vs ~63% gait) -- cleanest leakage
  demo in the project; KEY generalization result for the thesis
- SHAP: pitch variability (SD), jitter (PPQ5), max pitch, unvoiced-frame fraction, shimmer (APQ11)
  top drivers -- matches established PD-dysphonia literature

Statistical robustness (`scripts/18_statistical_robustness.py`, `docs/statistical_robustness.md`):
- Leakage gap is significant for EVERY classifier/scheme/modality (Wilcoxon p<0.001, Cohen's d
  1.4-12.6, 20 seeds) -- use these 20-seed numbers as final, supersedes the 5-seed ones
- Permutation test (chance-level), grouped protocol, Bagging: gait binary p=0.0099, gait 4-class
  p=0.0198 (both above chance) -- but VOICE BINARY p=0.1485, NOT significant. Report honestly:
  voice's value is replicating the LEAKAGE MECHANISM, not a validated diagnostic result (n=40 is
  small). Do not claim the grouped voice model confidently beats chance.
- Capacity-vs-gap table: gap size tracks model capacity to memorize subjects (NB smallest,
  Bagging largest) in nearly every scheme/modality -- a mechanistic, not just correlational, story

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

## Next steps (update this section as work lands — keep it current, not a historical log)
In priority order:
1. **PADS wrist model** (`20`-`22`, not started): download PADS (~1.4 GB, needs the user's
   go-ahead first), read its metadata for true group counts (PD / differential dx / healthy —
   unconfirmed, no finger-tapping task documented), extract tremor/movement features, run
   subject-grouped CV, repeat the leaky-vs-grouped comparison (mirror `14_leakage_comparison.py`).
2. **UCI speech voice model** (`30`-`32`, not started): load the 26 provided acoustic features,
   subject-grouped CV, repeat the leaky-vs-grouped comparison.
3. **Re-run the nested-CV correction (script 16 §1) for XGBoost/LightGBM too** — they were added
   in §5 of the same script but not yet validated with nested selection, so their 83.8-84.5%/
   50-52% numbers carry the same optimism risk already found and corrected for the sklearn models.
4. **SHAP for the 4-class model** (`40_shap_report.py` currently only covers binary) and switch
   its feature list to the 5-feature compact set from `16_model_improvements.py` §4 for the
   clinician-facing report, instead of the ~200-feature raw-sensor list.
5. **Review the Ju-cohort investigation output** (`11_investigate_ju_cohort.py`, run but never
   read) and **find the missing 166th subject** (165 unique IDs in `features_v2.csv`/`features_v3.csv`
   vs 166 documented).
6. Late-fusion demo (`50_fusion_demo.py`, not started) — after B and C exist; demo only, no
   fused-accuracy claim.
7. Remaining figures: confusion matrix, ROC curves, VGRF/COP plots (leakage-gap and SHAP plots
   are already done).
8. Optional sliding-window LSTM (`13_lstm_windows.py`) — expected to underperform at 165
   subjects; low priority.
9. Paper writing (Phase 5) and submission prep (Phase 6) — not started; do after B/C exist so the
   generalization section has real numbers, not placeholders.
10. `docs/project_overview.pdf` is now stale (predates features v3, nested-CV correction, SHAP,
    baseline comparison, model-improvement follow-ups) — regenerate before sharing it again.

Always keep this file in sync: update "Measured results" below and this list whenever a script
runs or a decision changes, in the same turn as the work — don't let it drift to a later cleanup.

## Open items / known gaps (longer-lived than "next steps" above)
- PADS/UCI speech not downloaded; their group counts, licensing and feature sets are unconfirmed.
- `docs/paper.pdf` copyright status in a public repo is still unconfirmed.
- `prompt.md`/`architecture.md` current-state summaries should be checked against this file
  periodically — they've needed manual re-sync more than once already.

## Style for answers
Concise, recommendation first, no exhaustive option surveys. Use they/them for people whose
pronouns are unknown.
