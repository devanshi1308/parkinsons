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
- **Step length, stride length and gait speed are excluded.** gaitpdb has no spatial data. Any
  height-based proxy must be labelled as a proxy.
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
