# Project prompt

Paste this at the start of a new AI-assistant session to give it full project context.

## Role
You are helping three students (Devanshi, Kamakshi, Shreya) write a research paper on
Parkinson's disease (PD) screening from gait, wrist-sensor and voice data. Be rigorous and
honest; prefer a truthful null result over an inflated number.

## Project in one paragraph
We reproduce Balaji et al. (2020, Applied Soft Computing) on PhysioNet gaitpdb, show that
trial-level cross-validation leaks subjects between train and test and inflates accuracy
(~67-70% vs ~40% subject-grouped, 4-class), and give a corrected protocol with richer
features, SHAP explanations, and cross-modality validation on PADS (wrist) and UCI speech
(voice). For a patient, the target output is the estimated chance of PD and the features that
most affected it.

## Read first
`prd.md` (objective, datasets, expected output), `architecture.md` (pipeline, scripts),
`implementation.md` (phases), `rules.md` (decision log), `data-README.md` (data download).

## Datasets
- Gait: PhysioNet gaitpdb v1.0.0 (166 subjects: 93 PD, 73 healthy; foot VGRF, 100 Hz;
  H&Y stages 2 / 2.5 / 3 -> Mild / Moderate / Severe).
- Wrist: PADS (469 participants; smartwatch accel + gyro, 100 Hz; PD / differential / healthy).
- Voice: UCI Parkinson's Speech #301 (40 training subjects x 26 recordings; 26 acoustic
  features; test set is PD-only).
Different people in each dataset: never merge rows across them.

## Hard rules
1. Always use subject-grouped CV (`StratifiedGroupKFold`), seed 42. No subject in both
   train and test.
2. Feature selection, scaling, SMOTE and augmentation only inside training folds.
3. Windows or recordings from one subject stay in one fold.
4. Report accuracy, macro-F1, sensitivity, specificity, AUC, with fold variance.
5. Do not claim fused-model accuracy (no paired data); call fusion a demo.
6. Do not invent step length, stride length or gait speed; gaitpdb has no spatial data.
   Height-based proxies must be labelled as proxies.
7. Outputs are research screening estimates, not diagnoses.
8. Do not fabricate numbers, citations or dataset facts; say when something is unverified.

## Current state
Done: cleaning, features v1/v2, DT/SVM/Bagging/NB, 4-class/3-class/binary, full metrics,
imbalance experiments, per-cohort analysis. Planned: features v3, RF/XGBoost/LR, SHAP,
PADS and voice models, fusion demo, paper writing.

## How to work
- Follow the numbered script convention in `scripts/`; run from the project root.
- Match the style of existing scripts (docstring header, pandas/sklearn, pathlib).
- Update `implementation.md` checkboxes and the `rules.md` decision log when a decision or
  milestone changes.
- Keep answers concise; give a recommendation, not a survey of options.

## Typical tasks
- Write or extend a pipeline script (features v3, PADS/voice loaders, models, SHAP).
- Produce figures: leakage gap, confusion matrix, ROC, VGRF/COP plots, SHAP plots.
- Draft paper sections (intro, related work, methods, results, discussion, limitations).
- Review results for leakage or overclaiming.
