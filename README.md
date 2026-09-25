# Leakage-free, explainable Parkinson's disease screening

Reproduces Balaji et al. (2020) gait-based PD classification on PhysioNet gaitpdb, shows that
trial-level cross-validation leaks subjects between train and test and inflates accuracy
(~67-70% vs ~40% subject-grouped, 4-class), and provides a corrected protocol with richer
features, SHAP explanations, and cross-modality validation on wrist-sensor (PADS) and voice
(UCI speech) data.

For a patient, the goal output is the estimated chance of Parkinson's disease plus the
features (e.g. stride variability, asymmetry, tremor power) that most affected that estimate.
This is a research screening estimate, not a diagnosis.

## Status
- Gait pipeline: cleaning, features v1/v2, classifiers, multi-scheme metrics, imbalance,
  per-cohort analysis: done.
- Planned: features v3, more models, SHAP, PADS and voice models, late-fusion demo, paper.

See [prd.md](prd.md) (objective, datasets, expected output), [architecture.md](architecture.md)
(pipeline and scripts), [implementation.md](implementation.md) (phases) and
[rules.md](rules.md) (decision log).

## Datasets
| Model | Dataset | Notes |
|---|---|---|
| Gait | PhysioNet gaitpdb v1.0.0 | 166 subjects, foot VGRF |
| Wrist tremor | PADS (PhysioNet) | 469 participants, smartwatch accel/gyro |
| Voice | UCI Parkinson's Speech (#301) | 40 training subjects, 26 recordings each |

Data is not committed. Download steps: [data-README.md](data-README.md).

## Run (gait pipeline)
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # if present; otherwise numpy pandas scikit-learn scipy
python scripts/01_organize_raw.py
python scripts/02_clean_signals.py
python scripts/03_extract_features_v2.py
python scripts/04_merge_labels.py
python scripts/07_train_multi_scheme.py
python scripts/08_full_metrics.py
```
All experiments use seed 42 and subject-grouped CV.

## Authors
Devanshi Kambli, Kamakshi, Shreya.
