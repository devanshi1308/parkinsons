# Fusion demo report (exploratory)

**Scope note.** This demo combines a gait model and a voice model. Wrist/PADS is not included -- that download was not feasible in this environment (see `CLAUDE.md`). The gait and voice patients shown below are DIFFERENT REAL PEOPLE from two different public datasets (no dataset provides gait, voice and wrist recordings from the same subjects). This script therefore demonstrates the fusion MECHANICS -- per-modality calibration, a combined score, a feature-attribution report -- and does NOT claim a validated fused-accuracy number. Any 'fused score' below is illustrative only, computed as a simple average of two unrelated patients' risk scores, as a stand-in for what the pipeline would produce if paired multimodal data existed.

## Example 1: a gait patient and a voice patient both flagged high-risk

### Gait: GaPt20

- Recorded label: **Mild**
- Calibrated PD-risk score: **96%**
- Top contributing features (from the modality's out-of-fold SHAP analysis, not re-computed per patient here -- see `docs/figures/*_shap_*` for the full out-of-fold, per-trial SHAP attributions):
  - Walking speed
  - Left-foot sensor-5 force skewness
  - Centre-of-pressure front-back sway (left foot)
  - Double-support share of walking time

### Voice: TRAIN2

- Recorded label: **PD**
- Calibrated PD-risk score: **57%**
- Top contributing features (from the modality's out-of-fold SHAP analysis, not re-computed per patient here -- see `docs/figures/*_shap_*` for the full out-of-fold, per-trial SHAP attributions):
  - Pitch variability (SD)
  - Jitter (PPQ5)
  - Maximum pitch
  - Fraction of unvoiced frames

**Illustrative fused score (simple average, NOT a validated joint estimate): 76%**

## Example 2: a gait patient and a voice patient both flagged low-risk

### Gait: SiCo23

- Recorded label: **Healthy**
- Calibrated PD-risk score: **12%**
- Top contributing features (from the modality's out-of-fold SHAP analysis, not re-computed per patient here -- see `docs/figures/*_shap_*` for the full out-of-fold, per-trial SHAP attributions):
  - Walking speed
  - Left-foot sensor-5 force skewness
  - Centre-of-pressure front-back sway (left foot)
  - Double-support share of walking time

### Voice: TRAIN27

- Recorded label: **Healthy**
- Calibrated PD-risk score: **44%**
- Top contributing features (from the modality's out-of-fold SHAP analysis, not re-computed per patient here -- see `docs/figures/*_shap_*` for the full out-of-fold, per-trial SHAP attributions):
  - Pitch variability (SD)
  - Jitter (PPQ5)
  - Maximum pitch
  - Fraction of unvoiced frames

**Illustrative fused score (simple average, NOT a validated joint estimate): 28%**

## How this would work with paired (same-patient) multimodal data

1. Calibrate each modality's model on its own training data (Platt/isotonic scaling, as done here).
2. Combine calibrated probabilities with a simple average, or weights set from each model's held-out validation AUC (gait ~0.86, voice ~0.57 -- see `docs/figures/roc_binary_gait_voice.png`; weighting down the near-chance voice model would be the honest choice if paired data existed).
3. Never tune fusion weights on the same data used to report the final accuracy.
4. Report top SHAP features per modality alongside the fused score, as above.

This would need a dataset with gait, voice (and ideally wrist) recordings from the same patients to validate -- no such public dataset currently exists for this project.