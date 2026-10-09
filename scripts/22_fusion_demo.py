"""
22_fusion_demo.py
Exploratory late-fusion DEMO combining gait and voice PD-risk scores into one patient report.

IMPORTANT SCOPE NOTE: this is a two-modality demo (gait + voice). The wrist/PADS modality is
not included -- PADS could not be downloaded in this environment (network too throttled; see
CLAUDE.md). gait and voice are DIFFERENT PEOPLE (165 gait subjects, 40 voice subjects): no
public dataset has gait, voice and wrist recordings from the same people, so this script does
NOT claim a validated "fused accuracy". It demonstrates the mechanics of the fusion pipeline
(per-model calibration, weighted combination, a feature-attribution report) on two real
patients drawn independently from each dataset, paired only for illustration.

What this script does:
  1. Fits a gait Random Forest (binary) and a voice Random Forest (binary), each with
     out-of-fold probability calibration (Platt scaling via sklearn's CalibratedClassifierCV,
     fit on each model's own training folds only -- never on test data).
  2. For one example gait patient and one example voice patient (different real people, since
     no shared-subject data exists), produces a combined report: each modality's PD-risk
     score, a simple-average fused score, and the top SHAP-style contributing features per
     modality (reusing the importance rankings already computed in scripts 40/32).
  3. States the caveat explicitly in the report output, not just in code comments.

Output: docs/fusion_demo_report.md

Run from project root:
    python scripts/22_fusion_demo.py
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 10
OUT_PATH = Path("docs/fusion_demo_report.md")


def m07():
    spec = importlib.util.spec_from_file_location("m07", Path(__file__).with_name("07_train_multi_scheme.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gait_model_and_data():
    mod = m07()
    df = mod.load_merged()
    X = mod.build_feature_matrix(df)
    cols = X.columns.tolist()
    X = X.to_numpy(float)
    y_raw = mod.get_label_scheme(df, "binary")
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    groups = df["subject_id"].to_numpy()
    return df, X, y, groups, cols, le


def voice_model_and_data():
    df = pd.read_csv("data/processed/voice_features.csv")
    df = df[df.partition == "train"].reset_index(drop=True)
    cols = [c for c in df.columns if c not in ("subject_id", "subject_num", "partition", "label", "class", "updrs")]
    X = df[cols].to_numpy(float)
    le = LabelEncoder().fit(df["label"])
    y = le.transform(df["label"])
    groups = df["subject_id"].to_numpy()
    return df, X, y, groups, cols, le


def fit_calibrated(X, y, groups, seed=SEED, n_folds=N_FOLDS):
    """One subject-grouped train/holdout split: fit RF + Platt calibration on the 'train' part
    (calibration is itself cross-validated within that part, never touching the holdout), so the
    holdout patients used for the demo report were not used to fit or calibrate the model."""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tr_idx, ho_idx = next(cv.split(X, y, groups))
    imp = SimpleImputer(strategy="median").fit(X[tr_idx])
    Xtr, Xho = imp.transform(X[tr_idx]), imp.transform(X[ho_idx])
    base = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=inner_cv.split(Xtr, y[tr_idx], groups[tr_idx]))
    calibrated.fit(Xtr, y[tr_idx])
    return calibrated, imp, ho_idx


def report_lines(modality, patient_id, true_label, risk, top_features):
    lines = [f"### {modality}: {patient_id}", "",
             f"- Recorded label: **{true_label}**",
             f"- Calibrated PD-risk score: **{risk:.0%}**",
             "- Top contributing features (from the modality's out-of-fold SHAP analysis, "
             "not re-computed per patient here -- see `docs/figures/*_shap_*` for the full "
             "out-of-fold, per-trial SHAP attributions):"]
    lines += [f"  - {f}" for f in top_features]
    return lines


def main():
    gdf, Xg, yg, groupsg, gcols, gle = gait_model_and_data()
    gmodel, gimp, gho = fit_calibrated(Xg, yg, groupsg)
    g_prob_ho = gmodel.predict_proba(gimp.transform(Xg[gho]))[:, 1]
    print(f"Gait holdout: {len(gho)} trials, PD-risk mean {g_prob_ho.mean():.2f}")

    vdf, Xv, yv, groupsv, vcols, vle = voice_model_and_data()
    vmodel, vimp, vho = fit_calibrated(Xv, yv, groupsv)
    v_prob_ho = vmodel.predict_proba(vimp.transform(Xv[vho]))[:, 1]
    print(f"Voice holdout: {len(vho)} recordings, PD-risk mean {v_prob_ho.mean():.2f}")

    # pick one clearly-PD and one clearly-healthy example from EACH modality's holdout
    g_order = np.argsort(-g_prob_ho)
    g_pd_i, g_healthy_i = gho[g_order[0]], gho[g_order[-1]]
    v_order = np.argsort(-v_prob_ho)
    v_pd_i, v_healthy_i = vho[v_order[0]], vho[v_order[-1]]

    gait_shap_top = ["Walking speed", "Left-foot sensor-5 force skewness",
                      "Centre-of-pressure front-back sway (left foot)", "Double-support share of walking time"]
    voice_shap_top = ["Pitch variability (SD)", "Jitter (PPQ5)", "Maximum pitch", "Fraction of unvoiced frames"]

    lines = ["# Fusion demo report (exploratory)", "",
             "**Scope note.** This demo combines a gait model and a voice model. Wrist/PADS is "
             "not included -- that download was not feasible in this environment (see "
             "`CLAUDE.md`). The gait and voice patients shown below are DIFFERENT REAL PEOPLE "
             "from two different public datasets (no dataset provides gait, voice and wrist "
             "recordings from the same subjects). This script therefore demonstrates the "
             "fusion MECHANICS -- per-modality calibration, a combined score, a feature-"
             "attribution report -- and does NOT claim a validated fused-accuracy number. Any "
             "'fused score' below is illustrative only, computed as a simple average of two "
             "unrelated patients' risk scores, as a stand-in for what the pipeline would "
             "produce if paired multimodal data existed.", "",
             "## Example 1: a gait patient and a voice patient both flagged high-risk", ""]
    lines += report_lines("Gait", gdf.loc[g_pd_i, "subject_id"], gdf.loc[g_pd_i, "stage"], g_prob_ho[g_order[0]], gait_shap_top)
    lines += [""]
    lines += report_lines("Voice", vdf.loc[v_pd_i, "subject_id"], vdf.loc[v_pd_i, "label"], v_prob_ho[v_order[0]], voice_shap_top)
    fused_high = np.mean([g_prob_ho[g_order[0]], v_prob_ho[v_order[0]]])
    lines += ["", f"**Illustrative fused score (simple average, NOT a validated joint estimate): {fused_high:.0%}**", "",
              "## Example 2: a gait patient and a voice patient both flagged low-risk", ""]
    lines += report_lines("Gait", gdf.loc[g_healthy_i, "subject_id"], gdf.loc[g_healthy_i, "stage"], g_prob_ho[g_order[-1]], gait_shap_top)
    lines += [""]
    lines += report_lines("Voice", vdf.loc[v_healthy_i, "subject_id"], vdf.loc[v_healthy_i, "label"], v_prob_ho[v_order[-1]], voice_shap_top)
    fused_low = np.mean([g_prob_ho[g_order[-1]], v_prob_ho[v_order[-1]]])
    lines += ["", f"**Illustrative fused score (simple average, NOT a validated joint estimate): {fused_low:.0%}**", "",
              "## How this would work with paired (same-patient) multimodal data", "",
              "1. Calibrate each modality's model on its own training data (Platt/isotonic scaling, "
              "as done here).",
              "2. Combine calibrated probabilities with a simple average, or weights set from each "
              "model's held-out validation AUC (gait ~0.86, voice ~0.57 -- see "
              "`docs/figures/roc_binary_gait_voice.png`; weighting down the near-chance voice "
              "model would be the honest choice if paired data existed).",
              "3. Never tune fusion weights on the same data used to report the final accuracy.",
              "4. Report top SHAP features per modality alongside the fused score, as above.",
              "",
              "This would need a dataset with gait, voice (and ideally wrist) recordings from the "
              "same patients to validate -- no such public dataset currently exists for this "
              "project."]
    OUT_PATH.write_text("\n".join(lines))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
