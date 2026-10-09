"""
32_voice_shap.py
Out-of-fold SHAP explanation for the voice modality (Healthy vs PD), mirroring
40_shap_report.py's protocol so gait and voice have parity in the paper: every recording is
explained by a model that never saw that subject (StratifiedGroupKFold by subject_id, 40-subject
train partition only -- see 30_voice_load.py for why the test partition is excluded).

Outputs:
  docs/figures/voice_shap_top_features.png
  docs/figures/voice_shap_beeswarm.png
  data/processed/voice_shap_importance.csv

Run from project root:
    python scripts/32_voice_shap.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

FEATURES_PATH = Path("data/processed/voice_features.csv")
FIG_DIR = Path("docs/figures")
OUT_IMPORTANCE = Path("data/processed/voice_shap_importance.csv")
SEED = 42
N_FOLDS = 10

CLINICAL_NAMES = {
    "jitter_local": "Jitter (local) - pitch-period variation", "jitter_local_abs": "Jitter (local, absolute)",
    "jitter_rap": "Jitter (RAP)", "jitter_ppq5": "Jitter (PPQ5)", "jitter_ddp": "Jitter (DDP)",
    "shimmer_local": "Shimmer (local) - amplitude variation", "shimmer_local_db": "Shimmer (local, dB)",
    "shimmer_apq3": "Shimmer (APQ3)", "shimmer_apq5": "Shimmer (APQ5)", "shimmer_apq11": "Shimmer (APQ11)", "shimmer_dda": "Shimmer (DDA)",
    "ac": "Autocorrelation (AC)", "nth": "Noise-to-harmonics ratio (NTH)", "htn": "Harmonics-to-noise ratio (HTN)",
    "pitch_median": "Median pitch (F0)", "pitch_mean": "Mean pitch (F0)", "pitch_sd": "Pitch variability (SD)",
    "pitch_min": "Minimum pitch", "pitch_max": "Maximum pitch",
    "n_pulses": "Number of glottal pulses", "n_periods": "Number of pitch periods",
    "period_mean": "Mean pitch period", "period_sd": "Pitch-period variability (SD)",
    "frac_unvoiced": "Fraction of unvoiced frames", "n_voice_breaks": "Number of voice breaks",
    "degree_voice_breaks": "Degree of voice breaks",
}


def main():
    df = pd.read_csv(FEATURES_PATH)
    df = df[df.partition == "train"].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "subject_num", "partition", "label", "class", "updrs")]
    X = df[feature_cols].to_numpy(float)
    y = LabelEncoder().fit_transform(df["label"])  # Healthy=0, PD=1 (alphabetical)
    subj = df["subject_id"].to_numpy()
    print(f"Recordings {len(df)}, subjects {df.subject_id.nunique()}, features {X.shape[1]}, PD share {y.mean():.2f}")

    n, p = X.shape
    sv_all = np.zeros((n, p))
    prob = np.zeros(n)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(X, y, subj)):
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        sv = shap.TreeExplainer(rf).shap_values(X[te])
        sv1 = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
        sv_all[te] = sv1
        prob[te] = rf.predict_proba(X[te])[:, 1]
        print(f"  fold {k + 1}/{N_FOLDS} done")

    pred = (prob >= 0.5).astype(int)
    print(f"Out-of-fold recording accuracy: {np.mean(pred == y):.3f}")

    names = [CLINICAL_NAMES.get(c, c) for c in feature_cols]
    imp_df = pd.DataFrame({"feature": feature_cols, "clinical_name": names, "mean_abs_shap": np.abs(sv_all).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    OUT_IMPORTANCE.parent.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(OUT_IMPORTANCE, index=False)
    print("\nTop 10 voice features by mean |SHAP|:")
    print(imp_df.head(10)[["clinical_name", "mean_abs_shap"]].to_string(index=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    top = imp_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.clinical_name, top.mean_abs_shap, color="#8e6bb0")
    ax.set_xlabel("Mean |SHAP| (effect on PD probability)")
    ax.set_title("Voice: top features driving PD vs healthy (out-of-fold Random Forest)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG_DIR / "voice_shap_top_features.png", dpi=200); plt.close(fig)

    order = [feature_cols.index(f) for f in imp_df.head(12).feature]
    shap.summary_plot(sv_all[:, order], X[:, order], feature_names=[names[i] for i in order], max_display=12, show=False, plot_size=(9, 5.5))
    plt.title("Voice: effect of each feature on PD probability (each dot = one recording)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "voice_shap_beeswarm.png", dpi=200); plt.close()
    print(f"\nSaved figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
