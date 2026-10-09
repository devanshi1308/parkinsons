"""
41_shap_4class_and_compact.py
Two follow-ups to 40_shap_report.py:

  A. Compact-feature SHAP (binary): the same out-of-fold Random Forest / SHAP protocol as
     40_shap_report.py, but restricted to the 5-feature compact set from
     16_model_improvements.py section 4 (walking speed, left-foot COP front-back sway, and
     three left-foot sensor force-shape features) instead of the ~200-feature v3_all set.
     This is the feature list to actually show a clinician.
  B. 4-class SHAP: out-of-fold multiclass Random Forest (v3_all) explained per H&Y stage
     (Healthy/Mild/Moderate/Severe), since 40_shap_report.py only covered binary.

Both use StratifiedGroupKFold by subject_id (10 folds, seed 42), so every explained trial
comes from a model that never saw that subject.

Outputs (docs/figures/):
  shap_compact_top_features.png, shap_compact_beeswarm.png
  shap_4class_top_features.png (one panel per stage), shap_4class_global_importance.png
Outputs (data/processed/):
  shap_compact_importance.csv, shap_4class_importance.csv

Run from project root:
    python scripts/41_shap_4class_and_compact.py
"""

import importlib.util
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

FEATURES_PATH = Path("data/processed/features_v3.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
COMPACT_PATH = Path("data/processed/compact_feature_list.csv")
FIG_DIR = Path("docs/figures")
OUT_DIR = Path("data/processed")
SEED = 42
N_FOLDS = 10
STAGE_ORDER = ["Healthy", "Mild", "Moderate", "Severe"]
STAGE_COLORS = {"Healthy": "#2a9d8f", "Mild": "#e9a23b", "Moderate": "#c0504d", "Severe": "#8e2a2a"}


def load_clinical_name_fn():
    spec = importlib.util.spec_from_file_location("m40", Path(__file__).with_name("40_shap_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.clinical_name


def load_data():
    feats = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    df = feats.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------- A. compact binary SHAP
def compact_shap(df, clinical_name):
    compact = pd.read_csv(COMPACT_PATH)
    cols = compact[compact.selected_fraction_of_folds >= 0.5].feature.tolist()
    print(f"Compact feature set ({len(cols)}): {cols}")
    X = df[cols].to_numpy(float)
    y = (df["stage"] != "Healthy").astype(int).to_numpy()
    subj = df["subject_id"].to_numpy()

    sv_all = np.zeros((len(y), len(cols)))
    prob = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, subj):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, y[tr])
        sv = shap.TreeExplainer(rf).shap_values(Xte)
        sv1 = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
        sv_all[te] = sv1
        prob[te] = rf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    print(f"Compact-feature out-of-fold accuracy: {np.mean(pred == y):.3f}")

    names = [clinical_name(c) for c in cols]
    imp_df = pd.DataFrame({"feature": cols, "clinical_name": names, "mean_abs_shap": np.abs(sv_all).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(OUT_DIR / "shap_compact_importance.csv", index=False)
    print(imp_df.to_string(index=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    top = imp_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.barh(top.clinical_name, top.mean_abs_shap, color="#1f3a5f")
    ax.set_xlabel("Mean |SHAP| (effect on PD probability)")
    ax.set_title("Compact 5-feature model: what drives the PD-risk estimate", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_compact_top_features.png", dpi=200); plt.close(fig)

    Ximp = SimpleImputer(strategy="median").fit_transform(X)
    shap.summary_plot(sv_all, Ximp, feature_names=names, show=False, plot_size=(8, 3.5))
    plt.title("Compact 5-feature model: effect on PD probability (each dot = one trial)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "shap_compact_beeswarm.png", dpi=200); plt.close()
    return imp_df


# --------------------------------------------------------------- B. 4-class SHAP
def shap_4class(df, groups_map, clinical_name):
    fam = ["stats", "timing", "cop", "symmetry", "spectral", "wavelet", "regularity", "double_support", "demographics", "speed"]
    cols = [c for c in groups_map[groups_map.group.isin(fam)].feature if c in df.columns and not df[c].isna().all()]
    X = df[cols].to_numpy(float)
    stage_idx = df["stage"].map({s: i for i, s in enumerate(STAGE_ORDER)}).to_numpy()
    subj = df["subject_id"].to_numpy()
    n_classes = len(STAGE_ORDER)

    sv_all = np.zeros((len(stage_idx), len(cols), n_classes))
    preds = np.zeros(len(stage_idx), int)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, stage_idx, subj):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, stage_idx[tr])
        sv = shap.TreeExplainer(rf).shap_values(Xte)  # (n_test, n_features, n_classes_in_fold)
        sv = sv if isinstance(sv, np.ndarray) else np.stack(sv, axis=-1)
        fold_classes = rf.classes_
        for j, c in enumerate(fold_classes):
            sv_all[te, :, c] = sv[:, :, j]
        preds[te] = rf.predict(Xte)
    acc = np.mean(preds == stage_idx)
    print(f"4-class out-of-fold accuracy: {acc:.3f}")

    names = [clinical_name(c) for c in cols]
    rows = []
    for c, stage in enumerate(STAGE_ORDER):
        mask = stage_idx == c  # SHAP for how the model argues FOR/AGAINST trials actually in this stage
        imp = np.abs(sv_all[mask, :, c]).mean(axis=0) if mask.any() else np.zeros(len(cols))
        for f, nm, v in zip(cols, names, imp):
            rows.append({"stage": stage, "feature": f, "clinical_name": nm, "mean_abs_shap": v})
    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(OUT_DIR / "shap_4class_importance.csv", index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n_classes, figsize=(16, 4), sharex=False)
    for ax, stage in zip(axes, STAGE_ORDER):
        top = imp_df[imp_df.stage == stage].nlargest(8, "mean_abs_shap").iloc[::-1]
        ax.barh(top.clinical_name, top.mean_abs_shap, color=STAGE_COLORS[stage])
        ax.set_title(stage, fontsize=10)
        ax.tick_params(labelsize=7.5)
    fig.suptitle("Top features by H&Y stage (out-of-fold multiclass Random Forest)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_4class_top_features.png", dpi=200); plt.close(fig)

    overall = imp_df.groupby(["feature", "clinical_name"]).mean_abs_shap.sum().sort_values(ascending=False).head(15).reset_index().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(overall.clinical_name, overall.mean_abs_shap, color="#1f3a5f")
    ax.set_xlabel("Summed mean |SHAP| across the 4 stages")
    ax.set_title("4-class model: overall top features (out-of-fold)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_4class_global_importance.png", dpi=200); plt.close(fig)

    print("\nTop 3 features per stage:")
    for stage in STAGE_ORDER:
        top3 = imp_df[imp_df.stage == stage].nlargest(3, "mean_abs_shap")
        print(f"  {stage}: " + ", ".join(top3.clinical_name))
    return imp_df, acc


def main():
    df = load_data()
    groups_map = pd.read_csv("data/processed/feature_groups_v3.csv")
    clinical_name = load_clinical_name_fn()
    print(f"Trials {len(df)}, subjects {df.subject_id.nunique()}\n")

    print("=== A. Compact-feature SHAP (binary) ===")
    compact_shap(df, clinical_name)

    print("\n=== B. 4-class SHAP ===")
    shap_4class(df, groups_map, clinical_name)


if __name__ == "__main__":
    main()
