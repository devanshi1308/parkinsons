"""
40_shap_report.py
Out-of-fold SHAP explanations for the binary (Healthy vs PD) Random Forest.

Every trial is explained by a model that never saw that trial's subject (StratifiedGroupKFold by
subject), so explanations and PD-risk estimates are honest, not fit-on-everything.

Outputs
  docs/figures/shap_top_features.png     mean |SHAP| of the top features (clinical names)
  docs/figures/shap_groups.png           importance summed by feature family
  docs/figures/shap_beeswarm.png         direction/size of effects for the top features
  docs/figures/shap_patient_*.png        per-patient contribution charts (3 example trials)
  data/processed/shap_global_importance.csv
  docs/patient_reports.md                plain-language reports for the example trials

Run from project root:
    python scripts/40_shap_report.py
"""

import importlib.util
import re
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

FEATURE_SET = "v3_all"          # feature families used (see 12_train_models_v3.py)
FAMILIES = ["stats", "timing", "cop", "symmetry", "spectral", "wavelet", "regularity", "double_support", "demographics", "speed"]
SEED = 42
N_FOLDS = 10
FEATURES_PATH = Path("data/processed/features_v3.csv")
GROUPS_PATH = Path("data/processed/feature_groups_v3.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
FIG_DIR = Path("docs/figures")
OUT_IMPORTANCE = Path("data/processed/shap_global_importance.csv")
OUT_REPORT = Path("docs/patient_reports.md")

STAT_NAMES = {"mean": "mean force", "median": "median force", "sd": "force variability (SD)", "min": "minimum force",
              "max": "peak force", "range": "force range", "iqr": "force spread (IQR)", "skew": "force skewness", "kurtosis": "force kurtosis"}
NAMED = {
    "speed_mps": "Walking speed", "stride_length_est_m": "Stride length (estimated from speed)", "step_length_est_m": "Step length (estimated)",
    "stride_length_over_height": "Stride length relative to height (estimated)", "age": "Age", "sex_male": "Sex (male)", "height": "Height",
    "weight": "Weight", "bmi": "BMI", "cadence_steps_per_min": "Cadence", "double_support_pct": "Double-support share of walking time",
    "stance_time_symmetry_index": "Stance-time asymmetry (L vs R)", "stride_time_symmetry_index": "Stride-time asymmetry (L vs R)",
    "step_regularity": "Step regularity", "stride_regularity": "Stride regularity", "regularity_ratio": "Stride/step regularity ratio",
    "stride_time_sampen": "Stride-time irregularity (sample entropy)", "stride_time_sd": "Stride-time variability (SD)",
}


def clinical_name(col: str) -> str:
    if col in NAMED:
        return NAMED[col]
    side = {"L": "left", "R": "right"}
    m = re.match(r"^([LR])_(stance_time|swing_time|stride_time|stride_time_cv|step_time_cv|stance_pct_time)$", col)
    if m:
        label = {"stance_time": "Stance time", "swing_time": "Swing time", "stride_time": "Stride time",
                 "stride_time_cv": "Stride-time variability (CV)", "step_time_cv": "Step-time variability (CV)", "stance_pct_time": "Time in stance"}[m.group(2)]
        return f"{label}, {side[m.group(1)]} foot"
    m = re.match(r"^([LR])_cop_(x_sd|y_sd|path_length)$", col)
    if m:
        label = {"x_sd": "Centre-of-pressure side-to-side sway", "y_sd": "Centre-of-pressure front-back sway", "path_length": "Centre-of-pressure path length"}[m.group(2)]
        return f"{label}, {side[m.group(1)]} foot"
    m = re.match(r"^([LR])(\d)_(\w+)$", col)
    if m and m.group(3) in STAT_NAMES:
        return f"Sensor {m.group(2)} ({side[m.group(1)]} foot) {STAT_NAMES[m.group(3)]}"
    m = re.match(r"^spec_([LR])_(.+)$", col)
    if m:
        return f"Spectral {m.group(2).replace('_', ' ')} ({side[m.group(1)]} foot)"
    m = re.match(r"^wav_([LR])_(.+)_relenergy$", col)
    if m:
        return f"Wavelet {m.group(2)} energy share ({side[m.group(1)]} foot)"
    return col.replace("_", " ")


def load_data():
    feats = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    groups_map = pd.read_csv(GROUPS_PATH)
    df = feats.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"]).reset_index(drop=True)
    cols = [c for c in groups_map[groups_map.group.isin(FAMILIES)].feature if c in df.columns and not df[c].isna().all()]
    y = (df["stage"] != "Healthy").astype(int).to_numpy()
    return df, cols, y, groups_map.set_index("feature")["group"]


def main():
    df, cols, y, fam = load_data()
    X = df[cols].to_numpy(float)
    subj = df["subject_id"].to_numpy()
    n, p = X.shape
    print(f"Trials {n}, subjects {df['subject_id'].nunique()}, features {p}, PD share {y.mean():.2f}")

    sv_all = np.zeros((n, p))
    base_all = np.zeros(n)
    prob = np.zeros(n)
    corr_cls = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("m12", Path(__file__).with_name("12_train_models_v3.py")))
    spec.loader.exec_module(corr_cls)

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(X, y, subj)):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
        var = VarianceThreshold(1e-6).fit(Xtr)
        keep1 = var.get_support()
        Xtr, Xte = Xtr[:, keep1], Xte[:, keep1]
        cf = corr_cls.CorrelationFilter(0.95).fit(Xtr)
        keep2 = cf.keep_
        Xtr, Xte = Xtr[:, keep2], Xte[:, keep2]
        idx = np.flatnonzero(keep1)[keep2]                      # positions in the original feature list
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, y[tr])
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(Xte)
        sv1 = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
        base = explainer.expected_value
        base = float(base[1]) if np.ndim(base) else float(base)
        sv_all[np.ix_(te, idx)] = sv1
        base_all[te] = base
        prob[te] = rf.predict_proba(Xte)[:, 1]
        print(f"  fold {k + 1}/{N_FOLDS} done")

    pred = (prob >= 0.5).astype(int)
    print(f"Out-of-fold trial accuracy {np.mean(pred == y):.3f}")

    names = [clinical_name(c) for c in cols]
    imp_df = pd.DataFrame({"feature": cols, "clinical_name": names, "family": [fam[c] for c in cols], "mean_abs_shap": np.abs(sv_all).mean(axis=0)})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
    OUT_IMPORTANCE.parent.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(OUT_IMPORTANCE, index=False)
    print("\nTop 15 features by mean |SHAP|:")
    print(imp_df.head(15)[["clinical_name", "family", "mean_abs_shap"]].to_string(index=False))
    fam_imp = imp_df.groupby("family").mean_abs_shap.sum().sort_values(ascending=False)
    print("\nImportance by family (sum of mean |SHAP|):")
    print(fam_imp.round(4).to_string())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.clinical_name, top.mean_abs_shap, color="#1f3a5f")
    ax.set_xlabel("Mean |SHAP| (effect on PD probability)")
    ax.set_title("Top features driving PD vs healthy (out-of-fold Random Forest)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_top_features.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    fam_imp.iloc[::-1].plot.barh(ax=ax, color="#2a9d8f")
    ax.set_xlabel("Summed mean |SHAP|"); ax.set_ylabel("")
    ax.set_title("Importance by feature family")
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_groups.png", dpi=200); plt.close(fig)

    Ximp = SimpleImputer(strategy="median").fit_transform(X)
    order = [cols.index(f) for f in imp_df.head(15).feature]
    shap.summary_plot(sv_all[:, order], Ximp[:, order], feature_names=[names[i] for i in order], max_display=15, show=False, plot_size=(9, 6))
    plt.title("Effect of each feature on PD probability (each dot = one trial)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=200); plt.close()

    # ---- per-patient examples: pick a confident, correct trial from Healthy, Mild and Severe
    examples = []
    for stage in ("Healthy", "Mild", "Severe"):
        mask = (df["stage"] == stage).to_numpy() & (pred == y)
        if mask.any():
            i = np.flatnonzero(mask)[np.argmax(np.abs(prob[mask] - 0.5))]
            examples.append((stage, i))
    healthy_median = np.nanmedian(X[y == 0], axis=0)
    lines = ["# Example patient reports (out-of-fold, real trials from the dataset)", "",
             "Each report comes from a model that never saw that person. Risk is the Random Forest's out-of-fold "
             "probability of PD for that single walking trial. **Research screening estimate, not a diagnosis.**", ""]
    for stage, i in examples:
        contrib = pd.DataFrame({"name": names, "shap": sv_all[i], "value": Ximp[i], "healthy_median": healthy_median}).reindex(np.argsort(-np.abs(sv_all[i]))[:8])
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        c = contrib.iloc[::-1]
        ax.barh(c.name, c.shap, color=["#c0504d" if s > 0 else "#2a9d8f" for s in c.shap])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Contribution to PD probability (red raises, green lowers)")
        ax.set_title(f"{df.loc[i, 'subject_id']} ({stage}): estimated PD probability {prob[i]:.0%}")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"shap_patient_{stage.lower()}.png", dpi=200); plt.close(fig)
        up = [r for r in contrib.itertuples() if r.shap > 0][:3]
        down = [r for r in contrib.itertuples() if r.shap < 0][:2]
        lines += [f"## {df.loc[i, 'subject_id']} - recorded stage: {stage}", "",
                  f"- Estimated chance of Parkinson's disease: **{prob[i]:.0%}** (model baseline {base_all[i]:.0%})",
                  "- Raising the estimate: " + ("; ".join(f"{r.name} ({r.value:.3g} vs healthy median {r.healthy_median:.3g})" for r in up) or "none"),
                  "- Lowering the estimate: " + ("; ".join(f"{r.name} ({r.value:.3g} vs healthy median {r.healthy_median:.3g})" for r in down) or "none"), ""]
    OUT_REPORT.write_text("\n".join(lines))
    print(f"\nWrote {OUT_REPORT} and figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
