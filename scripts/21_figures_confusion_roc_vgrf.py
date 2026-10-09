"""
21_figures_confusion_roc_vgrf.py
Remaining figures for the paper (everything except leakage-gap and SHAP, which already exist):
  A. Confusion matrices -- out-of-fold, subject-grouped CV, Random Forest: gait 4-class,
     gait binary, voice binary.
  B. ROC curves -- out-of-fold probabilities, subject-grouped CV: gait binary, voice binary,
     on one shared axis for comparison.
  C. Raw VGRF signal -- one healthy and one PD example trial, total force per foot over time.
  D. Centre-of-pressure (COP) trajectory -- same two example trials, per-foot COP path computed
     from the sensors' documented X-Y positions (same geometry as 03_extract_features_v2.py).

All out-of-fold predictions use StratifiedGroupKFold (10 folds, seed 42), so no subject's own
data is used to predict its own label.

Outputs (docs/figures/):
  confusion_gait_4class.png, confusion_gait_binary.png, confusion_voice_binary.png
  roc_binary_gait_voice.png
  vgrf_example_healthy_vs_pd.png
  cop_trajectory_example.png

Run from project root:
    python scripts/21_figures_confusion_roc_vgrf.py
"""

import importlib.util
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

FIG_DIR = Path("docs/figures")
SEED = 42
N_FOLDS = 10
STAGE_ORDER = ["Healthy", "Mild", "Moderate", "Severe"]

SENSOR_POSITIONS = {  # mirrors 03_extract_features_v2.py (Table 2, Balaji et al. 2020)
    "L1": (-500, -800), "L2": (-700, -400), "L3": (-300, -400), "L4": (-700, 0),
    "L5": (-300, 0), "L6": (-700, 400), "L7": (-300, 400), "L8": (-500, 800),
    "R1": (500, -800), "R2": (700, -400), "R3": (300, -400), "R4": (700, 0),
    "R5": (300, 0), "R6": (700, 400), "R7": (300, 400), "R8": (500, 800),
}


def m07():
    spec = importlib.util.spec_from_file_location("m07", Path(__file__).with_name("07_train_multi_scheme.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def out_of_fold(X, y, groups, n_folds=N_FOLDS, seed=SEED):
    """Out-of-fold predictions + probabilities from a Random Forest, subject-grouped CV."""
    pred = np.zeros(len(y), dtype=int)
    prob = None
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr, y[tr])
        pred[te] = rf.predict(Xte)
        if prob is None and len(np.unique(y)) == 2:
            prob = np.zeros(len(y))
        if prob is not None:
            prob[te] = rf.predict_proba(Xte)[:, 1]
    return pred, prob


# -------------------------------------------------------------- A/B: confusion + ROC
def gait_data():
    mod = m07()
    df = mod.load_merged()
    X = mod.build_feature_matrix(df).to_numpy(float)
    groups = df["subject_id"].to_numpy()
    return df, X, groups


def confusion_matrix_figure(y_true, y_pred, labels, title, out_name):
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=range(len(labels))), display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=200)
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / out_name}")


def confusion_and_roc():
    df, X_gait, groups_gait = gait_data()
    mod = m07()

    # 4-class
    y4_raw = mod.get_label_scheme(df, "4class")
    le4 = LabelEncoder().fit(y4_raw)
    y4 = le4.transform(y4_raw)
    pred4, _ = out_of_fold(X_gait, y4, groups_gait)
    acc4 = np.mean(pred4 == y4)
    print(f"Gait 4-class out-of-fold accuracy: {acc4:.3f}")
    confusion_matrix_figure(y4, pred4, list(le4.classes_), f"Gait 4-class confusion matrix (out-of-fold, acc={acc4:.1%})", "confusion_gait_4class.png")

    # binary gait
    yb_raw = mod.get_label_scheme(df, "binary")
    leb = LabelEncoder().fit(yb_raw)
    yb = leb.transform(yb_raw)
    predb, probb_gait = out_of_fold(X_gait, yb, groups_gait)
    accb = np.mean(predb == yb)
    print(f"Gait binary out-of-fold accuracy: {accb:.3f}")
    confusion_matrix_figure(yb, predb, list(leb.classes_), f"Gait binary confusion matrix (out-of-fold, acc={accb:.1%})", "confusion_gait_binary.png")

    # voice binary
    voice_df = pd.read_csv("data/processed/voice_features.csv")
    voice_df = voice_df[voice_df.partition == "train"].reset_index(drop=True)
    voice_cols = [c for c in voice_df.columns if c not in ("subject_id", "subject_num", "partition", "label", "class", "updrs")]
    X_voice = voice_df[voice_cols].to_numpy(float)
    y_voice = LabelEncoder().fit_transform(voice_df["label"])  # Healthy=0, PD=1
    groups_voice = voice_df["subject_id"].to_numpy()
    predv, probv = out_of_fold(X_voice, y_voice, groups_voice)
    accv = np.mean(predv == y_voice)
    print(f"Voice binary out-of-fold accuracy: {accv:.3f}")
    confusion_matrix_figure(y_voice, predv, ["Healthy", "PD"], f"Voice confusion matrix (out-of-fold, acc={accv:.1%})", "confusion_voice_binary.png")

    # ROC: gait binary + voice binary on one axis
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for prob, y_true, label, color in ((probb_gait, yb, "Gait (binary)", "#1f3a5f"), (probv, y_voice, "Voice (binary)", "#8e6bb0")):
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC: out-of-fold, subject-grouped CV")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_binary_gait_voice.png", dpi=200)
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'roc_binary_gait_voice.png'}")


# -------------------------------------------------------------- C/D: VGRF + COP
def pick_example_trials():
    labels = pd.read_csv("data/processed/subject_labels.csv")
    healthy_subj = labels[labels.stage == "Healthy"].subject_id.iloc[0]
    pd_subj = labels[labels.stage == "Mild"].subject_id.iloc[0]  # mild PD: clearest, least noisy contrast
    examples = {}
    for name, subj in (("Healthy", healthy_subj), ("Mild PD", pd_subj)):
        f = next(Path("data/interim").rglob(f"{subj}_*.txt"))
        examples[name] = pd.read_csv(f)
    return examples


def vgrf_and_cop_figures():
    examples = pick_example_trials()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, (name, df) in zip(axes, examples.items()):
        t = df["Time"] - df["Time"].iloc[0]
        ax.plot(t, df["TotalL"], color="#1f3a5f", lw=0.7, label="Left foot total force")
        ax.plot(t, df["TotalR"], color="#c0504d", lw=0.7, label="Right foot total force")
        ax.set_ylabel("Force (N)")
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 15)  # first 15s: a few gait cycles, readable
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Example VGRF signal: healthy vs mild-PD gait (first 15s of one trial each)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vgrf_example_healthy_vs_pd.png", dpi=200)
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'vgrf_example_healthy_vs_pd.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, (name, df) in zip(axes, examples.items()):
        for side, color in (("L", "#1f3a5f"), ("R", "#c0504d")):
            sensors = [f"{side}{i}" for i in range(1, 9)]
            forces = df[sensors].to_numpy()
            xs = np.array([SENSOR_POSITIONS[s][0] for s in sensors])
            ys = np.array([SENSOR_POSITIONS[s][1] for s in sensors])
            total = forces.sum(axis=1)
            valid = total > 15
            cop_x = np.divide((forces[valid] * xs).sum(axis=1), total[valid], out=np.zeros(valid.sum()), where=total[valid] != 0)
            cop_y = np.divide((forces[valid] * ys).sum(axis=1), total[valid], out=np.zeros(valid.sum()), where=total[valid] != 0)
            ax.plot(cop_x[:600], cop_y[:600], color=color, lw=0.5, alpha=0.8, label=f"{side} foot")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Medial-lateral (mm)"); ax.set_ylabel("Anterior-posterior (mm)")
        ax.set_aspect("equal")
    axes[0].legend(fontsize=8)
    fig.suptitle("Centre-of-pressure trajectory: healthy vs mild-PD (first ~6s of loaded stance)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cop_trajectory_example.png", dpi=200)
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'cop_trajectory_example.png'}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    confusion_and_roc()
    vgrf_and_cop_figures()


if __name__ == "__main__":
    main()
