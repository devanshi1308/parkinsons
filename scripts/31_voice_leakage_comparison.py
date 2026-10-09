"""
31_voice_leakage_comparison.py
Repeats the gait study's central experiment (scripts/14_leakage_comparison.py) on a second,
independent modality: the UCI voice dataset. Same question -- does trial/recording-level CV
(ignoring that one subject has many recordings) inflate accuracy relative to subject-grouped CV?

Uses the 40-subject TRAIN partition only (20 healthy, 20 PD, 26 recordings each) -- the test
partition has no healthy controls and cannot be used to measure accuracy (see 30_voice_load.py).

Protocols (identical structure to the gait script):
  leaky     StratifiedKFold on recordings (ignores subject_id) -- a subject's 26 recordings can
            land in both train and test folds.
  grouped   StratifiedGroupKFold by subject_id -- no subject in both train and test.
5 seeds, 4 classifiers (same families as the gait study: DT, SVM, Bagging, GaussianNB), plus a
leak audit (share of leaky-fold test recordings whose subject is also in that fold's training set).

Outputs:
  data/processed/voice_leakage_comparison.csv
  data/processed/voice_leakage_audit.csv
  docs/figures/voice_leakage_gap.png

Run from project root:
    python scripts/31_voice_leakage_comparison.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

FEATURES_PATH = Path("data/processed/voice_features.csv")
OUT_CMP = Path("data/processed/voice_leakage_comparison.csv")
OUT_AUDIT = Path("data/processed/voice_leakage_audit.csv")
OUT_FIG = Path("docs/figures/voice_leakage_gap.png")

BASE_SEED = 42
N_REPEATS = 5
N_FOLDS = 10


def classifiers(seed):
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "SVM": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", random_state=seed))]),
        "Ensemble (Bagging)": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=seed), n_estimators=30, random_state=seed),
        "Bayes (GaussianNB)": GaussianNB(),
    }


def make_splitter(protocol, n_splits, seed):
    if protocol == "leaky":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validate(clf, X, y, groups, splitter):
    accs, f1s = [], []
    for tr, te in splitter.split(X, y, groups):
        m = clone(clf).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        accs.append(np.mean(pred == y[te]))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return accs, f1s


def audit_leaky_folds(y, groups, n_splits, seed):
    rows = []
    splitter = make_splitter("leaky", n_splits, seed)
    for fold, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y)):
        train_subj = set(groups[tr])
        shared = np.array([g in train_subj for g in groups[te]])
        rows.append({"seed": seed, "fold": fold, "n_test_recordings": len(te),
                     "pct_test_recordings_subject_in_train": 100.0 * shared.mean()})
    return rows


def main():
    df = pd.read_csv(FEATURES_PATH)
    df = df[df.partition == "train"].reset_index(drop=True)  # only the labeled, balanced split
    feature_cols = [c for c in df.columns if c not in ("subject_id", "subject_num", "partition", "label", "class", "updrs")]
    X = df[feature_cols].to_numpy(float)
    y = LabelEncoder().fit_transform(df["label"])
    groups = df["subject_id"].to_numpy()
    print(f"Recordings: {len(df)}  Subjects: {df.subject_id.nunique()}  Features: {X.shape[1]}")

    rows, audit_rows = [], []
    for repeat in range(N_REPEATS):
        seed = BASE_SEED + repeat
        audit_rows.extend(audit_leaky_folds(y, groups, N_FOLDS, seed))
        for name, clf in classifiers(seed).items():
            for protocol in ("leaky", "grouped"):
                splitter = make_splitter(protocol, N_FOLDS, seed)
                accs, f1s = cross_validate(clf, X, y, groups, splitter)
                rows.append({"classifier": name, "protocol": protocol, "seed": seed,
                             "accuracy": np.mean(accs), "f1_macro": np.mean(f1s)})
        print(f"  repeat {repeat + 1}/{N_REPEATS} done")

    runs = pd.DataFrame(rows)
    summary = (runs.groupby(["classifier", "protocol"])
               .agg(accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                    f1_macro_mean=("f1_macro", "mean"), f1_macro_sd=("f1_macro", "std"))
               .reset_index())
    wide = summary.pivot_table(index="classifier", columns="protocol", values=["accuracy_mean", "f1_macro_mean"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide["accuracy_gap_pts"] = 100 * (wide["accuracy_mean_leaky"] - wide["accuracy_mean_grouped"])
    result = summary.merge(wide[["accuracy_gap_pts"]], on="classifier")

    OUT_CMP.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CMP, index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_AUDIT, index=False)
    print(f"\nSaved -> {OUT_CMP}\nSaved -> {OUT_AUDIT}")

    print("\n=== Accuracy (mean over 5 seeds): leaky vs grouped ===")
    show = result.pivot_table(index="classifier", columns="protocol", values="accuracy_mean")
    show["gap_pts"] = 100 * (show["leaky"] - show["grouped"])
    print(show.round(3).to_string())
    print(f"\nLeak audit: on average {audit.pct_test_recordings_subject_in_train.mean():.1f}% of leaky-fold "
          f"test recordings belong to a subject also in that fold's training set.")

    classifier_order = list(classifiers(BASE_SEED))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(classifier_order))
    width = 0.38
    for offset, protocol, color in ((-width / 2, "leaky", "#c0504d"), (width / 2, "grouped", "#4f81bd")):
        sub = result[result.protocol == protocol].set_index("classifier")
        means = [sub.loc[c, "accuracy_mean"] for c in classifier_order]
        errs = [sub.loc[c, "accuracy_sd"] for c in classifier_order]
        ax.bar(x + offset, means, width, yerr=errs, label=protocol, color=color, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([c.split(" (")[0] for c in classifier_order], rotation=15, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy (mean over 5 seeds)")
    ax.set_title("Voice (UCI #301): recording-level (leaky) vs subject-grouped CV")
    ax.legend(title="CV protocol")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    print(f"Saved -> {OUT_FIG}")


if __name__ == "__main__":
    main()
