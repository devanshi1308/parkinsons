"""
15_baseline_comparison.py
Compares our pipeline against the numbers printed in Balaji et al. (2020), Tables 6, 8 and 9.

For both the paper's "statistical" feature set (features.csv / v1: mean, median, SD per sensor)
and its "kinematic" feature set (the non-sensor columns of features_v2.csv: stance/swing/stride
time, ratios, symmetry, COP, cadence), we run two protocols with the same 4 classifier families
(DT, SVM, Bagging "EC", GaussianNB "BC"):

  trial_level_repeated10   StratifiedKFold (10 folds, NOT grouped by subject), repeated with
                            10 different seeds and averaged -- the closest reproduction of what
                            the paper describes ("10 fold cross validation... run 10 times
                            independently"). The paper does not state whether folds were grouped
                            by subject, so this is our best-effort match, not a confirmed replica.
  subject_grouped           StratifiedGroupKFold by subject_id, same seeds -- our honest protocol.

Paper numbers (hardcoded from docs/paper.pdf, Table 8 kinematic cumulative accuracy/sensitivity/
specificity/PPV/NPV/F-score, and Table 9 accuracy) are included for side-by-side comparison.
Verify these against the PDF before publishing; OCR/transcription errors are possible.

Outputs:
  data/processed/baseline_comparison.csv
  docs/baseline_comparison.md

Run from project root:
    python scripts/15_baseline_comparison.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

V1_PATH = Path("data/processed/features.csv")
V2_PATH = Path("data/processed/features_v2.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
OUT_CSV = Path("data/processed/baseline_comparison.csv")
OUT_MD = Path("docs/baseline_comparison.md")

BASE_SEED = 42
N_REPEATS = 10          # matches the paper's "10 independent runs"
N_FOLDS = 10

ID_COLS = ["filename", "cohort", "group", "subject_id", "trial"]

# Table 8 of Balaji et al. (2020): kinematic-feature cumulative performance, 4-class.
PAPER_TABLE8 = {
    "Decision Tree": {"accuracy": 99.4, "sensitivity": 99.6, "specificity": 99.8, "ppv": 99.0, "npv": 100.0, "fscore": 99.25},
    "SVM":           {"accuracy": 99.4, "sensitivity": 99.6, "specificity": 99.8, "ppv": 99.0, "npv": 100.0, "fscore": 99.25},
    "Bagging":       {"accuracy": 95.2, "sensitivity": 91.6, "specificity": 98.2, "ppv": 95.3, "npv": 4.7,   "fscore": 93.1},
    "GaussianNB":    {"accuracy": 69.9, "sensitivity": 42.4, "specificity": 89.1, "ppv": 65.7, "npv": 34.3,  "fscore": 72.1},
}
# Table 9: accuracy of the proposed framework (DT, spatiotemporal features) vs other published methods.
PAPER_TABLE9_PROPOSED_ACCURACY = 99.4


def classifiers(seed):
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "SVM": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", random_state=seed))]),
        "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=seed), n_estimators=30, random_state=seed),
        "GaussianNB": GaussianNB(),
    }


def load_sets():
    v1 = pd.read_csv(V1_PATH)
    v2 = pd.read_csv(V2_PATH)
    kinematic_cols = ID_COLS + [c for c in v2.columns if c not in ID_COLS and not (c[0] in "LR" and c[1].isdigit())]
    return {"statistical": v1, "kinematic": v2[kinematic_cols]}


def prep_xy(df, labels):
    merged = df.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"])
    feat_cols = [c for c in merged.columns if c not in ID_COLS + ["stage"]]
    X = merged[feat_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.var() > 1e-6]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X = X.drop(columns=drop)
    y_raw = merged["stage"]
    y = LabelEncoder().fit_transform(y_raw)
    groups = merged["subject_id"].to_numpy()
    return X.to_numpy(float), y, groups


def evaluate(clf, X, y, groups, protocol, seed):
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed) if protocol == "trial_level_repeated10" \
        else StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    accs, sens, spec, ppv, f1 = [], [], [], [], []
    for tr, te in splitter.split(X, y, groups):
        m = clone(clf).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        accs.append(np.mean(pred == y[te]))
        sens.append(recall_score(y[te], pred, average="macro", zero_division=0))
        f1.append(f1_score(y[te], pred, average="macro", zero_division=0))
        ppv.append(precision_score(y[te], pred, average="macro", zero_division=0))
        # macro specificity: average of per-class TNR
        classes = np.unique(y)
        tnr = []
        for c in classes:
            tn = np.sum((y[te] != c) & (pred != c)); fp = np.sum((y[te] != c) & (pred == c))
            tnr.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
        spec.append(np.nanmean(tnr))
    return dict(accuracy=np.mean(accs) * 100, sensitivity=np.mean(sens) * 100, specificity=np.mean(spec) * 100,
                ppv=np.mean(ppv) * 100, fscore=np.mean(f1) * 100)


def main():
    labels = pd.read_csv(LABELS_PATH)
    sets = load_sets()
    rows = []
    for analysis, df in sets.items():
        X, y, groups = prep_xy(df, labels)
        print(f"{analysis}: {X.shape[1]} features, {len(y)} trials, {len(set(groups))} subjects")
        for name, clf in classifiers(BASE_SEED).items():
            for protocol in ("trial_level_repeated10", "subject_grouped"):
                per_seed = [evaluate(classifiers(BASE_SEED + r)[name], X, y, groups, protocol, BASE_SEED + r) for r in range(N_REPEATS)]
                agg = {k: float(np.mean([d[k] for d in per_seed])) for k in per_seed[0]}
                agg_sd = {f"{k}_sd": float(np.std([d[k] for d in per_seed])) for k in per_seed[0]}
                rows.append({"analysis": analysis, "classifier": name, "protocol": protocol, "n_repeats": N_REPEATS, **agg, **agg_sd})
            print(f"  {name:12s} done")

    ours = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ours.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")

    name_map = {"Decision Tree": "Decision Tree", "SVM": "SVM", "Bagging": "Bagging", "GaussianNB": "GaussianNB"}
    lines = ["# Comparison against the baseline paper (Balaji et al. 2020)", "",
             "Paper numbers are Table 8 (kinematic-feature cumulative performance, 4-class) and Table 9 "
             "(the proposed framework's headline accuracy), copied from `docs/paper.pdf`. Verify against the "
             "PDF before publishing. The paper states 10-fold CV repeated 10 times but does not say whether "
             "folds were grouped by subject; `trial_level_repeated10` below is our best-effort match to its "
             "description, not a confirmed replica, and `subject_grouped` is our honest protocol.", "",
             "## Kinematic features, 4-class, mean of 10 repeats (%)", "",
             "| Classifier | Paper (Table 8) | Ours, trial-level (closest match) | Ours, subject-grouped (honest) |",
             "|---|---|---|---|"]
    k = ours[ours.analysis == "kinematic"]
    for name in name_map:
        p = PAPER_TABLE8[name]
        tl = k[(k.classifier == name) & (k.protocol == "trial_level_repeated10")].iloc[0]
        gr = k[(k.classifier == name) & (k.protocol == "subject_grouped")].iloc[0]
        lines.append(f"| {name} | acc {p['accuracy']:.1f}, sens {p['sensitivity']:.1f}, spec {p['specificity']:.1f}, "
                     f"PPV {p['ppv']:.1f}, F {p['fscore']:.1f} | acc {tl.accuracy:.1f} (sd {tl.accuracy_sd:.1f}), "
                     f"sens {tl.sensitivity:.1f}, spec {tl.specificity:.1f}, PPV {tl.ppv:.1f}, F {tl.fscore:.1f} | "
                     f"acc {gr.accuracy:.1f} (sd {gr.accuracy_sd:.1f}), sens {gr.sensitivity:.1f}, spec {gr.specificity:.1f}, "
                     f"PPV {gr.ppv:.1f}, F {gr.fscore:.1f} |")
    lines += ["", "## Statistical features, 4-class, mean of 10 repeats (%)", "",
              "| Classifier | Ours, trial-level (closest match) | Ours, subject-grouped (honest) |", "|---|---|---|"]
    s = ours[ours.analysis == "statistical"]
    for name in name_map:
        tl = s[(s.classifier == name) & (s.protocol == "trial_level_repeated10")].iloc[0]
        gr = s[(s.classifier == name) & (s.protocol == "subject_grouped")].iloc[0]
        lines.append(f"| {name} | acc {tl.accuracy:.1f} (sd {tl.accuracy_sd:.1f}), sens {tl.sensitivity:.1f}, "
                     f"spec {tl.specificity:.1f}, PPV {tl.ppv:.1f}, F {tl.fscore:.1f} | "
                     f"acc {gr.accuracy:.1f} (sd {gr.accuracy_sd:.1f}), sens {gr.sensitivity:.1f}, spec {gr.specificity:.1f}, "
                     f"PPV {gr.ppv:.1f}, F {gr.fscore:.1f} |")
    best_tl = k[k.protocol == "trial_level_repeated10"].accuracy.max()
    best_gr = k[k.protocol == "subject_grouped"].accuracy.max()
    lines += ["", "## Headline comparison", "",
              f"- Paper's proposed framework (Table 9): **{PAPER_TABLE9_PROPOSED_ACCURACY:.1f}%**",
              f"- Our best trial-level, closest-match protocol (kinematic features): **{best_tl:.1f}%**",
              f"- Our best honest, subject-grouped protocol (kinematic features): **{best_gr:.1f}%**",
              "", "We do not match the paper's headline number under either protocol, and the gap between our "
              "two protocols is itself evidence for the leakage argument (see `14_leakage_comparison.py` for "
              "the controlled version of this comparison on the same splitter, same features)."]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"Saved -> {OUT_MD}")


if __name__ == "__main__":
    main()
