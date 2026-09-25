"""
10_per_cohort_analysis.py
Checks whether classifier performance varies meaningfully across the three
PhysioNet sub-cohorts (Ga = Galit study, Ju = Yogev study, Si = Silvia/Frenkel
study) — they were collected by different sub-studies at Tel-Aviv, possibly
with different protocols/equipment calibration. If one cohort drives most of
the error, that's a real data-heterogeneity finding worth reporting, not just
noise averaged away in the pooled results.

Two things tested:
  1. Per-cohort test performance: using the SAME pooled model (all cohorts pooled
     in training, as in 08/09), how does accuracy break down when we look ONLY
     at test predictions for each cohort separately?
  2. Cross-cohort transfer: train on 2 cohorts, test on the 3rd (leave-one-
     cohort-out) — a stricter generalization test than random subject-grouped CV.

Run from project root:
    python scripts/10_per_cohort_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
FEATURES_PATH = Path("data/processed/features_v2.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
OUT_PATH = Path("data/processed/per_cohort_results.csv")


def load_merged() -> pd.DataFrame:
    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    return features.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"])


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("filename", "cohort", "group", "subject_id", "trial", "stage")]
    X = df[feature_cols].dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.var() > 1e-6]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    return X.drop(columns=to_drop) if to_drop else X


def part1_pooled_model_per_cohort_breakdown(df, X, y, groups):
    """Same pooled subject-grouped CV as before, but break down test predictions by cohort."""
    print("=== Part 1: Pooled model, test accuracy broken down by cohort ===\n")
    n_splits = min(10, pd.Series(y).value_counts().min())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        n_estimators=30, random_state=RANDOM_STATE,
    )

    cohort_true, cohort_pred, cohort_labels = [], [], []
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        cohort_true.extend(y_test)
        cohort_pred.extend(y_pred)
        cohort_labels.extend(df.iloc[test_idx]["cohort"].values)

    result_df = pd.DataFrame({"cohort": cohort_labels, "true": cohort_true, "pred": cohort_pred})
    rows = []
    for cohort in sorted(result_df["cohort"].unique()):
        subset = result_df[result_df["cohort"] == cohort]
        acc = accuracy_score(subset["true"], subset["pred"])
        f1 = f1_score(subset["true"], subset["pred"], average="macro", zero_division=0)
        n = len(subset)
        rows.append({"cohort": cohort, "n_test_trials": n, "accuracy": acc, "f1_macro": f1})
        print(f"  {cohort}: n={n:4d}  accuracy={acc:.3f}  f1_macro={f1:.3f}")

    return pd.DataFrame(rows)


def part2_leave_one_cohort_out(df, X, y_raw, le):
    """Stricter test: train on 2 cohorts, test on the 3rd entirely unseen cohort."""
    print("\n=== Part 2: Leave-one-cohort-out (stricter generalization test) ===\n")
    cohorts = sorted(df["cohort"].unique())
    rows = []

    for held_out in cohorts:
        train_mask = df["cohort"] != held_out
        test_mask = df["cohort"] == held_out

        X_train, X_test = X[train_mask.values], X[test_mask.values]
        y_train, y_test = y_raw[train_mask.values], y_raw[test_mask.values]

        clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            n_estimators=30, random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        rows.append({
            "held_out_cohort": held_out, "n_train": len(X_train), "n_test": len(X_test),
            "accuracy": acc, "f1_macro": f1,
        })
        print(f"  Held out {held_out} (train on others, n_train={len(X_train)}, n_test={len(X_test)}): "
              f"accuracy={acc:.3f}  f1_macro={f1:.3f}")

    return pd.DataFrame(rows)


def main():
    df = load_merged()
    X = build_feature_matrix(df)
    groups = df["subject_id"]

    le = LabelEncoder()
    y = le.fit_transform(df["stage"])
    print(f"Feature matrix: {X.shape}")
    print(f"Cohort sizes: {df['cohort'].value_counts().to_dict()}\n")

    part1 = part1_pooled_model_per_cohort_breakdown(df, X, y, groups)
    part2 = part2_leave_one_cohort_out(df, X, df["stage"], le)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("=== Part 1: pooled model, per-cohort test breakdown ===\n")
        part1.to_csv(f, index=False)
        f.write("\n=== Part 2: leave-one-cohort-out ===\n")
        part2.to_csv(f, index=False)

    print(f"\n\nSaved -> {OUT_PATH.resolve()}")
    print("\nInterpretation guide:")
    print("- If Part 1 shows similar accuracy across cohorts: pooling is safe, no heterogeneity issue.")
    print("- If Part 1 shows one cohort much worse: that cohort may have different noise/protocol.")
    print("- If Part 2 accuracy is much lower than pooled CV: model doesn't generalize well across")
    print("  cohorts (e.g. different equipment/calibration) — a real limitation worth reporting.")


if __name__ == "__main__":
    main()
