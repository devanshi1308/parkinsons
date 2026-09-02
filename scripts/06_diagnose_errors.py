"""
06_diagnose_errors.py
Runs ONE subject-grouped train/test split (not full CV) so we can inspect
exactly what the classifier gets wrong: confusion matrix + per-class report.
This is diagnostic only — final reported numbers should come from full CV
in 05_train_classifiers.py, not this script.

Run from project root:
    python scripts/06_diagnose_errors.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_STATE = 42
FEATURES_PATH = Path("data/processed/features_v2.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")


def main():
    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    df = features.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"])

    feature_cols = [c for c in df.columns if c not in ("filename", "cohort", "group", "subject_id", "trial", "stage")]
    X = df[feature_cols].dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))
    variances = X.var()
    X = X.loc[:, variances > 1e-6]

    le = LabelEncoder()
    y = le.fit_transform(df["stage"])
    groups = df["subject_id"]

    # Take just the first fold of a subject-grouped split for inspection
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(cv.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=30, random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== Confusion Matrix (rows=true, cols=predicted) ===")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)

    print("\n=== Per-class report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("=== Test set class counts (this fold) ===")
    print(pd.Series(le.inverse_transform(y_test)).value_counts())

    # Also check: how many unique subjects in train vs test (sanity check no leakage)
    print(f"\nTrain subjects: {df.iloc[train_idx]['subject_id'].nunique()}")
    print(f"Test subjects: {df.iloc[test_idx]['subject_id'].nunique()}")
    overlap = set(df.iloc[train_idx]["subject_id"]) & set(df.iloc[test_idx]["subject_id"])
    print(f"Overlapping subjects between train/test: {len(overlap)} (should be 0)")


if __name__ == "__main__":
    main()
