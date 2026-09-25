"""
09_class_imbalance.py
Tests two standard fixes for the Severe-class imbalance problem:
  1. class_weight="balanced" — free, built into sklearn, penalizes misclassifying
     minority class more during training. No risk of leakage.
  2. SMOTE oversampling — synthesizes new minority-class samples. Applied ONLY to
     the training fold in each CV split (never to test folds) to avoid leakage —
     this is critical and a common mistake in papers that misuse SMOTE.

Compares against the plain (no rebalancing) results from 07/08 as baseline.

Requires: pip install imbalanced-learn

Run from project root:
    python scripts/09_class_imbalance.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise SystemExit("Run: pip install imbalanced-learn")

RANDOM_STATE = 42
FEATURES_PATH = Path("data/processed/features_v2.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
OUT_PATH = Path("data/processed/class_imbalance_results.csv")


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


def run_with_strategy(X, y, groups, le, strategy: str) -> list:
    """strategy: 'none', 'class_weight', or 'smote'"""
    n_splits = min(10, pd.Series(y).value_counts().min())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    def make_classifiers():
        cw = "balanced" if strategy == "class_weight" else None
        return {
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=cw),
            "SVM": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", random_state=RANDOM_STATE, class_weight=cw))]),
            "Ensemble (Bagging)": BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=cw),
                n_estimators=30, random_state=RANDOM_STATE,
            ),
            "Bayes (GaussianNB)": GaussianNB(),  # doesn't support class_weight/sampling directly
        }

    rows = []
    classifiers = make_classifiers()

    for clf_name, clf in classifiers.items():
        pooled_true, pooled_pred = [], []

        for train_idx, test_idx in cv.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx]
            y_train, y_test = y[train_idx].copy(), y[test_idx]

            if strategy == "smote":
                # Only oversample the TRAINING fold — test fold stays untouched.
                min_class_count = pd.Series(y_train).value_counts().min()
                k_neighbors = min(5, max(1, min_class_count - 1))
                if min_class_count > 1:
                    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
                    X_train, y_train = smote.fit_resample(X_train, y_train)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pooled_true.extend(y_test)
            pooled_pred.extend(y_pred)

        pooled_true, pooled_pred = np.array(pooled_true), np.array(pooled_pred)
        acc = accuracy_score(pooled_true, pooled_pred)
        f1_macro = f1_score(pooled_true, pooled_pred, average="macro", zero_division=0)
        f1_per_class = f1_score(pooled_true, pooled_pred, average=None, zero_division=0)

        row = {"strategy": strategy, "classifier": clf_name, "accuracy": acc, "f1_macro": f1_macro}
        for i, class_name in enumerate(le.classes_):
            row[f"f1_{class_name}"] = f1_per_class[i] if i < len(f1_per_class) else np.nan
        rows.append(row)

        print(f"  [{strategy:12s}] {clf_name:22s} acc={acc:.3f}  f1_macro={f1_macro:.3f}  "
              f"per-class F1: {dict(zip(le.classes_, np.round(f1_per_class, 2)))}")

    return rows


def main():
    df = load_merged()
    X = build_feature_matrix(df)
    groups = df["subject_id"]

    le = LabelEncoder()
    y = le.fit_transform(df["stage"])  # 4-class scheme, where Severe imbalance is worst
    print(f"Feature matrix: {X.shape}")
    print(f"Class counts: {df['stage'].value_counts().to_dict()}\n")

    all_rows = []
    for strategy in ["none", "class_weight", "smote"]:
        print(f"\n{'='*70}\nStrategy: {strategy}")
        all_rows.extend(run_with_strategy(X, y, groups, le, strategy))

    report = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_PATH, index=False)
    print(f"\n\nSaved -> {OUT_PATH.resolve()}")

    print("\n=== Severe-class F1 across strategies (the class we're trying to rescue) ===")
    print(report.pivot_table(index="classifier", columns="strategy", values="f1_Severe"))


if __name__ == "__main__":
    main()
