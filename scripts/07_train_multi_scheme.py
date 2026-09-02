"""
07_train_multi_scheme.py
Evaluates the same feature set + classifiers under three label schemes:
  1. 4-class:  Healthy / Mild / Moderate / Severe   (original H&Y-derived staging)
  2. 3-class:  Healthy / Mild / Moderate+Severe      (merges the two smallest,
               most-confused classes — still clinically meaningful, more samples
               per class)
  3. Binary:   Healthy / PD                          (standard framing used by
               most comparison methods in the literature, e.g. Table 9 of
               Balaji et al. 2020)

All three use the SAME subject-grouped 10-fold CV protocol as 05_train_classifiers.py,
so results are directly comparable and equally rigorous (no leakage in any scheme).

Run from project root:
    python scripts/07_train_multi_scheme.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

RANDOM_STATE = 42
FEATURES_PATH = Path("data/processed/features_v2.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
OUT_PATH = Path("data/processed/classifier_results_all_schemes.csv")


def load_merged() -> pd.DataFrame:
    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    merged = features.merge(labels[["subject_id", "stage"]], on="subject_id", how="left")
    merged = merged.dropna(subset=["stage"])
    return merged


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("filename", "cohort", "group", "subject_id", "trial", "stage")]
    X = df[feature_cols].dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))
    variances = X.var()
    X = X.loc[:, variances > 1e-6]

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    if to_drop:
        X = X.drop(columns=to_drop)
    return X


def get_label_scheme(df: pd.DataFrame, scheme: str) -> pd.Series:
    if scheme == "4class":
        return df["stage"]
    elif scheme == "3class":
        return df["stage"].replace({"Severe": "Moderate+Severe", "Moderate": "Moderate+Severe"})
    elif scheme == "binary":
        return df["stage"].apply(lambda s: "Healthy" if s == "Healthy" else "PD")
    else:
        raise ValueError(scheme)


def run_scheme(X: pd.DataFrame, y_raw: pd.Series, groups: pd.Series, scheme_name: str) -> list:
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"\n{'='*60}\nScheme: {scheme_name}  |  Classes: {list(le.classes_)}")
    print(y_raw.value_counts().to_string())

    n_splits = min(10, y_raw.value_counts().min())  # can't have more folds than smallest class
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", random_state=RANDOM_STATE))]),
        "Ensemble (Bagging)": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=30, random_state=RANDOM_STATE,
        ),
        "Bayes (GaussianNB)": GaussianNB(),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }

    rows = []
    for name, clf in classifiers.items():
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, groups=groups)
        row = {"scheme": scheme_name, "classifier": name, "n_folds": n_splits}
        for metric in scoring:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
        print(f"  {name:22s} acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}  "
              f"f1={row['f1_macro_mean']:.3f}±{row['f1_macro_std']:.3f}")

    return rows


def main():
    df = load_merged()
    X = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape}")
    groups = df["subject_id"]

    all_results = []
    for scheme in ["4class", "3class", "binary"]:
        y_raw = get_label_scheme(df, scheme)
        all_results.extend(run_scheme(X, y_raw, groups, scheme))

    results_df = pd.DataFrame(all_results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\n\nSaved all results -> {OUT_PATH.resolve()}")

    print("\n=== Summary: best accuracy per scheme ===")
    for scheme in ["4class", "3class", "binary"]:
        subset = results_df[results_df["scheme"] == scheme]
        best = subset.loc[subset["accuracy_mean"].idxmax()]
        print(f"  {scheme:8s}: {best['classifier']:22s} {best['accuracy_mean']:.3f} ± {best['accuracy_std']:.3f}")


if __name__ == "__main__":
    main()
