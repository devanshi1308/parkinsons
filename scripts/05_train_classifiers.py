"""
05_train_classifiers.py
Merges features.csv + subject_labels.csv, then trains and evaluates
Decision Tree, SVM, Ensemble (Random Forest/Bagging), and Naive Bayes
classifiers using 10-fold stratified cross-validation.

Requires: pip install pandas numpy scikit-learn

Run from project root:
    python scripts/05_train_classifiers.py
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
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_STATE = 42

FEATURES_PATH = Path("data/processed/features_v2.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
OUT_DIR = Path("data/processed")


def build_subject_id(row) -> str:
    """Recreate GaCo01-style ID from demographics Study/Group/Subjnum columns."""
    study = str(row["Study"]).strip()
    group = str(row["Group"]).strip()
    num = str(row["Subjnum"]).strip()
    num = num.zfill(2) if num.isdigit() else num
    return f"{study}{group}{num}"


def load_merged() -> pd.DataFrame:
    features = pd.read_csv(FEATURES_PATH)
    labels_raw = pd.read_csv(LABELS_PATH)

    # subject_labels.csv only has subject_id, HoehnYahr-col, stage (from 04_merge_labels.py)
    # If the ID column didn't already match GaCo01 format, reconstruct it here.
    # Adjust this block if your subject_labels.csv columns differ.
    if not labels_raw["subject_id"].astype(str).str.match(r"^(Ga|Ju|Si)(Co|Pt)\d+$").any():
        raise SystemExit(
            "subject_id format in subject_labels.csv doesn't look like 'GaCo01'. "
            "Open data/processed/subject_labels.csv and check the ID format, "
            "then adjust build_subject_id() / the merge key here."
        )

    merged = features.merge(labels_raw[["subject_id", "stage"]], on="subject_id", how="left")

    unmatched = merged[merged["stage"].isna()]
    if len(unmatched) > 0:
        print(f"WARNING: {len(unmatched)} trials had no matching label (dropped):")
        print(unmatched["subject_id"].unique())
        merged = merged.dropna(subset=["stage"])

    return merged


def main():
    df = load_merged()

    feature_cols = [
        c for c in df.columns
        if c not in ("filename", "cohort", "group", "subject_id", "trial", "stage")
    ]

    # Drop columns that are entirely NaN (e.g. step_length/stride_length/speed —
    # not computable from VGRF-only data, see 03_extract_features.py note)
    X = df[feature_cols].dropna(axis=1, how="all")
    dropped = set(feature_cols) - set(X.columns)
    if dropped:
        print(f"Dropped all-NaN feature columns: {dropped}")

    # Impute any remaining per-row NaNs with column median (some trials may lack
    # detectable stance phases for a sensor)
    X = X.fillna(X.median(numeric_only=True))

    # Drop near-zero-variance features (uninformative, e.g. a sensor that's
    # almost always 0 for most subjects)
    variances = X.var()
    X = X.loc[:, variances > 1e-6]

    # Correlation-based feature selection: if two features are near-duplicates
    # (|corr| > 0.95), drop one — reduces redundancy/multicollinearity before
    # feeding classifiers, similar in spirit to the paper's approach.
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated redundant features (|corr| > 0.95)")
        X = X.drop(columns=to_drop)
    print(f"Final feature count after selection: {X.shape[1]}\n")

    y_raw = df["stage"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
    print(f"Class distribution:\n{y_raw.value_counts()}\n")

    # Group by subject_id so trials from the same subject never split across
    # train/test folds — prevents the model "recognizing" a subject it already
    # saw, which inflates accuracy without reflecting true generalization.
    groups = df["subject_id"]
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", random_state=RANDOM_STATE)),
        ]),
        "Ensemble (Bagging)": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=30, random_state=RANDOM_STATE,
        ),
        "Bayes (GaussianNB)": GaussianNB(),
    }

    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    results_summary = []

    for name, clf in classifiers.items():
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, groups=groups)
        row = {"classifier": name}
        for metric in scoring.keys():
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std()
        results_summary.append(row)

        print(f"{name}:")
        print(f"  Accuracy:  {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
        print(f"  Precision: {scores['test_precision_macro'].mean():.4f} ± {scores['test_precision_macro'].std():.4f}")
        print(f"  Recall:    {scores['test_recall_macro'].mean():.4f} ± {scores['test_recall_macro'].std():.4f}")
        print(f"  F1:        {scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}\n")

    results_df = pd.DataFrame(results_summary)
    out_path = OUT_DIR / "classifier_results_summary.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved summary -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
