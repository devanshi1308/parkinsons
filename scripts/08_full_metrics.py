"""
08_full_metrics.py
Computes the full metric suite the baseline paper reports (Tables 5-9):
  accuracy, sensitivity, specificity, PPV, NPV, F-score, confusion matrix, ROC/AUC
using the SAME subject-grouped, leakage-free 10-fold CV protocol as before.

For multi-class schemes (4class, 3class), metrics are computed one-vs-rest per class
(same approach the paper uses per H&Y stage) then macro-averaged, plus reported per-class.
For binary (Healthy/PD), standard binary metrics apply directly.

Requires: pip install scikit-learn scipy pandas numpy

Run from project root:
    python scripts/08_full_metrics.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
FEATURES_PATH = Path("data/processed/features_v2.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
OUT_PATH = Path("data/processed/full_metrics_report.csv")


def load_merged() -> pd.DataFrame:
    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    merged = features.merge(labels[["subject_id", "stage"]], on="subject_id", how="left")
    return merged.dropna(subset=["stage"])


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("filename", "cohort", "group", "subject_id", "trial", "stage")]
    X = df[feature_cols].dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.var() > 1e-6]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    return X.drop(columns=to_drop) if to_drop else X


def get_label_scheme(df: pd.DataFrame, scheme: str) -> pd.Series:
    if scheme == "4class":
        return df["stage"]
    elif scheme == "3class":
        return df["stage"].replace({"Severe": "Moderate+Severe", "Moderate": "Moderate+Severe"})
    elif scheme == "binary":
        return df["stage"].apply(lambda s: "Healthy" if s == "Healthy" else "PD")
    raise ValueError(scheme)


def per_class_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """One-vs-rest TP/TN/FP/FN per class, then derive sensitivity/specificity/PPV/NPV.
    Same logic the paper uses (Section 5, Eqs 2-7)."""
    per_class = {}
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        tn = np.sum((y_true != c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        f_score = (2 * ppv * sensitivity / (ppv + sensitivity)) if (ppv and sensitivity and (ppv + sensitivity) > 0) else np.nan

        per_class[c] = dict(tp=tp, tn=tn, fp=fp, fn=fn, sensitivity=sensitivity,
                             specificity=specificity, ppv=ppv, npv=npv, f_score=f_score)
    return per_class


def run_scheme_full_metrics(X: pd.DataFrame, y_raw: pd.Series, groups: pd.Series, scheme_name: str) -> list:
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    n_splits = min(10, y_raw.value_counts().min())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))]),
        "Ensemble (Bagging)": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=30, random_state=RANDOM_STATE),
        "Bayes (GaussianNB)": GaussianNB(),
    }

    rows = []
    for clf_name, clf in classifiers.items():
        # Accumulate per-class confusion counts across all folds (pooled, not averaged per-fold —
        # more stable given small per-fold class counts, e.g. Severe)
        pooled_true, pooled_pred = [], []
        fold_aucs = []

        for train_idx, test_idx in cv.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pooled_true.extend(y_test)
            pooled_pred.extend(y_pred)

            # ROC-AUC (needs probability scores; one-vs-rest for multiclass)
            try:
                y_proba = clf.predict_proba(X_test)
                if n_classes == 2:
                    if len(np.unique(y_test)) > 1:
                        fold_aucs.append(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
                    present = np.unique(y_test)
                    if len(present) > 1:
                        fold_aucs.append(roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr"))
            except Exception:
                pass  # some folds may lack both classes for a given label; skip AUC that fold

        pooled_true = np.array(pooled_true)
        pooled_pred = np.array(pooled_pred)
        per_class = per_class_confusion_metrics(pooled_true, pooled_pred, n_classes)

        overall_accuracy = np.mean(pooled_true == pooled_pred)
        mean_auc = np.nanmean(fold_aucs) if fold_aucs else np.nan

        for c_idx, class_name in enumerate(le.classes_):
            m = per_class[c_idx]
            rows.append({
                "scheme": scheme_name, "classifier": clf_name, "class": class_name,
                "TP": m["tp"], "TN": m["tn"], "FP": m["fp"], "FN": m["fn"],
                "accuracy_overall": overall_accuracy,
                "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                "PPV": m["ppv"], "NPV": m["npv"], "f_score": m["f_score"],
                "macro_auc": mean_auc,
            })

        print(f"[{scheme_name}] {clf_name}: acc={overall_accuracy:.3f}  macro_AUC={mean_auc:.3f}")

    return rows


def main():
    df = load_merged()
    X = build_feature_matrix(df)
    groups = df["subject_id"]
    print(f"Feature matrix: {X.shape}\n")

    all_rows = []
    for scheme in ["4class", "3class", "binary"]:
        print(f"\n{'='*60}\nScheme: {scheme}")
        y_raw = get_label_scheme(df, scheme)
        all_rows.extend(run_scheme_full_metrics(X, y_raw, groups, scheme))

    report = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_PATH, index=False)
    print(f"\n\nSaved full metrics report -> {OUT_PATH.resolve()}")
    print("\nColumns match the baseline paper's Table 5/7 format: TN, TP, FN, FP, Accuracy, Sensitivity, Specificity (+ PPV, NPV, F-score, AUC added)")


if __name__ == "__main__":
    main()
