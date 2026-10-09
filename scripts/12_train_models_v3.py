"""
12_train_models_v3.py
Feature-family ablation x extended model set under the honest (subject-grouped) protocol.

Feature sets (cumulative, from feature_groups_v3.csv)
  v2               stats + timing + cop + symmetry            (what scripts 05-11 used)
  v3_signal        v2 + spectral + wavelet + regularity + double_support   (VGRF-derived only)
  v3_signal_demo   v3_signal + demographics (age, sex, height, weight, BMI)
  v3_all           v3_signal_demo + speed (walking speed and derived stride/step length)
  demo_speed_only  demographics + speed, no gait-signal features (diagnostic baseline)

Models: Decision Tree, SVM (linear), Bagging, GaussianNB (baseline four) plus Random Forest,
Extra Trees, HistGradientBoosting, Logistic Regression, soft-voting ensemble.

Protocol: StratifiedGroupKFold by subject_id (10 folds), seeds 42..42+N_REPEATS-1.
All preprocessing (median fill, variance filter, |r|>0.95 correlation filter, 1-99 percentile
clipping, scaling) is fit inside each training fold.

Outputs:
  data/processed/model_comparison_v3.csv         (per scheme/feature set/model, mean over seeds)
  data/processed/model_comparison_v3_runs.csv    (per seed)

Run from project root:
    python scripts/12_train_models_v3.py
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

FEATURES_PATH = Path("data/processed/features_v3.csv")
GROUPS_PATH = Path("data/processed/feature_groups_v3.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
OUT_SUMMARY = Path("data/processed/model_comparison_v3.csv")
OUT_RUNS = Path("data/processed/model_comparison_v3_runs.csv")

BASE_SEED = 42
N_REPEATS = 3
N_FOLDS = 10
SCHEMES = ["4class", "3class", "binary"]

FEATURE_SETS = {
    "v2": ["stats", "timing", "cop", "symmetry"],
    "v3_signal": ["stats", "timing", "cop", "symmetry", "spectral", "wavelet", "regularity", "double_support"],
    "v3_signal_demo": ["stats", "timing", "cop", "symmetry", "spectral", "wavelet", "regularity", "double_support", "demographics"],
    "v3_all": ["stats", "timing", "cop", "symmetry", "spectral", "wavelet", "regularity", "double_support", "demographics", "speed"],
    "demo_speed_only": ["demographics", "speed"],
}


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of each feature pair with |corr| > threshold (fit on training data only)."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = pd.DataFrame(X).corr().abs().fillna(0).values
        self.keep_ = ~(np.triu(corr, k=1) > self.threshold).any(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


class PercentileClip(BaseEstimator, TransformerMixin):
    """Clip each feature to its [lo, hi] percentiles from the training fold (tames detection outliers)."""

    def __init__(self, lo=1, hi=99):
        self.lo, self.hi = lo, hi

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_, self.hi_ = np.percentile(X, self.lo, axis=0), np.percentile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


def make_models(seed, corr_filter_cls):
    def pipe(clf, scale):
        steps = [("impute", SimpleImputer(strategy="median")), ("var", VarianceThreshold(1e-6)),
                 ("corr", corr_filter_cls(0.95)), ("clip", PercentileClip())]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps + [("clf", clf)])

    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, l2_regularization=1.0, random_state=seed)
    lr = LogisticRegression(C=0.5, max_iter=3000)
    return {
        "Decision Tree": pipe(DecisionTreeClassifier(random_state=seed), False),
        "SVM": pipe(SVC(kernel="linear", probability=True, random_state=seed), True),
        "Bagging": pipe(BaggingClassifier(estimator=DecisionTreeClassifier(random_state=seed), n_estimators=30, random_state=seed), False),
        "Naive Bayes": pipe(GaussianNB(), False),
        "Random Forest": pipe(rf, False),
        "Extra Trees": pipe(ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1), False),
        "HistGradientBoosting": pipe(hgb, False),
        "Logistic Regression": pipe(lr, True),
        "Voting (RF+HGB+LR)": pipe(VotingClassifier([("rf", clone(rf)), ("hgb", clone(hgb)), ("lr", Pipeline([("s", StandardScaler()), ("lr", clone(lr))]))], voting="soft"), False),
    }


def run_task(scheme, fset, model_name, seed, X, y, groups, n_splits, corr_filter_cls):
    model = make_models(seed, corr_filter_cls)[model_name]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s, bals, aucs = [], [], [], []
    for tr, te in cv.split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        accs.append(np.mean(pred == y[te]))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        bals.append(balanced_accuracy_score(y[te], pred))
        if scheme == "binary" and len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return {"scheme": scheme, "feature_set": fset, "model": model_name, "seed": seed,
            "accuracy": np.mean(accs), "f1_macro": np.mean(f1s), "balanced_accuracy": np.mean(bals),
            "auc": np.mean(aucs) if aucs else np.nan, "n_features_in": X.shape[1]}


def main():
    m07 = load_module("07_train_multi_scheme.py", "m07")
    feats = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    df = feats.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"]).reset_index(drop=True)
    groups_map = pd.read_csv(GROUPS_PATH)
    subj = df["subject_id"].to_numpy()
    print(f"Trials: {len(df)}  Subjects: {df['subject_id'].nunique()}")

    tasks = []
    for scheme in SCHEMES:
        y_raw = m07.get_label_scheme(df, scheme)
        y = LabelEncoder().fit_transform(y_raw)
        n_splits = int(min(N_FOLDS, y_raw.value_counts().min()))
        for fset, fgroups in FEATURE_SETS.items():
            cols = groups_map[groups_map.group.isin(fgroups)].feature.tolist()
            cols = [c for c in cols if c in df.columns and not df[c].isna().all()]
            X = df[cols].to_numpy(float)
            for model_name in make_models(BASE_SEED, CorrelationFilter):
                for r in range(N_REPEATS):
                    tasks.append((scheme, fset, model_name, BASE_SEED + r, X, y, subj, n_splits, CorrelationFilter))
    print(f"Running {len(tasks)} experiments in parallel ...")
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(run_task)(*t) for t in tasks)

    runs = pd.DataFrame(rows)
    OUT_RUNS.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT_RUNS, index=False)
    summary = (runs.groupby(["scheme", "feature_set", "model"])
               .agg(accuracy=("accuracy", "mean"), accuracy_sd_seeds=("accuracy", "std"), f1_macro=("f1_macro", "mean"),
                    balanced_accuracy=("balanced_accuracy", "mean"), auc=("auc", "mean"), n_features_in=("n_features_in", "first"))
               .reset_index())
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"\nSaved -> {OUT_SUMMARY}\nSaved -> {OUT_RUNS}")

    for scheme in SCHEMES:
        print(f"\n=== {scheme}: accuracy (mean over {N_REPEATS} seeds) ===")
        p = summary[summary.scheme == scheme].pivot_table(index="model", columns="feature_set", values="accuracy")
        print(p[list(FEATURE_SETS)].round(3).to_string())
        print(f"--- {scheme}: macro-F1 ---")
        p = summary[summary.scheme == scheme].pivot_table(index="model", columns="feature_set", values="f1_macro")
        print(p[list(FEATURE_SETS)].round(3).to_string())
    print("\n=== binary AUC ===")
    p = summary[summary.scheme == "binary"].pivot_table(index="model", columns="feature_set", values="auc")
    print(p[list(FEATURE_SETS)].round(3).to_string())


if __name__ == "__main__":
    main()
