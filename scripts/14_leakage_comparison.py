"""
14_leakage_comparison.py
Quantifies subject-level leakage: same features, labels, classifiers and metrics as
07_train_multi_scheme.py, but compares two cross-validation protocols:

  leaky    StratifiedKFold on trials (ignores subject) - a subject's trials can land in
           both train and test folds (what trial-level CV in the baseline effectively does)
  grouped  StratifiedGroupKFold by subject_id - no subject in both train and test

Each protocol is run with two preprocessing modes:
  global   median-fill / variance / correlation filters fit on the whole dataset first
           (identical to 07_train_multi_scheme.py; label-free but not fold-internal)
  infold   the same filters fit inside each training fold only (sensitivity check)

Each combination is repeated over N_REPEATS seeds (42 .. 42+N_REPEATS-1).
A leak audit reports, per leaky fold, the share of test trials whose subject also
appears in the training fold.

Outputs:
  data/processed/leakage_comparison.csv
  data/processed/leakage_audit.csv
  docs/figures/leakage_gap.png   (skipped if matplotlib is not installed)

Run from project root:
    python scripts/14_leakage_comparison.py
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_SEED = 42
N_REPEATS = 5
N_FOLDS = 10
SCHEMES = ["4class", "3class", "binary"]
OUT_CMP = Path("data/processed/leakage_comparison.csv")
OUT_AUDIT = Path("data/processed/leakage_audit.csv")
OUT_FIG = Path("docs/figures/leakage_gap.png")


def load_script_07():
    """Reuse loading / label / feature code from 07 so both scripts cannot drift apart."""
    path = Path(__file__).with_name("07_train_multi_scheme.py")
    spec = importlib.util.spec_from_file_location("multi_scheme", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of each feature pair with |corr| > threshold (fit on training data only)."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = pd.DataFrame(X).corr().abs().fillna(0).values
        upper = np.triu(corr, k=1)
        self.keep_ = ~(upper > self.threshold).any(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def make_classifiers(seed):
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "SVM": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", random_state=seed))]),
        "Ensemble (Bagging)": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=seed), n_estimators=30, random_state=seed),
        "Bayes (GaussianNB)": GaussianNB(),
    }


def with_infold_preprocessing(clf):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-6)),
        ("corr", CorrelationFilter(0.95)),
        ("clf", clf),
    ])


def make_splitter(protocol, n_splits, seed):
    if protocol == "leaky":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validate_manual(clf, X, y, groups, splitter):
    accs, f1s = [], []
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = clone(clf)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        accs.append(float(np.mean(pred == y[test_idx])))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    return accs, f1s


def audit_leaky_folds(y, groups, n_splits, seed, scheme):
    rows = []
    splitter = make_splitter("leaky", n_splits, seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        train_subjects = set(groups[train_idx])
        shared = np.array([g in train_subjects for g in groups[test_idx]])
        rows.append({
            "scheme": scheme, "seed": seed, "fold": fold,
            "n_test_trials": len(test_idx),
            "pct_test_trials_subject_in_train": 100.0 * shared.mean(),
        })
    return rows


def main():
    m07 = load_script_07()
    df = m07.load_merged()
    X_global = m07.build_feature_matrix(df)
    feature_cols = [c for c in df.columns if c not in ("filename", "cohort", "group", "subject_id", "trial", "stage")]
    X_raw = df[feature_cols].dropna(axis=1, how="all")
    groups = df["subject_id"].to_numpy()
    print(f"Trials: {len(df)}  Subjects: {df['subject_id'].nunique()}")
    print(f"Features: global-filtered={X_global.shape[1]}  raw={X_raw.shape[1]}")

    rows, audit_rows = [], []
    for scheme in SCHEMES:
        y_raw = m07.get_label_scheme(df, scheme)
        y = LabelEncoder().fit_transform(y_raw)
        n_splits = int(min(N_FOLDS, y_raw.value_counts().min()))
        print(f"\n=== {scheme}  classes={sorted(y_raw.unique())}  folds={n_splits} ===")

        for repeat in range(N_REPEATS):
            seed = BASE_SEED + repeat
            audit_rows.extend(audit_leaky_folds(y, groups, n_splits, seed, scheme))

            for prep, X_df in (("global", X_global), ("infold", X_raw)):
                X = X_df.to_numpy()
                for name, clf in make_classifiers(seed).items():
                    model = with_infold_preprocessing(clf) if prep == "infold" else clf
                    for protocol in ("leaky", "grouped"):
                        splitter = make_splitter(protocol, n_splits, seed)
                        accs, f1s = cross_validate_manual(model, X, y, groups, splitter)
                        rows.append({
                            "scheme": scheme, "classifier": name, "protocol": protocol,
                            "preprocessing": prep, "seed": seed, "n_folds": n_splits,
                            "accuracy": np.mean(accs), "f1_macro": np.mean(f1s),
                        })
            print(f"  repeat {repeat + 1}/{N_REPEATS} done")

    runs = pd.DataFrame(rows)
    summary = (runs.groupby(["scheme", "classifier", "preprocessing", "protocol"])
               .agg(accuracy_mean=("accuracy", "mean"), accuracy_std_across_seeds=("accuracy", "std"),
                    f1_macro_mean=("f1_macro", "mean"), f1_macro_std_across_seeds=("f1_macro", "std"),
                    n_seeds=("seed", "nunique"))
               .reset_index())
    wide = summary.pivot_table(index=["scheme", "classifier", "preprocessing"], columns="protocol",
                               values=["accuracy_mean", "f1_macro_mean"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide["accuracy_gap_pts"] = 100 * (wide["accuracy_mean_leaky"] - wide["accuracy_mean_grouped"])
    wide["f1_gap_pts"] = 100 * (wide["f1_macro_mean_leaky"] - wide["f1_macro_mean_grouped"])
    result = summary.merge(wide[["scheme", "classifier", "preprocessing", "accuracy_gap_pts", "f1_gap_pts"]],
                           on=["scheme", "classifier", "preprocessing"])

    OUT_CMP.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CMP, index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_AUDIT, index=False)
    print(f"\nSaved -> {OUT_CMP}\nSaved -> {OUT_AUDIT}")

    print("\n=== Accuracy (mean over seeds): leaky vs grouped, global preprocessing ===")
    show = result[result["preprocessing"] == "global"].pivot_table(
        index=["scheme", "classifier"], columns="protocol", values="accuracy_mean")
    show["gap_pts"] = 100 * (show["leaky"] - show["grouped"])
    print(show.round(3).to_string())
    print("\n=== Same, in-fold preprocessing ===")
    show2 = result[result["preprocessing"] == "infold"].pivot_table(
        index=["scheme", "classifier"], columns="protocol", values="accuracy_mean")
    show2["gap_pts"] = 100 * (show2["leaky"] - show2["grouped"])
    print(show2.round(3).to_string())
    print("\n=== Leak audit: % of leaky-fold test trials whose subject is also in training ===")
    print(audit.groupby("scheme")["pct_test_trials_subject_in_train"].agg(["mean", "min", "max"]).round(1).to_string())

    make_figure(result)


def make_figure(result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed - skipping figure (pip install matplotlib)")
        return

    data = result[result["preprocessing"] == "global"]
    classifiers = list(data["classifier"].unique())
    fig, axes = plt.subplots(1, len(SCHEMES), figsize=(13, 4), sharey=True)
    width = 0.38
    for ax, scheme in zip(axes, SCHEMES):
        sub = data[data["scheme"] == scheme].set_index(["classifier", "protocol"])
        x = np.arange(len(classifiers))
        for offset, protocol, color in ((-width / 2, "leaky", "#c0504d"), (width / 2, "grouped", "#4f81bd")):
            means = [sub.loc[(c, protocol), "accuracy_mean"] for c in classifiers]
            errs = [sub.loc[(c, protocol), "accuracy_std_across_seeds"] for c in classifiers]
            ax.bar(x + offset, means, width, yerr=errs, label=protocol, color=color, capsize=2)
        ax.set_title(scheme)
        ax.set_xticks(x)
        ax.set_xticklabels([c.split(" (")[0] for c in classifiers], rotation=25, ha="right")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy (mean over seeds)")
    axes[0].legend(title="CV protocol")
    fig.suptitle("Trial-level (leaky) vs subject-grouped cross-validation")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    print(f"Saved -> {OUT_FIG}")


if __name__ == "__main__":
    main()
