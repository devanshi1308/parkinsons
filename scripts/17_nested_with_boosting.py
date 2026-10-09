"""
17_nested_with_boosting.py
Extends 16_model_improvements.py's nested selection (section 1) to include XGBoost and
LightGBM as candidates, closing the gap noted in docs/model_improvements.md section 5
("still needs nested-CV confirmation").

Same de-biasing logic as script 16: for each outer StratifiedGroupKFold fold, the winning
(feature_set, model) pair is chosen on an inner grouped CV over training data only, then scored
once on the untouched outer fold. Candidates = the same sklearn set as script 16 (5 feature sets
x 4 models) PLUS XGBoost/LightGBM on the 3 feature sets used in script 16 section 5
(v2, v3_all, demo_speed_only).

Outputs:
  data/processed/nested_selection_with_boosting.csv
Appends a comparison section to docs/model_improvements.md.

Run from project root:
    python scripts/17_nested_with_boosting.py
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

OUT_DIR = Path("data/processed")
DOC_PATH = Path("docs/model_improvements.md")
SEED = 42
N_OUTER = 10
N_INNER = 5
BOOST_FEATURE_SETS = ("v2", "v3_all", "demo_speed_only")


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boosted_pipelines(seed):
    return {
        "XGBoost": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, eval_metric="mlogloss"))]),
        "LightGBM": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, verbosity=-1))]),
    }


def nested_selection_boosted(df, groups_map, scheme, m12, m07):
    y_raw = m07.get_label_scheme(df, scheme)
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y_raw)
    subj = df["subject_id"].to_numpy()
    n_outer = int(min(N_OUTER, y_raw.value_counts().min()))

    sklearn_candidates = [(fset, name) for fset in m12.FEATURE_SETS for name in ["Random Forest", "HistGradientBoosting", "Logistic Regression", "Bagging"]]
    boosted_candidates = [(fset, name) for fset in BOOST_FEATURE_SETS for name in ["XGBoost", "LightGBM"]]
    candidates = sklearn_candidates + boosted_candidates

    def cols_for(fset):
        fam = m12.FEATURE_SETS[fset]
        return [c for c in groups_map[groups_map.group.isin(fam)].feature if c in df.columns and not df[c].isna().all()]

    X_by_fset = {fset: df[cols_for(fset)].to_numpy(float) for fset in m12.FEATURE_SETS}

    def model_for(fset, name, seed):
        if name in ("XGBoost", "LightGBM"):
            return boosted_pipelines(seed)[name]
        return m12.make_models(seed, m12.CorrelationFilter)[name]

    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X_by_fset["v3_all"], y, subj)):
        best_score, best_cand = -1, None
        n_inner = int(min(N_INNER, pd.Series(y[tr]).value_counts().min()))
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=SEED)
        for fset, name in candidates:
            X = X_by_fset[fset]
            accs = []
            for itr, ite in inner.split(X[tr], y[tr], subj[tr]):
                m = clone(model_for(fset, name, SEED)).fit(X[tr][itr], y[tr][itr])
                accs.append(np.mean(m.predict(X[tr][ite]) == y[tr][ite]))
            score = np.mean(accs)
            if score > best_score:
                best_score, best_cand = score, (fset, name)
        fset, name = best_cand
        X = X_by_fset[fset]
        model = clone(model_for(fset, name, SEED)).fit(X[tr], y[tr])
        outer_acc = np.mean(model.predict(X[te]) == y[te])
        outer_f1 = f1_score(y[te], model.predict(X[te]), average="macro", zero_division=0)
        rows.append({"scheme": scheme, "outer_fold": fold, "selected_feature_set": fset, "selected_model": name,
                     "inner_cv_score": best_score, "outer_accuracy": outer_acc, "outer_f1_macro": outer_f1})
        print(f"  [{scheme}] outer fold {fold + 1}/{n_outer}: picked {fset}/{name} (inner {best_score:.3f}) -> outer acc {outer_acc:.3f}")
    return pd.DataFrame(rows)


def main():
    m12 = load_module("12_train_models_v3.py", "m12")
    m07 = load_module("07_train_multi_scheme.py", "m07")
    feats = pd.read_csv("data/processed/features_v3.csv")
    labels = pd.read_csv("data/processed/subject_labels.csv")
    groups_map = pd.read_csv("data/processed/feature_groups_v3.csv")
    df = feats.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"]).reset_index(drop=True)
    print(f"Trials {len(df)}, subjects {df.subject_id.nunique()}\n")

    results = pd.concat([nested_selection_boosted(df, groups_map, s, m12, m07) for s in ("4class", "binary")])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "nested_selection_with_boosting.csv", index=False)

    # compare against the sklearn-only nested results from script 16
    sklearn_only = pd.read_csv(OUT_DIR / "nested_selection_results.csv")
    lines = ["", "## 1b. Nested selection WITH XGBoost/LightGBM as candidates (`17_nested_with_boosting.py`)", "",
             "Extends section 1's nested (de-biased) selection to also consider XGBoost and LightGBM "
             "(on v2 / v3_all / demo_speed_only, matching section 5's ablation). Closes the "
             "\"still needs nested-CV confirmation\" caveat from section 5.", ""]
    for scheme in ("4class", "binary"):
        old = sklearn_only[sklearn_only.scheme == scheme]
        new = results[results.scheme == scheme]
        top = (new.selected_feature_set + "/" + new.selected_model).value_counts().head(3)
        lines += [f"**{scheme}**: sklearn-only nested accuracy {old.outer_accuracy.mean():.3f} (sd {old.outer_accuracy.std():.3f}) "
                  f"vs with boosting {new.outer_accuracy.mean():.3f} (sd {new.outer_accuracy.std():.3f}); "
                  f"macro-F1 {old.outer_f1_macro.mean():.3f} vs {new.outer_f1_macro.mean():.3f}. "
                  f"Folds most often picked: " + ", ".join(f"{k} ({v})" for k, v in top.items()), ""]
    with open(DOC_PATH, "a") as f:
        f.write("\n".join(lines))
    print(f"\nAppended to {DOC_PATH}")
    print(results.groupby("scheme")[["outer_accuracy", "outer_f1_macro"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
