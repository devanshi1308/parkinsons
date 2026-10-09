"""
16_model_improvements.py
Five follow-ups to the model_comparison_v3 results (see rules.md "Results log" and the
2026-10 model-improvement discussion). Each section is independent; comment out calls in
main() to run a subset.

  1. nested_selection   Picking the best (feature_set, model) on the SAME outer folds used to
                         report accuracy is model-selection bias. Here, for each outer
                         StratifiedGroupKFold fold, the winning (feature_set, model) pair is
                         chosen on an INNER grouped CV over the training data only, then scored
                         once on the held-out outer fold. This is the de-biased estimate of the
                         "best" accuracy from model_comparison_v3.
  2. subject_aggregation Per-trial out-of-fold probabilities (grouped CV) are averaged per
                         subject before thresholding. Compares trial-level vs subject-level
                         accuracy for the same underlying predictions (no new leakage: grouping
                         was already by subject).
  3. ordinal_vs_multiclass   H&Y stage is ordinal. Compares plain multiclass RF against a
                         Frank-Hall ordinal decomposition (K-1 binary "is stage > k" classifiers
                         combined into class probabilities) on accuracy, macro-F1, and mean
                         absolute stage error (0..3) -- the metric that actually rewards
                         being "off by one" less than "off by two".
  4. compact_features    L1-logistic feature selection fit INSIDE each training fold, with
                         selection frequency across folds used to build one final, named,
                         compact feature list -- vs. the full ~200-feature set.
  5. boosted_models      Adds XGBoost and LightGBM (now that libomp is installed) to the
                         feature-set ablation for binary and 4-class on v2 / v3_all /
                         demo_speed_only.

Outputs (data/processed/):
  nested_selection_results.csv, subject_level_aggregation.csv, ordinal_vs_multiclass.csv,
  compact_feature_list.csv, compact_vs_full.csv, model_comparison_v3_boosted.csv
and docs/model_improvements.md summarizing all five.

Run from project root:
    python scripts/16_model_improvements.py
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

FEATURES_PATH = Path("data/processed/features_v3.csv")
GROUPS_PATH = Path("data/processed/feature_groups_v3.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")
OUT_DIR = Path("data/processed")
DOC_PATH = Path("docs/model_improvements.md")
SEED = 42
N_OUTER = 10
N_INNER = 5
STAGE_ORDER = ["Healthy", "Mild", "Moderate", "Severe"]


def m12():
    spec = importlib.util.spec_from_file_location("m12", Path(__file__).with_name("12_train_models_v3.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def m07():
    spec = importlib.util.spec_from_file_location("m07", Path(__file__).with_name("07_train_multi_scheme.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load():
    feats = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    groups_map = pd.read_csv(GROUPS_PATH)
    df = feats.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"]).reset_index(drop=True)
    return df, groups_map


def cols_for(fset, groups_map, df):
    fam = m12().FEATURE_SETS[fset]
    cols = [c for c in groups_map[groups_map.group.isin(fam)].feature if c in df.columns and not df[c].isna().all()]
    return cols


# ---------------------------------------------------------------- 1. nested selection
def nested_selection(df, groups_map, scheme):
    mod = m12()
    y_raw = m07().get_label_scheme(df, scheme)
    y = LabelEncoder().fit_transform(y_raw)
    subj = df["subject_id"].to_numpy()
    n_outer = int(min(N_OUTER, y_raw.value_counts().min()))
    candidates = [(fset, name) for fset in mod.FEATURE_SETS for name in ["Random Forest", "HistGradientBoosting", "Logistic Regression", "Bagging"]]
    X_by_fset = {fset: df[cols_for(fset, groups_map, df)].to_numpy(float) for fset in mod.FEATURE_SETS}

    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X_by_fset["v3_all"], y, subj)):
        best_score, best_cand = -1, None
        n_inner = int(min(N_INNER, pd.Series(y[tr]).value_counts().min()))
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=SEED)
        for fset, name in candidates:
            X = X_by_fset[fset]
            model = mod.make_models(SEED, mod.CorrelationFilter)[name]
            accs = []
            for itr, ite in inner.split(X[tr], y[tr], subj[tr]):
                m = clone(model).fit(X[tr][itr], y[tr][itr])
                accs.append(np.mean(m.predict(X[tr][ite]) == y[tr][ite]))
            score = np.mean(accs)
            if score > best_score:
                best_score, best_cand = score, (fset, name)
        fset, name = best_cand
        X = X_by_fset[fset]
        model = clone(mod.make_models(SEED, mod.CorrelationFilter)[name]).fit(X[tr], y[tr])
        outer_acc = np.mean(model.predict(X[te]) == y[te])
        outer_f1 = f1_score(y[te], model.predict(X[te]), average="macro", zero_division=0)
        rows.append({"scheme": scheme, "outer_fold": fold, "selected_feature_set": fset, "selected_model": name,
                     "inner_cv_score": best_score, "outer_accuracy": outer_acc, "outer_f1_macro": outer_f1})
        print(f"  [{scheme}] outer fold {fold + 1}/{n_outer}: picked {fset}/{name} (inner {best_score:.3f}) -> outer acc {outer_acc:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. subject-level aggregation
def subject_aggregation(df, groups_map):
    mod = m12()
    y = (df["stage"] != "Healthy").astype(int).to_numpy()
    subj = df["subject_id"].to_numpy()
    cols = cols_for("v3_all", groups_map, df)
    X = df[cols].to_numpy(float)
    n_splits = N_OUTER
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    prob = np.zeros(len(y))
    for tr, te in cv.split(X, y, subj):
        model = clone(mod.make_models(SEED, mod.CorrelationFilter)["Random Forest"]).fit(X[tr], y[tr])
        prob[te] = model.predict_proba(X[te])[:, 1]
    trial_acc = np.mean((prob >= 0.5).astype(int) == y)
    trial_f1 = f1_score(y, (prob >= 0.5).astype(int), average="macro")

    agg = pd.DataFrame({"subject_id": subj, "prob": prob, "y": y}).groupby("subject_id").agg(prob=("prob", "mean"), y=("y", "first"))
    subj_pred = (agg.prob >= 0.5).astype(int)
    subj_acc = np.mean(subj_pred == agg.y)
    subj_f1 = f1_score(agg.y, subj_pred, average="macro")
    out = pd.DataFrame([
        {"level": "trial", "n": len(y), "accuracy": trial_acc, "f1_macro": trial_f1},
        {"level": "subject (mean of trial probabilities)", "n": len(agg), "accuracy": subj_acc, "f1_macro": subj_f1},
    ])
    print(out.round(3).to_string(index=False))
    return out


# ---------------------------------------------------------------- 3. ordinal vs multiclass
def ordinal_vs_multiclass(df, groups_map):
    mod = m12()
    cols = cols_for("v3_all", groups_map, df)
    X = df[cols].to_numpy(float)
    stage_idx = df["stage"].map({s: i for i, s in enumerate(STAGE_ORDER)}).to_numpy()
    subj = df["subject_id"].to_numpy()
    n_splits = N_OUTER
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    def prep(Xtr, Xte):
        imp = SimpleImputer(strategy="median").fit(Xtr)
        return imp.transform(Xtr), imp.transform(Xte)

    rows = []
    for method in ("multiclass_rf", "ordinal_frank_hall"):
        preds = np.zeros(len(stage_idx), int)
        for tr, te in cv.split(X, stage_idx, subj):
            Xtr, Xte = prep(X[tr], X[te])
            if method == "multiclass_rf":
                m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, stage_idx[tr])
                preds[te] = m.predict(Xte)
            else:
                # Frank & Hall (2001) ordinal decomposition: K-1 binary classifiers "stage > k"
                k_vals = sorted(set(stage_idx[tr]))[:-1]
                bin_probs = []
                for k in k_vals:
                    clf = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=0.5, max_iter=3000))])
                    clf.fit(Xtr, (stage_idx[tr] > k).astype(int))
                    bin_probs.append(clf.predict_proba(Xte)[:, 1])  # P(stage > k)
                bin_probs = np.array(bin_probs)  # (K-1, n_test)
                n_classes = len(STAGE_ORDER)
                class_probs = np.zeros((n_classes, Xte.shape[0]))
                class_probs[0] = 1 - bin_probs[0]
                for i in range(1, n_classes - 1):
                    class_probs[i] = bin_probs[i - 1] - bin_probs[i]
                class_probs[-1] = bin_probs[-1]
                class_probs = np.clip(class_probs, 0, None)
                preds[te] = class_probs.argmax(axis=0)
        acc = np.mean(preds == stage_idx)
        f1 = f1_score(stage_idx, preds, average="macro", zero_division=0)
        mae = mean_absolute_error(stage_idx, preds)
        rows.append({"method": method, "accuracy": acc, "f1_macro": f1, "mean_abs_stage_error": mae})
        print(f"  {method}: acc {acc:.3f}  f1_macro {f1:.3f}  mean_abs_stage_error {mae:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. compact feature set
def compact_features(df, groups_map, clinical_name_fn):
    mod = m12()
    cols = cols_for("v3_all", groups_map, df)
    X = df[cols].to_numpy(float)
    y = (df["stage"] != "Healthy").astype(int).to_numpy()
    subj = df["subject_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)

    selected_counts = np.zeros(len(cols), int)
    full_accs, compact_accs = [], []
    for tr, te in cv.split(X, y, subj):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        sel = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.03, random_state=SEED)).fit(Xtr_s, y[tr])  # default threshold: strictly nonzero L1 coefficients
        mask = sel.get_support()
        selected_counts += mask.astype(int)

        rf_full = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr, y[tr])
        full_accs.append(np.mean(rf_full.predict(Xte) == y[te]))
        rf_compact = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr[:, mask], y[tr])
        compact_accs.append(np.mean(rf_compact.predict(Xte[:, mask]) == y[te]))

    freq = selected_counts / N_OUTER
    final_cols = [c for c, f in zip(cols, freq) if f >= 0.5]
    feat_df = pd.DataFrame({"feature": cols, "clinical_name": [clinical_name_fn(c) for c in cols], "selected_fraction_of_folds": freq}).sort_values("selected_fraction_of_folds", ascending=False)
    compare = pd.DataFrame([
        {"feature_set": f"full ({len(cols)} features)", "accuracy_mean": np.mean(full_accs), "accuracy_sd": np.std(full_accs)},
        {"feature_set": f"compact (>=50% folds, {len(final_cols)} features)", "accuracy_mean": np.mean(compact_accs), "accuracy_sd": np.std(compact_accs)},
    ])
    print(f"  Compact feature set ({len(final_cols)} features, selected in >=50% of folds):")
    print(feat_df[feat_df.selected_fraction_of_folds >= 0.5][["clinical_name", "selected_fraction_of_folds"]].to_string(index=False))
    print(compare.round(3).to_string(index=False))
    return feat_df, compare


# ---------------------------------------------------------------- 5. boosted models
def boosted_models(df, groups_map):
    import lightgbm as lgb
    import xgboost as xgb
    mod = m12()
    rows = []
    for scheme in ("4class", "binary"):
        y_raw = m07().get_label_scheme(df, scheme)
        y = LabelEncoder().fit_transform(y_raw)
        n_splits = int(min(N_OUTER, y_raw.value_counts().min()))
        for fset in ("v2", "v3_all", "demo_speed_only"):
            cols = cols_for(fset, groups_map, df)
            X = df[cols].to_numpy(float)
            models = {
                "XGBoost": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", xgb.XGBClassifier(
                    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=SEED, eval_metric="mlogloss"))]),
                "LightGBM": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", lgb.LGBMClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=SEED, verbosity=-1))]),
            }
            for name, model in models.items():
                for seed_add in range(3):
                    seed = SEED + seed_add
                    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    accs, f1s = [], []
                    for tr, te in cv.split(X, y, df["subject_id"]):
                        m = clone(model).fit(X[tr], y[tr])
                        pred = m.predict(X[te])
                        accs.append(np.mean(pred == y[te]))
                        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
                    rows.append({"scheme": scheme, "feature_set": fset, "model": name, "seed": seed, "accuracy": np.mean(accs), "f1_macro": np.mean(f1s)})
            print(f"  [{scheme}/{fset}] XGBoost & LightGBM done")
    runs = pd.DataFrame(rows)
    summary = runs.groupby(["scheme", "feature_set", "model"]).agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), f1_macro=("f1_macro", "mean")).reset_index()
    print(summary.round(3).to_string(index=False))
    return summary


def clinical_name(col):
    spec = importlib.util.spec_from_file_location("m40", Path(__file__).with_name("40_shap_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.clinical_name(col)


def main():
    df, groups_map = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Trials {len(df)}, subjects {df.subject_id.nunique()}\n")

    print("=== 1. Nested feature-set/model selection (de-biased) ===")
    nested = pd.concat([nested_selection(df, groups_map, "4class"), nested_selection(df, groups_map, "binary")])
    nested.to_csv(OUT_DIR / "nested_selection_results.csv", index=False)

    print("\n=== 2. Subject-level probability aggregation (binary) ===")
    subj_agg = subject_aggregation(df, groups_map)
    subj_agg.to_csv(OUT_DIR / "subject_level_aggregation.csv", index=False)

    print("\n=== 3. Ordinal vs multiclass (4-class H&Y) ===")
    ordinal = ordinal_vs_multiclass(df, groups_map)
    ordinal.to_csv(OUT_DIR / "ordinal_vs_multiclass.csv", index=False)

    print("\n=== 4. Compact (L1-selected) feature set vs full ===")
    feat_df, compact_cmp = compact_features(df, groups_map, clinical_name)
    feat_df.to_csv(OUT_DIR / "compact_feature_list.csv", index=False)
    compact_cmp.to_csv(OUT_DIR / "compact_vs_full.csv", index=False)

    print("\n=== 5. XGBoost / LightGBM added to the ablation ===")
    boosted = boosted_models(df, groups_map)
    boosted.to_csv(OUT_DIR / "model_comparison_v3_boosted.csv", index=False)

    # ---- markdown summary
    lines = ["# Model-improvement follow-ups", "",
             "Five checks requested after the first features-v3 / extended-model results. All subject-grouped CV, seed 42 (+ repeats where noted).", "",
             "## 1. De-biased selection (nested CV)", "",
             "`model_comparison_v3.csv` picked the best of 45 (feature_set, model) pairs on the same folds used to report accuracy -- optimistic. "
             "Here, each outer fold's winning pair is chosen on an inner CV over training data only, then scored once on the untouched outer fold.", ""]
    for scheme in ("4class", "binary"):
        sub = nested[nested.scheme == scheme]
        lines += [f"**{scheme}**: nested outer accuracy {sub.outer_accuracy.mean():.3f} (sd {sub.outer_accuracy.std():.3f}), "
                  f"outer macro-F1 {sub.outer_f1_macro.mean():.3f}. Folds most often picked: "
                  + ", ".join(f"{k} ({v})" for k, v in (sub.selected_feature_set + '/' + sub.selected_model).value_counts().head(3).items()), ""]
    lines += ["## 2. Subject-level aggregation (binary, Random Forest on v3_all)", ""]
    lines.append(subj_agg.round(3).to_markdown(index=False))
    lines += ["", "## 3. Ordinal vs multiclass (4-class)", ""]
    lines.append(ordinal.round(3).to_markdown(index=False))
    lines += ["", "Mean absolute stage error counts a Healthy-predicted-as-Severe mistake as worse than Healthy-predicted-as-Mild; plain accuracy does not.", "",
              "## 4. Compact, clinically-named feature set", "",
              f"L1-logistic selection inside each fold; {len(feat_df[feat_df.selected_fraction_of_folds >= 0.5])} features selected in >=50% of folds.", ""]
    lines.append(compact_cmp.round(3).to_markdown(index=False))
    lines += ["", "Top compact features:", ""]
    lines.append(feat_df[feat_df.selected_fraction_of_folds >= 0.5].head(15)[["clinical_name", "selected_fraction_of_folds"]].round(2).to_markdown(index=False))
    lines += ["", "## 5. XGBoost / LightGBM added", ""]
    lines.append(boosted.round(3).to_markdown(index=False))
    DOC_PATH.write_text("\n".join(lines))
    print(f"\nSaved -> {DOC_PATH}")


if __name__ == "__main__":
    main()
