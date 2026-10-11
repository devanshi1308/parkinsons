"""
23_generalization_calibration_clinical.py
Three checks on the improved gait models (features v3) that earlier scripts left open:

  A. Leave-one-cohort-out (LOCO) generalization. Train on two of the three independently
     recruited sub-cohorts (Ga/Ju/Si), test on the third. Compared against the majority-class
     baseline of the held-out cohort (Ju is 80.6% PD, so "always predict PD" scores 80.6% there)
     and against a demographics+speed-only model, so any gait-signal contribution is judged
     against what is available without the gait signal. Feature sets: v3_all, compact-5
     (from 16_model_improvements.py section 4), demo_speed_only.
  B. Probability calibration of the binary "chance of PD" output. Out-of-fold, subject-grouped
     probabilities from Random Forest and LightGBM, raw vs Platt (sigmoid) vs isotonic
     calibration (calibrator fit on training folds only, with grouped inner CV). Brier score,
     log-loss, expected calibration error (10 equal-width bins) and AUC; reliability diagram.
  C. Clinical metrics with fold variance for the binary model: sensitivity, specificity, PPV,
     NPV, AUC, accuracy (mean and SD across 10 subject-grouped folds x 3 seeds) for LightGBM and
     Random Forest on v3_all. CLAUDE.md hard rule 4 requires these for the final model.
     NOTE: C evaluates a FIXED configuration (LightGBM on v3_all, the modal winner of the nested
     selection in 17_nested_with_boosting.py), so it carries no selection bias from picking among
     candidates, but it is not itself a nested estimate; the nested accuracy (81.7%, sd 10.6)
     remains the number to cite for accuracy.

Outputs:
  data/processed/loco_results.csv, calibration_results.csv, binary_clinical_metrics.csv
  docs/figures/calibration_reliability.png
  docs/generalization_calibration_clinical.md

Run from project root:
    python scripts/23_generalization_calibration_clinical.py
"""

import importlib.util
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, f1_score, log_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

SEED = 42
OUT_DIR = Path("data/processed")
FIG_PATH = Path("docs/figures/calibration_reliability.png")
DOC_PATH = Path("docs/generalization_calibration_clinical.md")
SCHEMES = ("binary", "4class")


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data(m07, m12):
    feats = pd.read_csv("data/processed/features_v3.csv")
    labels = pd.read_csv("data/processed/subject_labels.csv")
    groups_map = pd.read_csv("data/processed/feature_groups_v3.csv")
    df = feats.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"]).reset_index(drop=True)

    def cols_for(fset):
        fam = m12.FEATURE_SETS[fset]
        return [c for c in groups_map[groups_map.group.isin(fam)].feature if c in df.columns and not df[c].isna().all()]

    compact = pd.read_csv("data/processed/compact_feature_list.csv")
    compact_cols = compact[compact.selected_fraction_of_folds >= 0.5].feature.tolist()
    feature_sets = {"v3_all": cols_for("v3_all"), "compact5": compact_cols, "demo_speed_only": cols_for("demo_speed_only")}
    return df, feature_sets


def model_zoo(m12, m17, seed=SEED):
    sk = m12.make_models(seed, m12.CorrelationFilter)
    return {"Random Forest": sk["Random Forest"], "LightGBM": m17.boosted_pipelines(seed)["LightGBM"]}


# ------------------------------------------------------------------ A. LOCO
def loco(df, feature_sets, m07, m12, m17):
    rows = []
    cohorts = sorted(df.cohort.unique())
    for scheme in SCHEMES:
        y_raw = m07.get_label_scheme(df, scheme)
        y = LabelEncoder().fit_transform(y_raw)
        for held in cohorts:
            te = (df.cohort == held).to_numpy()
            tr = ~te
            majority = max(np.mean(y[te] == c) for c in np.unique(y[te]))
            for fset, cols in feature_sets.items():
                X = df[cols].to_numpy(float)
                for name, model in model_zoo(m12, m17).items():
                    m = clone(model).fit(X[tr], y[tr])
                    pred = m.predict(X[te])
                    row = {"scheme": scheme, "held_out_cohort": held, "n_test": int(te.sum()), "feature_set": fset,
                           "model": name, "accuracy": np.mean(pred == y[te]), "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                           "f1_macro": f1_score(y[te], pred, average="macro", zero_division=0), "majority_baseline": majority}
                    if scheme == "binary":
                        row["auc"] = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
                    rows.append(row)
        print(f"  LOCO {scheme} done")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B. calibration
def expected_calibration_error(y, p, n_bins=10):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return ece


def out_of_fold_proba(model, X, y, groups, method, seed=SEED, n_folds=10):
    prob = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        if method == "raw":
            m = clone(model).fit(X[tr], y[tr])
        else:
            inner = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X[tr], y[tr], groups[tr]))
            m = CalibratedClassifierCV(clone(model), method=method, cv=inner).fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
    return prob


def calibration(df, feature_sets, m07, m12, m17):
    y = LabelEncoder().fit_transform(m07.get_label_scheme(df, "binary"))
    X = df[feature_sets["v3_all"]].to_numpy(float)
    groups = df["subject_id"].to_numpy()
    rows, curves = [], {}
    for name, model in model_zoo(m12, m17).items():
        for method in ("raw", "sigmoid", "isotonic"):
            p = np.clip(out_of_fold_proba(model, X, y, groups, method), 1e-4, 1 - 1e-4)
            rows.append({"model": name, "calibration": method, "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
                         "ece": expected_calibration_error(y, p), "auc": roc_auc_score(y, p), "accuracy": np.mean((p >= 0.5) == y)})
            curves[(name, method)] = (y, p)
            print(f"  calibration {name}/{method}: brier {rows[-1]['brier']:.3f} ece {rows[-1]['ece']:.3f}")
    return pd.DataFrame(rows), curves


def reliability_figure(curves):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    colors = {"raw": "#c0504d", "sigmoid": "#4f81bd", "isotonic": "#2a9d8f"}
    for ax, model in zip(axes, ("Random Forest", "LightGBM")):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
        for method in ("raw", "sigmoid", "isotonic"):
            y, p = curves[(model, method)]
            bins = np.minimum((p * 8).astype(int), 7)
            xs, ys = [], []
            for b in range(8):
                mask = bins == b
                if mask.sum() >= 8:  # skip bins too small to estimate a frequency
                    xs.append(p[mask].mean()); ys.append(y[mask].mean())
            ax.plot(xs, ys, "o-", color=colors[method], label=method, lw=1.6, ms=5)
        ax.set_title(model, fontsize=10.5)
        ax.set_xlabel("Predicted probability of PD")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed fraction with PD")
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Reliability of the PD-risk output (out-of-fold, subject-grouped, 306 trials; bins with <8 trials omitted)", fontsize=10)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ C. clinical metrics
def clinical_metrics(df, feature_sets, m07, m12, m17, n_seeds=3, n_folds=10):
    y = LabelEncoder().fit_transform(m07.get_label_scheme(df, "binary"))  # Healthy=0, PD=1
    X = df[feature_sets["v3_all"]].to_numpy(float)
    groups = df["subject_id"].to_numpy()
    per_fold = []
    for name in ("LightGBM", "Random Forest"):
        for r in range(n_seeds):
            seed = SEED + r
            model = model_zoo(m12, m17, seed)[name]
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for tr, te in cv.split(X, y, groups):
                m = clone(model).fit(X[tr], y[tr])
                prob = m.predict_proba(X[te])[:, 1]
                pred = (prob >= 0.5).astype(int)
                tp = np.sum((pred == 1) & (y[te] == 1)); tn = np.sum((pred == 0) & (y[te] == 0))
                fp = np.sum((pred == 1) & (y[te] == 0)); fn = np.sum((pred == 0) & (y[te] == 1))
                per_fold.append({"model": name, "seed": seed,
                                 "sensitivity": tp / (tp + fn) if tp + fn else np.nan, "specificity": tn / (tn + fp) if tn + fp else np.nan,
                                 "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan,
                                 "accuracy": (tp + tn) / len(te), "auc": roc_auc_score(y[te], prob) if len(np.unique(y[te])) == 2 else np.nan})
        print(f"  clinical metrics {name} done")
    pf = pd.DataFrame(per_fold)
    metrics = ["sensitivity", "specificity", "ppv", "npv", "accuracy", "auc"]
    summary = pf.groupby("model")[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary.reset_index()


def main():
    m07, m12, m17 = load_module("07_train_multi_scheme.py", "m07"), load_module("12_train_models_v3.py", "m12"), load_module("17_nested_with_boosting.py", "m17")
    df, feature_sets = load_data(m07, m12)
    print(f"Trials {len(df)}, subjects {df.subject_id.nunique()}, feature sets: " + ", ".join(f"{k}={len(v)}" for k, v in feature_sets.items()))

    print("\n=== A. Leave-one-cohort-out ===")
    loco_df = loco(df, feature_sets, m07, m12, m17)
    loco_df.to_csv(OUT_DIR / "loco_results.csv", index=False)

    print("\n=== B. Calibration ===")
    calib_df, curves = calibration(df, feature_sets, m07, m12, m17)
    calib_df.to_csv(OUT_DIR / "calibration_results.csv", index=False)
    reliability_figure(curves)

    print("\n=== C. Clinical metrics with fold variance (binary) ===")
    clin = clinical_metrics(df, feature_sets, m07, m12, m17)
    clin.to_csv(OUT_DIR / "binary_clinical_metrics.csv", index=False)
    print(clin.round(3).T.to_string())

    # ---- markdown
    lines = ["# Generalization, calibration and clinical metrics (gait, features v3)", "",
             "Three checks left open by earlier scripts. Subject-grouped throughout. See the docstring in "
             "`scripts/23_generalization_calibration_clinical.py` for exact protocols.", "",
             "## A. Leave-one-cohort-out", "",
             "Train on two sub-cohorts, test on the third. `majority_baseline` is the accuracy of always predicting the held-out "
             "cohort's most common class, and `demo_speed_only` is a model with no gait-signal features at all.", ""]
    for scheme in SCHEMES:
        sub = loco_df[loco_df.scheme == scheme]
        cols = ["held_out_cohort", "n_test", "feature_set", "model", "accuracy", "balanced_accuracy", "majority_baseline"] + (["auc"] if scheme == "binary" else [])
        lines += [f"### {scheme}", "", sub[cols].round(3).to_markdown(index=False), ""]
    lines += ["## B. Calibration of the binary PD-risk output", "", calib_df.round(4).to_markdown(index=False), "",
              "Lower Brier / log-loss / ECE is better. Figure: `docs/figures/calibration_reliability.png`.", "",
              "## C. Binary clinical metrics, mean and SD across 10 folds x 3 seeds (threshold 0.5)", "",
              clin.round(3).to_markdown(index=False), ""]
    DOC_PATH.write_text("\n".join(lines))
    print(f"\nSaved -> {DOC_PATH}")


if __name__ == "__main__":
    main()
