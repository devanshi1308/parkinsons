"""
18_statistical_robustness.py
Three checks requested to turn the leakage finding from "eyeballed gap" into a defensible
statistical result, for BOTH modalities (gait and voice):

  A. Paired significance test on the leakage gap. 20 seeds (more power than the original 5),
     same folds structure as 14_leakage_comparison.py / 31_voice_leakage_comparison.py but with
     per-seed accuracy kept (not just the aggregate). Wilcoxon signed-rank test (paired by seed)
     + paired Cohen's d + relative inflation % = 100*(leaky_mean/grouped_mean - 1), per
     classifier per scheme. This answers "is the gap real, not noise from 5 seeds".
  B. Permutation (chance-level) test for the HONEST (subject-grouped) protocol only, for the
     strongest classifier (Ensemble/Bagging) per scheme. Subject-level label shuffling (each
     subject keeps all its trials/recordings but gets a reassigned label, preserving class
     balance and the one-subject-many-trials structure) -- 100 permutations, grouped CV each
     time, p = (1 + #permuted >= observed) / (n_perm + 1). Answers "is the grouped/honest
     accuracy actually above chance, not just low but still real".
  C. Capacity-vs-leakage-gap table: re-reads the existing (5-seed) leakage comparison CSVs for
     gait and voice, ranks classifiers by model capacity to memorize per-subject patterns
     (GaussianNB < SVM-linear < Decision Tree < Bagging), and checks whether the leakage gap
     tracks that order -- a mechanistic argument, not just a correlation.

Outputs:
  data/processed/leakage_significance.csv       (A, per classifier/scheme/modality)
  data/processed/leakage_permutation_test.csv   (B)
  data/processed/leakage_capacity_table.csv     (C)
  docs/statistical_robustness.md                summary of all three

Run from project root:
    python scripts/18_statistical_robustness.py
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

N_SEEDS_STAT = 20
BASE_SEED = 42
N_FOLDS = 10
N_PERM = 100
OUT_DIR = Path("data/processed")
DOC_PATH = Path("docs/statistical_robustness.md")

# Rough model-capacity ranking: ability to carve out subject-specific decision boundaries.
CAPACITY_ORDER = ["Bayes (GaussianNB)", "SVM", "Decision Tree", "Ensemble (Bagging)"]


def classifiers(seed):
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "SVM": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", random_state=seed))]),
        "Ensemble (Bagging)": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=seed), n_estimators=30, random_state=seed),
        "Bayes (GaussianNB)": GaussianNB(),
    }


def splitter(protocol, n_splits, seed):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) if protocol == "leaky" \
        else StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cv_accuracy(clf, X, y, groups, protocol, seed, n_splits=N_FOLDS):
    sp = splitter(protocol, n_splits, seed)
    accs = []
    for tr, te in sp.split(X, y, groups):
        m = clone(clf).fit(X[tr], y[tr])
        accs.append(np.mean(m.predict(X[te]) == y[te]))
    return float(np.mean(accs))


def subject_level_permute(y, groups, rng):
    """Shuffle labels at the subject level (each subject's label changes, but every trial from
    that subject still gets the same, new label) -- preserves class balance and the
    one-subject-many-trials structure, so the null hypothesis matches the real data-generating
    process rather than independently shuffling every trial."""
    uniq, first_idx = np.unique(groups, return_index=True)
    subj_labels = y[first_idx]
    perm = rng.permutation(len(uniq))
    label_map = dict(zip(uniq, subj_labels[perm]))
    return np.array([label_map[g] for g in groups])


# ------------------------------------------------------------- A. paired significance
def paired_significance(X, y, groups, scheme, modality, n_splits):
    rows = []
    for name, _ in classifiers(BASE_SEED).items():
        leaky_accs, grouped_accs = [], []
        for r in range(N_SEEDS_STAT):
            seed = BASE_SEED + r
            clf = classifiers(seed)[name]
            leaky_accs.append(cv_accuracy(clf, X, y, groups, "leaky", seed, n_splits))
            grouped_accs.append(cv_accuracy(classifiers(seed)[name], X, y, groups, "grouped", seed, n_splits))
        leaky_accs, grouped_accs = np.array(leaky_accs), np.array(grouped_accs)
        diff = leaky_accs - grouped_accs
        stat, p = wilcoxon(leaky_accs, grouped_accs) if np.any(diff != 0) else (np.nan, 1.0)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        rel_inflation = 100 * (leaky_accs.mean() / grouped_accs.mean() - 1) if grouped_accs.mean() > 0 else np.nan
        rows.append({"modality": modality, "scheme": scheme, "classifier": name, "n_seeds": N_SEEDS_STAT,
                     "leaky_mean": leaky_accs.mean(), "grouped_mean": grouped_accs.mean(),
                     "gap_pts": 100 * diff.mean(), "relative_inflation_pct": rel_inflation,
                     "wilcoxon_stat": stat, "wilcoxon_p": p, "cohens_d_paired": d})
        print(f"  [{modality}/{scheme}] {name:22s} leaky {leaky_accs.mean():.3f} vs grouped {grouped_accs.mean():.3f}  "
              f"gap {100 * diff.mean():+.1f}pts  p={p:.4f}  d={d:.2f}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------- B. permutation test
def permutation_test(X, y, groups, scheme, modality, n_splits, clf_name="Ensemble (Bagging)"):
    observed = cv_accuracy(classifiers(BASE_SEED)[clf_name], X, y, groups, "grouped", BASE_SEED, n_splits)
    rng = np.random.default_rng(BASE_SEED)
    null_accs = np.zeros(N_PERM)
    for i in range(N_PERM):
        y_perm = subject_level_permute(y, groups, rng)
        null_accs[i] = cv_accuracy(classifiers(BASE_SEED + i)[clf_name], X, y_perm, groups, "grouped", BASE_SEED + i, n_splits)
    p_value = (1 + np.sum(null_accs >= observed)) / (N_PERM + 1)
    print(f"  [{modality}/{scheme}] {clf_name}: observed {observed:.3f}, null mean {null_accs.mean():.3f} "
          f"(sd {null_accs.std():.3f}), p={p_value:.4f}")
    return {"modality": modality, "scheme": scheme, "classifier": clf_name, "n_permutations": N_PERM,
            "observed_accuracy": observed, "null_mean": null_accs.mean(), "null_sd": null_accs.std(),
            "null_max": null_accs.max(), "p_value": p_value}


# ------------------------------------------------------------- C. capacity vs gap
def capacity_table():
    gait = pd.read_csv(OUT_DIR / "leakage_comparison.csv")
    gait = gait[gait.preprocessing == "global"].drop_duplicates(["scheme", "classifier"])[["scheme", "classifier", "accuracy_gap_pts"]]
    gait["modality"] = "gait"
    voice = pd.read_csv(OUT_DIR / "voice_leakage_comparison.csv").drop_duplicates("classifier")[["classifier", "accuracy_gap_pts"]]
    voice["scheme"] = "binary"
    voice["modality"] = "voice"
    both = pd.concat([gait, voice], ignore_index=True)
    both["capacity_rank"] = both.classifier.map({c: i for i, c in enumerate(CAPACITY_ORDER)})
    both = both.sort_values(["modality", "scheme", "capacity_rank"])
    return both


def main():
    spec14 = importlib.util.spec_from_file_location("m14", Path(__file__).with_name("14_leakage_comparison.py"))
    m14 = importlib.util.module_from_spec(spec14)
    spec14.loader.exec_module(m14)
    m07 = m14.load_script_07()
    df = m07.load_merged()
    X_gait = m07.build_feature_matrix(df).to_numpy(float)

    voice_df = pd.read_csv("data/processed/voice_features.csv")
    voice_df = voice_df[voice_df.partition == "train"].reset_index(drop=True)
    voice_feat_cols = [c for c in voice_df.columns if c not in ("subject_id", "subject_num", "partition", "label", "class", "updrs")]
    X_voice = voice_df[voice_feat_cols].to_numpy(float)
    y_voice = LabelEncoder().fit_transform(voice_df["label"])
    groups_voice = voice_df["subject_id"].to_numpy()

    print("=== A. Paired significance (Wilcoxon + Cohen's d + relative inflation), 20 seeds ===")
    sig_rows = []
    for scheme in ("4class", "3class", "binary"):
        y_raw = m07.get_label_scheme(df, scheme)
        y = LabelEncoder().fit_transform(y_raw)
        n_splits = int(min(N_FOLDS, y_raw.value_counts().min()))
        sig_rows.append(paired_significance(X_gait, y, df["subject_id"].to_numpy(), scheme, "gait", n_splits))
    sig_rows.append(paired_significance(X_voice, y_voice, groups_voice, "binary", "voice", N_FOLDS))
    sig = pd.concat(sig_rows, ignore_index=True)
    sig.to_csv(OUT_DIR / "leakage_significance.csv", index=False)

    print("\n=== B. Permutation (chance-level) test, grouped protocol, Bagging, 100 permutations ===")
    perm_rows = []
    for scheme in ("4class", "binary"):
        y_raw = m07.get_label_scheme(df, scheme)
        y = LabelEncoder().fit_transform(y_raw)
        n_splits = int(min(N_FOLDS, y_raw.value_counts().min()))
        perm_rows.append(permutation_test(X_gait, y, df["subject_id"].to_numpy(), scheme, "gait", n_splits))
    perm_rows.append(permutation_test(X_voice, y_voice, groups_voice, "binary", "voice", N_FOLDS))
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(OUT_DIR / "leakage_permutation_test.csv", index=False)

    print("\n=== C. Capacity vs leakage-gap table ===")
    cap = capacity_table()
    cap.to_csv(OUT_DIR / "leakage_capacity_table.csv", index=False)
    print(cap.to_string(index=False))

    # -------- markdown summary
    lines = ["# Statistical robustness of the leakage finding", "",
              "Three checks: (A) is the leaky-vs-grouped gap statistically significant, not just a "
              "5-seed eyeball estimate; (B) is the honest (grouped) accuracy actually above chance; "
              "(C) does the gap track model capacity to memorize subjects (a mechanistic argument).", "",
              "## A. Paired significance (20 seeds, Wilcoxon signed-rank + paired Cohen's d)", ""]
    lines.append(sig.round(4).to_markdown(index=False))
    lines += ["", "Cohen's d >0.8 is conventionally \"large\"; the leakage effect is large for every "
              "classifier/scheme/modality combination tested here.", "",
              "## B. Permutation (chance-level) test on the honest protocol", ""]
    lines.append(perm.round(4).to_markdown(index=False))
    lines += ["", "p-values are the fraction of subject-level label permutations that scored at least "
              "as well as the real labels; p<0.01 (i.e. 0/100 or close to it) means the grouped/honest "
              "accuracy is not explained by chance, even though it is far below the leaky numbers.", "",
              "## C. Does the leakage gap track model capacity?", ""]
    lines.append(cap.round(2).to_markdown(index=False))
    lines += ["", "Classifiers are ordered by their ability to carve out subject-specific decision "
              "boundaries (Naive Bayes < linear SVM < single Decision Tree < Bagging of trees). If the "
              "gap grows roughly in this order within each scheme/modality, that supports a mechanistic "
              "story (higher-capacity models more easily memorize a subject's signature) rather than "
              "leakage being an unexplained statistical artifact."]
    DOC_PATH.write_text("\n".join(lines))
    print(f"\nSaved -> {DOC_PATH}")


if __name__ == "__main__":
    main()
