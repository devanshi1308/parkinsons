"""
43_4class_accuracy_push.py  (revised)
4-class (Healthy/Mild/Moderate/Severe) feature-family ablation + nested-CV de-biased estimate.

Changes vs. the first draft:
  1. Feature families are derived from the real per-version feature files (features.csv = v1,
     features_v2.csv = v2, features_v3_dfa.csv = v3 + DFA) instead of keyword guessing, which
     had v1 swallowing v2 columns and v3 missing spectral/wavelet/demographic features.
  2. Part B nests the FEATURE-SET choice together with the classifier choice, so the
     reported number is not optimistic from Part A's max-over-configs selection.
  3. Both parts repeat over several seeds (CV fold sd is ~10 pts at n=166 subjects, so a
     single split cannot separate feature sets).
  4. Adds reference rows: demographics+speed only, and a majority-class baseline.

Run from project root:  python scripts/43_4class_accuracy_push.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
SEEDS = [42, 43, 44, 45, 46]
N_OUTER, N_INNER = 5, 4
P = Path("data/processed")
IDS = {"filename", "cohort", "group", "subject_id", "trial", "stage"}
DEMO_SPEED = ["age", "sex_male", "height", "weight", "bmi", "speed_mps"]


def load():
    v3 = pd.read_csv(P / "features_v3_dfa.csv") if (P / "features_v3_dfa.csv").exists() else pd.read_csv(P / "features_v3.csv")
    lab = pd.read_csv(P / "subject_labels.csv")[["subject_id", "stage"]]
    df = v3.merge(lab, on="subject_id", how="inner").dropna(subset=["stage"]).reset_index(drop=True)
    cols = set(df.columns) - IDS
    f1 = set(pd.read_csv(P / "features.csv").columns) - IDS
    f2 = set(pd.read_csv(P / "features_v2.csv").columns) - IDS
    dfa = {c for c in cols if c.startswith("dfa_")}
    V1 = sorted(f1 & cols)
    V2 = sorted(((f2 & cols) - set(V1)))
    V3 = sorted(cols - set(V1) - set(V2) - dfa)
    sets = {"demo_speed_only": [c for c in DEMO_SPEED if c in cols],
            "v1": V1, "v1+v2": V1 + V2, "v1+v2+v3": V1 + V2 + V3}
    if dfa:
        sets["v1+v2+v3+dfa"] = V1 + V2 + V3 + sorted(dfa)
    return df, sets


CLASSIFIERS = {
    "DecisionTree": lambda s: DecisionTreeClassifier(random_state=s, class_weight="balanced"),
    "SVM": lambda s: SVC(kernel="linear", random_state=s, class_weight="balanced"),
    "Bagging": lambda s: BaggingClassifier(n_estimators=30, random_state=s),
    "NaiveBayes": lambda s: GaussianNB(),
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=200, random_state=s, class_weight="balanced", n_jobs=-1),
    "HistGB": lambda s: HistGradientBoostingClassifier(random_state=s),
    "LogReg": lambda s: LogisticRegression(max_iter=2000, random_state=s, class_weight="balanced"),
}


def fit_eval(clf_fn, seed, Xtr, ytr, Xte, yte):
    # imputer + scaler fit on the training fold only (height/weight/bmi/speed have a few NaNs)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    prep = lambda X: sc.transform(imp.transform(X))
    clf = clf_fn(seed).fit(prep(Xtr), ytr)
    p = clf.predict(prep(Xte))
    return accuracy_score(yte, p), f1_score(yte, p, average="macro", zero_division=0)


def part_a(df, sets):
    print("=" * 78 + "\nPART A: feature-family ablation (4-class, subject-grouped, %d seeds x %d folds)\n" % (len(SEEDS), N_OUTER) + "=" * 78)
    y, g = df["stage"].values, df["subject_id"].values
    rows = []
    for sname, cols in sets.items():
        X = df[cols].values.astype(float)
        for cname, cfn in CLASSIFIERS.items():
            for seed in SEEDS:
                cv = StratifiedGroupKFold(N_OUTER, shuffle=True, random_state=seed)
                a, f = zip(*[fit_eval(cfn, seed, X[tr], y[tr], X[te], y[te]) for tr, te in cv.split(X, y, g)])
                rows.append(dict(feature_set=sname, n_features=len(cols), classifier=cname, seed=seed,
                                 acc=np.mean(a), f1_macro=np.mean(f)))
    res = pd.DataFrame(rows)
    res.to_csv(P / "ablation_4class_runs.csv", index=False)
    summ = (res.groupby(["feature_set", "n_features", "classifier"])
               .agg(acc_mean=("acc", "mean"), acc_seed_sd=("acc", "std"),
                    f1_macro_mean=("f1_macro", "mean")).reset_index())
    summ.to_csv(P / "ablation_4class_results.csv", index=False)
    order = list(sets)
    by_set = summ.groupby("feature_set").agg(best_acc=("acc_mean", "max"), mean_over_clf=("acc_mean", "mean"),
                                             best_f1=("f1_macro_mean", "max")).loc[order]
    print(by_set.round(3).to_string())
    # paired DFA effect
    if "v1+v2+v3+dfa" in sets:
        piv = res.pivot_table(index=["classifier", "seed"], columns="feature_set", values="acc")
        d = (piv["v1+v2+v3+dfa"] - piv["v1+v2+v3"])
        print("\nDFA effect (v1+v2+v3+dfa minus v1+v2+v3), paired over classifier x seed:")
        print(d.groupby("classifier").mean().round(4).to_string())
        print(f"overall mean delta {d.mean():+.4f} acc; positive in {(d>0).sum()}/{len(d)} pairs")
    return res


def part_b(df, sets):
    print("\n" + "=" * 78 + "\nPART B: nested CV, feature set AND classifier chosen in the inner loop\n" + "=" * 78)
    y, g = df["stage"].values, df["subject_id"].values
    Xs = {k: df[c].values.astype(float) for k, c in sets.items()}
    cand = [(s, c) for s in sets for c in CLASSIFIERS]
    accs, f1s, picks = [], [], []
    for seed in SEEDS:
        outer = StratifiedGroupKFold(N_OUTER, shuffle=True, random_state=seed)
        sa, sf = [], []
        for tr, te in outer.split(df, y, g):
            inner = StratifiedGroupKFold(N_INNER, shuffle=True, random_state=seed)
            score = {k: [] for k in cand}
            for itr, iva in inner.split(tr, y[tr], g[tr]):
                a_idx, v_idx = tr[itr], tr[iva]
                for (s, c) in cand:
                    score[(s, c)].append(fit_eval(CLASSIFIERS[c], seed, Xs[s][a_idx], y[a_idx], Xs[s][v_idx], y[v_idx])[1])
            best = max(cand, key=lambda k: np.mean(score[k]))
            picks.append(best)
            a, f = fit_eval(CLASSIFIERS[best[1]], seed, Xs[best[0]][tr], y[tr], Xs[best[0]][te], y[te])
            sa.append(a); sf.append(f)
        accs.append(np.mean(sa)); f1s.append(np.mean(sf))
        print(f"  seed {seed}: nested acc {accs[-1]:.3f}  macro-F1 {f1s[-1]:.3f}")
    pk = pd.Series([f"{s} + {c}" for s, c in picks]).value_counts()
    print("\nMost-selected (feature set + classifier) across all outer folds:\n" + pk.head(8).to_string())
    print("\nFeature-set selection frequency:\n" + pd.Series([s for s, _ in picks]).value_counts().to_string())
    out = dict(acc_mean=np.mean(accs), acc_sd_across_seeds=np.std(accs, ddof=1), f1_mean=np.mean(f1s),
               f1_sd_across_seeds=np.std(f1s, ddof=1))
    print(f"\nDE-BIASED 4-class: acc {out['acc_mean']:.3f} (sd across seeds {out['acc_sd_across_seeds']:.3f}), "
          f"macro-F1 {out['f1_mean']:.3f} (sd {out['f1_sd_across_seeds']:.3f})")
    pd.DataFrame([out]).to_csv(P / "nested_4class_results.csv", index=False)
    pk.rename("n_folds_selected").to_csv(P / "nested_4class_selection_counts.csv")
    return out


def main():
    df, sets = load()
    print(f"{len(df)} trials, {df['subject_id'].nunique()} subjects")
    vc = df["stage"].value_counts()
    print(vc.to_string())
    print(f"Majority-class (always '{vc.index[0]}') trial-level accuracy: {vc.iloc[0]/len(df):.3f}")
    for k, v in sets.items():
        print(f"  {k:16s} {len(v)} features")
    part_a(df, sets)
    part_b(df, sets)


if __name__ == "__main__":
    main()
