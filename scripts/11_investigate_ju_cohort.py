"""
11_investigate_ju_cohort.py
Digs into WHY the Ju cohort underperforms relative to Ga and Si. Checks four
candidate explanations:
  1. Class imbalance within the cohort (does Ju have a harder stage mix?)
  2. Demographic differences (age/weight/height) that might affect VGRF signal
     characteristics independent of PD status
  3. Trial length / recording duration differences (shorter trials = less
     signal to extract stable features from)
  4. Raw signal-level differences (mean sensor force magnitude, noise level)
     that might indicate different equipment/calibration

Run from project root:
    python scripts/11_investigate_ju_cohort.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
FEATURES_PATH = Path("data/processed/features_v2.csv")
LABELS_PATH = Path("data/processed/subject_labels.csv")


def check_class_balance(df):
    print("=== 1. Class balance per cohort ===")
    balance = pd.crosstab(df["cohort"], df["stage"], normalize="index") * 100
    print(balance.round(1))
    print()


def check_demographics(df_features):
    print("=== 2. Demographics per cohort ===")
    demo_path = RAW_DIR / "demographics.xls"
    if not demo_path.exists():
        demo_path = RAW_DIR / "demographics.xlsx"
    demo = pd.read_excel(demo_path)

    # BUGFIX (2026-10): demo['Group'] is 'PD'/'CO', but subject_id (and the raw filenames) use
    # 'Pt'/'Co'. The original f-string built IDs like "GaPD03", which never matched anything, so
    # this merge silently produced all-NaN demographics. demo['ID'] already has the right format
    # (e.g. "GaPt03") and should be used directly instead of reconstructing it.
    demo["subject_id"] = demo["ID"]

    merged = df_features[["subject_id", "cohort"]].drop_duplicates().merge(demo, on="subject_id", how="left")

    numeric_cols = [c for c in ["Age", "Height (meters)", "Weight (kg)"] if c in demo.columns]
    summary = merged.groupby("cohort")[numeric_cols].agg(["mean", "std"])
    print(summary.round(2))
    print()


def check_trial_length(df_features):
    print("=== 3. Trial length / signal duration per cohort ===")
    rows = []
    for _, row in df_features.iterrows():
        filepath = INTERIM_DIR / row["cohort"] / row["group"] / row["filename"]
        if filepath.exists():
            d = pd.read_csv(filepath)
            duration = d["Time"].iloc[-1] - d["Time"].iloc[0]
            rows.append({"cohort": row["cohort"], "duration_s": duration, "n_rows": len(d)})

    length_df = pd.DataFrame(rows)
    print(length_df.groupby("cohort")[["duration_s", "n_rows"]].agg(["mean", "std", "min", "max"]).round(1))
    print()


def check_signal_characteristics(df_features):
    print("=== 4. Raw signal characteristics per cohort (mean sensor force, noise) ===")
    # Use already-computed mean/SD features as a proxy — average across all sensors
    mean_cols = [c for c in df_features.columns if c.endswith("_mean") and (c.startswith("L") or c.startswith("R"))]
    sd_cols = [c for c in df_features.columns if c.endswith("_sd") and (c.startswith("L") or c.startswith("R"))]

    df_features["avg_sensor_mean"] = df_features[mean_cols].mean(axis=1)
    df_features["avg_sensor_sd"] = df_features[sd_cols].mean(axis=1)

    print(df_features.groupby("cohort")[["avg_sensor_mean", "avg_sensor_sd"]].agg(["mean", "std"]).round(2))
    print()


def check_missing_kinematic_features(df_features):
    print("=== 5. Missing/NaN rate for key kinematic features per cohort ===")
    key_cols = [c for c in ["L_stance_time", "R_stance_time", "L_stride_time", "R_stride_time",
                              "cadence_steps_per_min", "L_cop_path_length", "R_cop_path_length"]
                if c in df_features.columns]
    nan_rates = df_features.groupby("cohort")[key_cols].apply(lambda g: g.isna().mean() * 100)
    print(nan_rates.round(1))
    print("(% of trials where this feature couldn't be computed — high values suggest")
    print(" stance-detection issues, possibly from different equipment/force ranges)\n")


def main():
    df = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    df = df.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"])

    check_class_balance(df)
    check_demographics(df)
    check_trial_length(df)
    check_signal_characteristics(df)
    check_missing_kinematic_features(df)


if __name__ == "__main__":
    main()
