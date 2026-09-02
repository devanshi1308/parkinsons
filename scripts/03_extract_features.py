"""
03_extract_features.py
Extracts the features used in Balaji et al. (2020):
  - Statistical: mean, median, SD per sensor (16 sensors)
  - Kinematic/spatiotemporal: step time, stance time, swing time, swing/stance ratio,
    stride time, cadence, speed, step length (approximated), stride length (approximated)

VGRF-only dataset does NOT include direct spatial (distance) measurements — step length
and stride length in the original paper come from the same lab's separate kinematic
capture. Here we approximate temporal features directly from VGRF stance/swing detection
(thresholding total force), which is standard practice in the gait-analysis literature.
Spatial features (step/stride length) are left as NaN unless you have supplementary
distance data — flag this honestly in your methodology section.

Run from project root:
    python scripts/03_extract_features.py
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

INTERIM_DIR = Path("data/interim")
OUT_PATH = Path("data/processed/features.csv")

FORCE_THRESHOLD_N = 20  # below this = foot is in swing (off ground); above = stance
SAMPLE_RATE_HZ = 100

FILENAME_PATTERN = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$", re.IGNORECASE)


def statistical_features(df: pd.DataFrame) -> dict:
    sensor_cols = [c for c in df.columns if c.startswith("L") or c.startswith("R")]
    feats = {}
    for col in sensor_cols:
        feats[f"{col}_mean"] = df[col].mean()
        feats[f"{col}_median"] = df[col].median()
        feats[f"{col}_sd"] = df[col].std()
    return feats


def detect_stance_phases(force_series: pd.Series) -> list:
    """Returns list of (start_idx, end_idx) tuples where foot is in stance (force > threshold)."""
    in_stance = force_series > FORCE_THRESHOLD_N
    phases = []
    start = None
    for i, val in enumerate(in_stance):
        if val and start is None:
            start = i
        elif not val and start is not None:
            phases.append((start, i))
            start = None
    if start is not None:
        phases.append((start, len(in_stance)))
    return phases


def kinematic_features(df: pd.DataFrame) -> dict:
    feats = {}

    for foot, col in [("L", "TotalL"), ("R", "TotalR")]:
        stance_phases = detect_stance_phases(df[col])
        if len(stance_phases) < 2:
            feats[f"{foot}_stance_time"] = np.nan
            feats[f"{foot}_swing_time"] = np.nan
            feats[f"{foot}_stride_time"] = np.nan
            continue

        stance_durations = [(e - s) / SAMPLE_RATE_HZ for s, e in stance_phases]
        swing_durations = [
            (stance_phases[i + 1][0] - stance_phases[i][1]) / SAMPLE_RATE_HZ
            for i in range(len(stance_phases) - 1)
        ]
        stride_durations = [
            (stance_phases[i + 1][0] - stance_phases[i][0]) / SAMPLE_RATE_HZ
            for i in range(len(stance_phases) - 1)
        ]

        feats[f"{foot}_stance_time"] = np.mean(stance_durations)
        feats[f"{foot}_swing_time"] = np.mean(swing_durations) if swing_durations else np.nan
        feats[f"{foot}_stride_time"] = np.mean(stride_durations) if stride_durations else np.nan

    # combine both feet for step time / cadence
    total_duration_s = df["Time"].iloc[-1] - df["Time"].iloc[0]
    n_stance_events = len(detect_stance_phases(df["TotalL"])) + len(detect_stance_phases(df["TotalR"]))
    feats["cadence_steps_per_min"] = (n_stance_events / total_duration_s) * 60 if total_duration_s > 0 else np.nan

    l_stride = feats.get("L_stride_time", np.nan)
    r_stride = feats.get("R_stride_time", np.nan)
    avg_stride_time = np.nanmean([l_stride, r_stride])
    feats["step_time"] = avg_stride_time / 2 if not np.isnan(avg_stride_time) else np.nan

    l_stance, l_swing = feats.get("L_stance_time", np.nan), feats.get("L_swing_time", np.nan)
    feats["swing_stance_ratio"] = l_swing / l_stance if l_stance and not np.isnan(l_stance) else np.nan

    # NOTE: step length / stride length / speed require spatial (distance) data
    # not present in this VGRF-only dataset. Left as NaN — document this limitation.
    feats["step_length_m"] = np.nan
    feats["stride_length_m"] = np.nan
    feats["speed_m_s"] = np.nan

    return feats


def main():
    files = list(INTERIM_DIR.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No cleaned files in {INTERIM_DIR.resolve()} — run 02_clean_signals.py first.")

    rows = []
    for f in files:
        m = FILENAME_PATTERN.match(f.name)
        if not m:
            continue
        cohort, group, subj_id, trial = m.groups()

        df = pd.read_csv(f)
        row = {
            "filename": f.name,
            "cohort": cohort,
            "group": group,  # Co = control, Pt = patient
            "subject_id": f"{cohort}{group}{subj_id}",
            "trial": trial,
        }
        row.update(statistical_features(df))
        row.update(kinematic_features(df))
        rows.append(row)

    result = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"Extracted features for {len(result)} trials -> {OUT_PATH.resolve()}")
    print(f"Columns: {result.shape[1]}")


if __name__ == "__main__":
    main()
