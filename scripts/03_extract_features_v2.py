"""
03_extract_features_v2.py
Improved feature extraction over v1:
  1. Hysteresis-based stance/swing detection (two thresholds) instead of a single
     cutoff — reduces false stance/swing flips from sensor noise near the threshold.
  2. Center of pressure (COP) features computed from sensor X-Y geometry (Table 2
     in Balaji et al. 2020) — real spatial information the paper's pure-VGRF
     statistical/kinematic split doesn't use.
  3. Gait variability features (coefficient of variation of stride/step time) —
     a well-established PD biomarker (increased stride-time variability is a
     known PD gait signature), not used in the baseline paper.
  4. Left-right symmetry index — PD gait is often asymmetric; healthy gait is
     more symmetric.
  5. Extended statistical features: skew, kurtosis, min, max, range, IQR
     (baseline paper only used mean/median/SD).

Run from project root:
    python scripts/03_extract_features_v2.py
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

INTERIM_DIR = Path("data/interim")
OUT_PATH = Path("data/processed/features_v2.csv")

SAMPLE_RATE_HZ = 100
FILENAME_PATTERN = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$", re.IGNORECASE)

# Hysteresis thresholds for stance detection: enter stance above HIGH, exit below LOW.
# Prevents rapid flip-flopping when force hovers near a single cutoff.
THRESH_HIGH_N = 25
THRESH_LOW_N = 15

# Sensor X-Y positions in mm, from Table 2 of Balaji et al. (2020).
# Origin (0,0) at each foot, leg facing +Y. Used to compute center of pressure.
SENSOR_POSITIONS = {
    "L1": (-500, -800), "L2": (-700, -400), "L3": (-300, -400), "L4": (-700, 0),
    "L5": (-300, 0),    "L6": (-700, 400),  "L7": (-300, 400),  "L8": (-500, 800),
    "R1": (500, -800),  "R2": (700, -400),  "R3": (300, -400),  "R4": (700, 0),
    "R5": (300, 0),     "R6": (700, 400),   "R7": (300, 400),   "R8": (500, 800),
}


def extended_statistical_features(df: pd.DataFrame) -> dict:
    sensor_cols = [c for c in df.columns if c.startswith("L") or c.startswith("R")]
    feats = {}
    for col in sensor_cols:
        vals = df[col].values
        feats[f"{col}_mean"] = np.mean(vals)
        feats[f"{col}_median"] = np.median(vals)
        feats[f"{col}_sd"] = np.std(vals)
        feats[f"{col}_min"] = np.min(vals)
        feats[f"{col}_max"] = np.max(vals)
        feats[f"{col}_range"] = np.max(vals) - np.min(vals)
        feats[f"{col}_iqr"] = np.percentile(vals, 75) - np.percentile(vals, 25)
        feats[f"{col}_skew"] = skew(vals)
        feats[f"{col}_kurtosis"] = kurtosis(vals)
    return feats


def detect_stance_phases_hysteresis(force: np.ndarray) -> list:
    """Hysteresis thresholding: enter stance when force crosses above HIGH,
    exit stance when force drops below LOW. Returns list of (start, end) idx pairs."""
    phases = []
    in_stance = False
    start = None
    for i, val in enumerate(force):
        if not in_stance and val > THRESH_HIGH_N:
            in_stance = True
            start = i
        elif in_stance and val < THRESH_LOW_N:
            in_stance = False
            phases.append((start, i))
    if in_stance and start is not None:
        phases.append((start, len(force)))
    return phases


def gait_cycle_features(df: pd.DataFrame, foot_prefix: str, total_col: str) -> dict:
    """Stance/swing/stride timing + variability for one foot."""
    feats = {}
    stance_phases = detect_stance_phases_hysteresis(df[total_col].values)

    if len(stance_phases) < 2:
        for k in ["stance_time", "swing_time", "stride_time", "stride_time_cv", "step_time_cv"]:
            feats[f"{foot_prefix}_{k}"] = np.nan
        return feats

    stance_durations = np.array([(e - s) / SAMPLE_RATE_HZ for s, e in stance_phases])
    swing_durations = np.array([
        (stance_phases[i + 1][0] - stance_phases[i][1]) / SAMPLE_RATE_HZ
        for i in range(len(stance_phases) - 1)
    ])
    stride_durations = np.array([
        (stance_phases[i + 1][0] - stance_phases[i][0]) / SAMPLE_RATE_HZ
        for i in range(len(stance_phases) - 1)
    ])

    feats[f"{foot_prefix}_stance_time"] = stance_durations.mean()
    feats[f"{foot_prefix}_swing_time"] = swing_durations.mean() if len(swing_durations) else np.nan
    feats[f"{foot_prefix}_stride_time"] = stride_durations.mean() if len(stride_durations) else np.nan

    # Coefficient of variation = SD/mean — a known PD biomarker (irregular gait timing).
    # Higher CV = more irregular gait, associated with PD severity in the literature.
    if len(stride_durations) > 1 and stride_durations.mean() > 0:
        feats[f"{foot_prefix}_stride_time_cv"] = stride_durations.std() / stride_durations.mean()
    else:
        feats[f"{foot_prefix}_stride_time_cv"] = np.nan

    return feats


def center_of_pressure_features(df: pd.DataFrame, side: str) -> dict:
    """Computes COP trajectory per timestep from weighted sensor positions,
    then summarizes its spread/movement as spatial gait features."""
    sensors = [f"{side}{i}" for i in range(1, 9)]
    forces = df[sensors].values  # shape (T, 8)
    xs = np.array([SENSOR_POSITIONS[s][0] for s in sensors])
    ys = np.array([SENSOR_POSITIONS[s][1] for s in sensors])

    total_force = forces.sum(axis=1)
    valid = total_force > THRESH_LOW_N  # only compute COP while foot is loaded

    if valid.sum() < 2:
        return {
            f"{side}_cop_x_sd": np.nan, f"{side}_cop_y_sd": np.nan,
            f"{side}_cop_path_length": np.nan,
        }

    cop_x = np.divide(
        (forces[valid] * xs).sum(axis=1), total_force[valid],
        out=np.zeros(valid.sum()), where=total_force[valid] != 0,
    )
    cop_y = np.divide(
        (forces[valid] * ys).sum(axis=1), total_force[valid],
        out=np.zeros(valid.sum()), where=total_force[valid] != 0,
    )

    # Mediolateral (x) and anteroposterior (y) sway
    cop_x_sd = np.std(cop_x)
    cop_y_sd = np.std(cop_y)

    # Total COP path length (sum of step-to-step displacement) — a common
    # postural-control/gait-stability metric in biomechanics literature.
    path_length = np.sum(np.sqrt(np.diff(cop_x) ** 2 + np.diff(cop_y) ** 2))

    return {
        f"{side}_cop_x_sd": cop_x_sd,
        f"{side}_cop_y_sd": cop_y_sd,
        f"{side}_cop_path_length": path_length,
    }


def symmetry_features(feats: dict) -> dict:
    """Left-right symmetry index for stance/stride time — PD gait tends to be
    more asymmetric than healthy gait."""
    sym = {}
    for metric in ["stance_time", "stride_time"]:
        l_val = feats.get(f"L_{metric}", np.nan)
        r_val = feats.get(f"R_{metric}", np.nan)
        if pd.isna(l_val) or pd.isna(r_val) or (l_val + r_val) == 0:
            sym[f"{metric}_symmetry_index"] = np.nan
        else:
            # Symmetry index formula (Robinson et al.): 0 = perfectly symmetric
            sym[f"{metric}_symmetry_index"] = abs(l_val - r_val) / (0.5 * (l_val + r_val))
    return sym


def cadence_feature(df: pd.DataFrame) -> dict:
    total_duration_s = df["Time"].iloc[-1] - df["Time"].iloc[0]
    n_events = (
        len(detect_stance_phases_hysteresis(df["TotalL"].values))
        + len(detect_stance_phases_hysteresis(df["TotalR"].values))
    )
    cadence = (n_events / total_duration_s) * 60 if total_duration_s > 0 else np.nan
    return {"cadence_steps_per_min": cadence}


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
            "group": group,
            "subject_id": f"{cohort}{group}{subj_id}",
            "trial": trial,
        }

        row.update(extended_statistical_features(df))
        row.update(gait_cycle_features(df, "L", "TotalL"))
        row.update(gait_cycle_features(df, "R", "TotalR"))
        row.update(symmetry_features(row))
        row.update(center_of_pressure_features(df, "L"))
        row.update(center_of_pressure_features(df, "R"))
        row.update(cadence_feature(df))

        rows.append(row)

    result = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"Extracted features for {len(result)} trials -> {OUT_PATH.resolve()}")
    print(f"Columns: {result.shape[1]}")
    print(f"New feature families vs v1: COP (path length + sway), stride-time CV, symmetry index, skew/kurtosis/min/max/range/IQR")


if __name__ == "__main__":
    main()
