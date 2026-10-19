"""
42_add_dfa_feature.py
Adds Detrended Fluctuation Analysis (DFA) scaling exponent features — a stride-to-stride
long-range correlation / regularity biomarker well established in the PD gait literature
(e.g. Hausdorff et al.), flagged in rules.md as "not implemented" under features v3's
regularity family. This script ADDS to the existing feature table rather than rewriting
the v3 extractor, so it's safe to run without touching anything already validated.

Computes DFA alpha on each trial's TotalL and TotalR force signal (and their mean), per
Peng et al. 1994's standard algorithm, then merges into a copy of features_v3.csv.

Assumes (matches the project's established conventions):
  - Cleaned per-trial signals live in data/interim/<cohort>/<group>/<filename>.txt
    with columns Time, L1-L8, R1-R8, TotalL, TotalR (as produced by 02_clean_signals.py)
  - data/processed/features_v3.csv exists with filename, cohort, group, subject_id columns

Run from project root:
    python scripts/42_add_dfa_feature.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

INTERIM_DIR = Path("data/interim")
FEATURES_V3_PATH = Path("data/processed/features_v3.csv")
OUT_PATH = Path("data/processed/features_v3_dfa.csv")

# DFA box sizes (number of samples), log-spaced, kept modest given trial lengths
# (shortest trials are ~36s at 100Hz = ~3600 samples; cap max box size accordingly)
DFA_MIN_BOX = 16
DFA_MAX_BOX = 400
DFA_N_BOXES = 15


def compute_dfa(signal: np.ndarray, min_box=DFA_MIN_BOX, max_box=DFA_MAX_BOX, n_boxes=DFA_N_BOXES) -> float:
    """Standard DFA (Peng et al. 1994): integrate the mean-centered signal, detrend in
    boxes of increasing size, measure RMS fluctuation per box size, fit log-log slope.
    Returns the DFA alpha exponent (NaN if signal too short or degenerate)."""
    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]
    n = len(signal)
    if n < min_box * 4:
        return np.nan

    # Step 1: integrate the mean-centered series
    y = np.cumsum(signal - np.mean(signal))

    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(min(max_box, n // 4)), n_boxes).astype(int))
    if len(box_sizes) < 4:
        return np.nan

    fluctuations = []
    valid_sizes = []
    for box_size in box_sizes:
        n_segments = n // box_size
        if n_segments < 2:
            continue
        segments = y[: n_segments * box_size].reshape(n_segments, box_size)
        x = np.arange(box_size)
        rms_list = []
        for seg in segments:
            # linear detrend within the box
            coeffs = np.polyfit(x, seg, 1)
            trend = np.polyval(coeffs, x)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluctuations.append(np.mean(rms_list))
        valid_sizes.append(box_size)

    if len(valid_sizes) < 4:
        return np.nan

    log_sizes = np.log10(valid_sizes)
    log_fluct = np.log10(fluctuations)
    # guard against zero/negative fluctuation (degenerate flat signal)
    if not np.all(np.isfinite(log_fluct)):
        return np.nan

    alpha, _ = np.polyfit(log_sizes, log_fluct, 1)
    return alpha


def main():
    if not FEATURES_V3_PATH.exists():
        raise SystemExit(f"{FEATURES_V3_PATH} not found — run the v3 feature extraction first.")

    features = pd.read_csv(FEATURES_V3_PATH)
    required_cols = {"filename", "cohort", "group"}
    if not required_cols.issubset(features.columns):
        raise SystemExit(
            f"features_v3.csv is missing expected columns {required_cols - set(features.columns)}. "
            "Check this script's assumptions against the actual v3 schema and adjust."
        )

    dfa_rows = []
    n_ok, n_fail = 0, []

    for _, row in features.iterrows():
        filepath = INTERIM_DIR / row["cohort"] / row["group"] / row["filename"]
        if not filepath.exists():
            n_fail.append((row["filename"], "interim file not found"))
            continue

        try:
            df = pd.read_csv(filepath)
            dfa_l = compute_dfa(df["TotalL"].values)
            dfa_r = compute_dfa(df["TotalR"].values)
            dfa_mean = compute_dfa(df["TotalL"].values + df["TotalR"].values)

            dfa_rows.append({
                "filename": row["filename"],
                "dfa_alpha_L": dfa_l,
                "dfa_alpha_R": dfa_r,
                "dfa_alpha_combined": dfa_mean,
            })
            n_ok += 1
        except Exception as e:
            n_fail.append((row["filename"], str(e)))

    dfa_df = pd.DataFrame(dfa_rows)
    merged = features.merge(dfa_df, on="filename", how="left")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f"Computed DFA for {n_ok}/{len(features)} trials")
    if n_fail:
        print(f"Failed/missing: {len(n_fail)} trials")
        for name, reason in n_fail[:10]:
            print(f"  {name}: {reason}")
    print(f"\nSaved -> {OUT_PATH.resolve()}")
    print(f"New columns: dfa_alpha_L, dfa_alpha_R, dfa_alpha_combined")
    print(f"\nNote: DFA alpha ~0.5 = uncorrelated (white noise), ~1.0 = long-range correlated")
    print("(typical healthy gait); PD literature generally reports LOWER alpha (more random,")
    print("less correlated stride-to-stride structure) in PD vs healthy — check this direction")
    print("empirically here rather than assuming it transfers from stride-interval studies to VGRF.")


if __name__ == "__main__":
    main()
