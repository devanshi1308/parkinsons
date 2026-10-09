"""
30_voice_load.py
Loads the UCI "Parkinson's Speech with Multiple Types of Sound Recordings" dataset
(Sakar et al. 2013; UCI ML Repository #301) and writes one tidy CSV.

Source files (not committed; see data-README.md for the download step):
  data/raw_voice/Parkinson_Multiple_Sound_Recording/train_data.txt   40 subjects (20 PD, 20
    healthy) x 26 recordings each = 1040 rows. Columns: subject_id, 26 acoustic features,
    UPDRS, class (0=healthy, 1=PD).
  data/raw_voice/Parkinson_Multiple_Sound_Recording/test_data.txt    28 PD-ONLY subjects x 6
    recordings each = 168 rows. No healthy controls and no UPDRS column, so this split cannot
    be used to measure accuracy (see prd.md / rules.md) -- it is loaded for completeness only.

Feature name order is taken from the dataset's own UCI page description (verified there, not
guessed): 5 Jitter, 6 Shimmer, AC/NTH/HTN, 5 pitch, 4 periodicity, 3 voice-quality measures.

Subject IDs are prefixed "UCI{train|test}{n}" so they can never collide with gaitpdb or PADS
subject IDs if files are ever opened side by side -- these are still a DIFFERENT set of people
and must never be merged with the gait or wrist data.

Output: data/processed/voice_features.csv

Run from project root:
    python scripts/30_voice_load.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw_voice/Parkinson_Multiple_Sound_Recording")
OUT_PATH = Path("data/processed/voice_features.csv")

FEATURE_NAMES = [
    "jitter_local", "jitter_local_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp",
    "shimmer_local", "shimmer_local_db", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11", "shimmer_dda",
    "ac", "nth", "htn",
    "pitch_median", "pitch_mean", "pitch_sd", "pitch_min", "pitch_max",
    "n_pulses", "n_periods", "period_mean", "period_sd",
    "frac_unvoiced", "n_voice_breaks", "degree_voice_breaks",
]
assert len(FEATURE_NAMES) == 26


def load_split(filename: str, partition: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise SystemExit(f"{path} not found -- download and extract the dataset first (see data-README.md)")
    cols = ["subject_num"] + FEATURE_NAMES + (["updrs", "class"] if partition == "train" else ["class"])
    df = pd.read_csv(path, header=None, names=cols)
    df["partition"] = partition
    df["subject_id"] = partition.upper() + df["subject_num"].astype(str)
    if partition == "test":
        df["updrs"] = pd.NA
    df["label"] = df["class"].map({0: "Healthy", 1: "PD"})
    return df


def main():
    train = load_split("train_data.txt", "train")
    test = load_split("test_data.txt", "test")
    print(f"train: {train.shape[0]} recordings, {train.subject_id.nunique()} subjects, "
          f"classes {train.label.value_counts().to_dict()}")
    print(f"test:  {test.shape[0]} recordings, {test.subject_id.nunique()} subjects, "
          f"classes {test.label.value_counts().to_dict()} (PD-only -- not usable for accuracy)")

    out = pd.concat([train, test], ignore_index=True)
    id_cols = ["subject_id", "subject_num", "partition", "label", "class", "updrs"]
    out = out[id_cols + FEATURE_NAMES]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}  shape {out.shape}")


if __name__ == "__main__":
    main()
