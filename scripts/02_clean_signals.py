"""
02_clean_signals.py
Loads each raw VGRF .txt file, validates it, trims start/end gait-cycle
edge effects (as the Balaji et al. 2020 paper does), and writes a cleaned
version to data/interim/.

Column layout (per format.txt, 19 columns, tab-separated, no header):
  0     : Time (s)
  1-8   : L1-L8  (left foot VGRF sensors, Newtons)
  9-16  : R1-R8  (right foot VGRF sensors, Newtons)
  17    : Total force under left foot
  18    : Total force under right foot

Run from the project root:
    python scripts/02_clean_signals.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_SORTED_DIR = Path("data/raw_sorted")
INTERIM_DIR = Path("data/interim")

COLS = (
    ["Time"]
    + [f"L{i}" for i in range(1, 9)]
    + [f"R{i}" for i in range(1, 9)]
    + ["TotalL", "TotalR"]
)

SAMPLE_RATE_HZ = 100
EDGE_TRIM_SECONDS = 2.0  # discard this much from start and end to avoid startup/stop artifacts


def load_raw(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    if df.shape[1] != len(COLS):
        raise ValueError(f"{filepath.name}: expected {len(COLS)} columns, got {df.shape[1]}")
    df.columns = COLS
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop exact duplicate rows / rows with NaNs
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # 2. Trim edge effects (first/last ~2s, ~ first & last couple gait cycles)
    trim_rows = int(EDGE_TRIM_SECONDS * SAMPLE_RATE_HZ)
    if len(df) > 2 * trim_rows:
        df = df.iloc[trim_rows:-trim_rows].reset_index(drop=True)

    # 3. Sanity-check: VGRF should be non-negative; clip tiny negative noise to 0
    sensor_cols = [c for c in df.columns if c != "Time"]
    df[sensor_cols] = df[sensor_cols].clip(lower=0)

    return df


def main():
    if not RAW_SORTED_DIR.exists():
        raise SystemExit("Run 01_organize_raw.py first.")

    n_ok, n_fail = 0, []

    for filepath in RAW_SORTED_DIR.rglob("*.txt"):
        rel = filepath.relative_to(RAW_SORTED_DIR)  # e.g. Ga/Co/GaCo01_01.txt
        out_path = INTERIM_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = load_raw(filepath)
            df_clean = clean(df)
            df_clean.to_csv(out_path, index=False)
            n_ok += 1
        except Exception as e:
            n_fail.append((filepath.name, str(e)))

    print(f"Cleaned {n_ok} files -> {INTERIM_DIR.resolve()}")
    if n_fail:
        print("Failed files:")
        for name, err in n_fail:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
