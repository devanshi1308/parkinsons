"""
check_cleaned_data.py
Sanity-checks the output of 02_clean_signals.py before moving on to feature extraction.

Run from project root:
    python scripts/check_cleaned_data.py
"""

import pandas as pd
from pathlib import Path

INTERIM_DIR = Path("data/interim")


def main():
    files = list(INTERIM_DIR.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No cleaned files found in {INTERIM_DIR.resolve()} — run 02_clean_signals.py first.")

    print(f"Found {len(files)} cleaned files\n")

    # Breakdown by cohort/group
    counts = {}
    for f in files:
        cohort, group = f.parts[-3], f.parts[-2]
        counts.setdefault((cohort, group), 0)
        counts[(cohort, group)] += 1

    print("Files per cohort/group:")
    for (cohort, group), n in sorted(counts.items()):
        print(f"  {cohort}/{group}: {n}")

    # Spot-check one file in detail
    sample = files[0]
    df = pd.read_csv(sample)
    print(f"\nSpot-check: {sample.name}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Any NaNs left: {df.isna().any().any()}")
    print(f"  Any negative sensor values: {(df.drop(columns='Time') < 0).any().any()}")
    print(f"  Duration: {df['Time'].iloc[-1] - df['Time'].iloc[0]:.1f}s")

    # Flag files that look too short after trimming (edge case)
    short_files = []
    for f in files:
        d = pd.read_csv(f)
        if len(d) < 500:  # < 5 seconds at 100Hz is suspiciously short
            short_files.append((f.name, len(d)))

    if short_files:
        print(f"\nWarning — {len(short_files)} files are unusually short after cleaning:")
        for name, n in short_files:
            print(f"  {name}: {n} rows")
    else:
        print("\nAll files have reasonable length after cleaning.")


if __name__ == "__main__":
    main()
