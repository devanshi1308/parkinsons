"""
04_merge_labels.py
Parses demographics.xls (subject metadata + H&Y stage) and builds a lookup table
mapping subject ID -> stage label (Healthy / Mild / Moderate / Severe), matched
against the cleaned trial files.

Requires: pip install pandas xlrd openpyxl

Run from project root:
    python scripts/04_merge_labels.py
"""

import re
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
OUT_PATH = Path("data/processed/subject_labels.csv")

FILENAME_PATTERN = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$", re.IGNORECASE)


def hy_to_stage(hy_value) -> str:
    """Map Hoehn & Yahr numeric scale to the paper's 4-class labels."""
    if pd.isna(hy_value) or hy_value == 0:
        return "Healthy"
    hy_value = float(hy_value)
    if hy_value == 2:
        return "Mild"
    elif hy_value == 2.5:
        return "Moderate"
    elif hy_value == 3:
        return "Severe"
    else:
        return f"Unmapped_HY_{hy_value}"


def load_demographics() -> pd.DataFrame:
    xls_path = RAW_DIR / "demographics.xls"
    if not xls_path.exists():
        # sometimes distributed as .xlsx
        xls_path = RAW_DIR / "demographics.xlsx"
    if not xls_path.exists():
        raise SystemExit(f"demographics file not found in {RAW_DIR.resolve()}")

    demo = pd.read_excel(xls_path)
    print("Demographics columns found:", list(demo.columns))
    return demo


def main():
    demo = load_demographics()

    # NOTE: column names in the raw file vary slightly by PhysioNet export version.
    # Print the columns above and adjust these two lines to match exactly.
    id_col = next(c for c in demo.columns if "id" in c.lower() or "subject" in c.lower() or "id_pt" in c.lower())
    hy_col = next((c for c in demo.columns if "hy" in c.lower() or "hoehn" in c.lower()), None)

    if hy_col is None:
        raise SystemExit(
            "Could not auto-detect the H&Y column. Open demographics.xls, find the exact "
            "column name for Hoehn & Yahr stage, and hardcode it here as hy_col."
        )

    demo["stage"] = demo[hy_col].apply(hy_to_stage)
    demo = demo.rename(columns={id_col: "subject_id"})

    demo[["subject_id", hy_col, "stage"]].to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(demo)} subject labels to {OUT_PATH.resolve()}")
    print(demo["stage"].value_counts())


if __name__ == "__main__":
    main()
