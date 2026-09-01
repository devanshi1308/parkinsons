"""
01_organize_raw.py
Sorts the flat PhysioNet gaitpdb .txt files into cohort/group subfolders
based on filename prefix, without touching file content.

Filename convention: <Cohort><Group><SubjectID>_<Trial>.txt
  Cohort: Ga (Galit), Ju (Juvenile/Study2), Si (Silvia/Study3)
  Group:  Co (control/healthy), Pt (patient/PD)

Run from the parkinsons-desiese/ project root:
    python scripts/01_organize_raw.py
"""

import re
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
SORTED_DIR = Path("data/raw_sorted")

PATTERN = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$", re.IGNORECASE)


def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"Expected raw data at {RAW_DIR.resolve()} — put the unzipped files there first.")

    moved, skipped = 0, []

    for f in RAW_DIR.glob("*.txt"):
        m = PATTERN.match(f.name)
        if not m:
            skipped.append(f.name)  # e.g. demographics.txt, format.txt, SHA256SUMS.txt
            continue

        cohort, group, subj_id, trial = m.groups()
        dest_dir = SORTED_DIR / cohort / group
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest_dir / f.name)
        moved += 1

    print(f"Sorted {moved} sensor files into {SORTED_DIR.resolve()}")
    if skipped:
        print(f"Skipped (non-trial files, left in place): {skipped}")


if __name__ == "__main__":
    main()
