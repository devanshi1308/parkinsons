# Data directory

Raw and intermediate data files are **not committed to this repo** (see `.gitignore`) —
they're large and PhysioNet's Open Data Commons Attribution License expects
attribution/download from the source rather than redistribution.

## To reproduce locally

1. Download the dataset from PhysioNet:
   https://physionet.org/content/gaitpdb/1.0.0/
   (or `wget -r -N -c -np https://physionet.org/files/gaitpdb/1.0.0/`)

2. Unzip so that the `.txt` files (GaCo01_01.txt, GaPt03_01.txt, ...),
   `demographics.xls`, and `format.txt` sit directly inside `data/raw/`.

3. From the project root, run:
   ```
   python scripts/01_organize_raw.py
   python scripts/02_clean_signals.py
   ```

## Citation

Hausdorff JM. Gait in Parkinson's Disease. PhysioNet.
https://doi.org/10.13026/C24H3N

Goldberger, A., et al. PhysioBank, PhysioToolkit, and PhysioNet:
Components of a New Research Resource for Complex Physiologic Signals.
Circulation 101(23):e215-e220.
