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

## Additional datasets (planned, for cross-modality validation)

These are not downloaded yet. Place each in its own folder so they never mix with gaitpdb.

### PADS — Parkinson's Disease Smartwatch (wrist tremor / hand movement, Model B)
- Page: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
- ~1.4 GB uncompressed (~735 MB zip). Open access, CC BY-NC-SA 4.0 (non-commercial, share-alike).
- Download into `data/raw_pads/`:
  ```
  wget -r -N -c -np -P data/raw_pads https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/
  ```
- After download, read the metadata/patient files to confirm the per-group counts
  (PD / differential diagnosis / healthy) before writing the loader.

### UCI Parkinson's Speech with Multiple Types of Sound Recordings (voice, Model C)
- Page: https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+Audio+recordings
- CC BY 4.0. Column 1 is the subject ID; 26 acoustic features per recording, no raw audio.
- Training set: 20 PD + 20 healthy, 26 recordings each. Test set: 28 PD only (no controls),
  so use the training set with subject-grouped CV.
- Download into `data/raw_voice/` (zip from the page above, or `pip install ucimlrepo`).

Note: these three datasets contain different people. Never merge rows across them.
Keep `data/raw_pads/` and `data/raw_voice/` out of git (add to `.gitignore`).

## Citation

Hausdorff JM. Gait in Parkinson's Disease. PhysioNet.
https://doi.org/10.13026/C24H3N

Goldberger, A., et al. PhysioBank, PhysioToolkit, and PhysioNet:
Components of a New Research Resource for Complex Physiologic Signals.
Circulation 101(23):e215-e220.


PADS: see the dataset page and the paper "Machine Learning in the Parkinson's disease
smartwatch (PADS) dataset", npj Parkinson's Disease (2024).

UCI speech: Sakar BE et al. Collection and analysis of a Parkinson speech dataset with
multiple types of sound recordings. IEEE J Biomed Health Inform 2013.
