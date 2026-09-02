# Rules & Conventions

## Code
- Python 3, scripts (not notebooks) for anything that's part of the reproducible pipeline;
  notebooks only in `notebooks/` for exploration
- All scripts run from project root: `python scripts/NN_name.py`
- Every script prints a summary of what it did (files processed, failures) — no silent runs
- Fixed random seed (`RANDOM_STATE = 42`) for every train/test split and model — required
  for reproducibility

## Data
- Raw data never committed to git (see `data/README.md`) — always re-downloadable from
  PhysioNet
- Never edit files in `data/raw/` — all cleaning writes to `data/interim/` or `data/processed/`
- Any change to feature extraction logic = new version of `master_dataset.csv`, old one
  kept until the new one is validated

## Folder discipline
- `data/raw/` → untouched original files
- `data/raw_sorted/` → same files, organized by cohort/group
- `data/interim/` → cleaned per-file signals
- `data/processed/` → final feature tables ready for modeling
- `scripts/` → numbered, single-purpose, idempotent (safe to rerun)
- `figures/` → only generated outputs, regenerable from scripts — not hand-edited
- `manuscript/` → paper drafts only, versioned by date or section

## Research integrity
- Every reported metric must be traceable to a script + git commit that produced it
- No cherry-picking runs — report mean ± SD across the 10-fold CV, not the best fold
- If a result contradicts the baseline paper, investigate before including it (data
  mismatch, feature difference, etc.) rather than silently adjusting

## Novel contribution (decision gate)
- The specific "what's new" angle is not yet locked in — must be decided and written into
  `prd.md` before Phase 4 of `implementationplan.md` starts
- Whatever it is, it must be evaluated on the exact same CV protocol/metrics as the
  baseline reproduction, so the comparison is fair

## Git
- Commit after each completed script/step, not one giant commit
- Commit messages describe what changed and why, not just "update"
- No raw data, no `venv/`, no `__pycache__` ever committed (enforced via `.gitignore`)