"""
03_extract_features_v3.py
Extends features_v2.csv with new feature families and writes features_v3.csv plus
feature_groups_v3.csv (feature -> group), so experiments can ablate whole families.

New families
  spectral        Welch spectrum of TotalL / TotalR: dominant frequency, relative band power,
                  spectral entropy, spectral centroid
  wavelet         db4 level-5 relative energy per band of TotalL / TotalR
  regularity      step / stride regularity (autocorrelation peaks of TotalL+TotalR),
                  sample entropy and SD of stride-time series
  double_support  share of time both feet are loaded, stance/swing percentage of the stride cycle
  demographics    age, sex, height (m), weight (kg), BMI from demographics.xls
  speed           usual-pace walking speed (Speed_01, m/s, subject level) and derived
                  stride/step length = speed x stride time (DERIVED ESTIMATES, not measured
                  distances; speed was timed by the clinicians in trial 01 and is applied to
                  all of the subject's trials)

Deliberately NOT used as features: UPDRS, UPDRSM, TUAG, Hoehn & Yahr (clinical severity
scores that are label proxies or missing for controls).

Height in demographics.xls is in metres for the Ga cohort and centimetres for Ju/Si;
values > 3 are divided by 100.

Run from project root:
    python scripts/03_extract_features_v3.py
"""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks, welch

INTERIM_DIR = Path("data/interim")
V2_PATH = Path("data/processed/features_v2.csv")
DEMO_PATH = Path("data/raw/demographics.xls")
OUT_PATH = Path("data/processed/features_v3.csv")
GROUPS_PATH = Path("data/processed/feature_groups_v3.csv")

FS = 100
FILENAME_PATTERN = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$", re.IGNORECASE)
BANDS = {"low_0p3_1p5": (0.3, 1.5), "mid_1p5_3": (1.5, 3.0), "high_3_10": (3.0, 10.0)}


def load_v2_module():
    path = Path(__file__).with_name("03_extract_features_v2.py")
    spec = importlib.util.spec_from_file_location("features_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spectral_features(sig: np.ndarray, tag: str) -> dict:
    sig = sig - sig.mean()
    f, pxx = welch(sig, fs=FS, nperseg=min(512, len(sig)))
    total = pxx[(f >= 0.1) & (f <= 20)].sum()
    out = {}
    gait_band = (f >= 0.3) & (f <= 5)
    out[f"spec_{tag}_dom_freq_hz"] = f[gait_band][np.argmax(pxx[gait_band])] if gait_band.any() else np.nan
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[f"spec_{tag}_relpower_{name}"] = pxx[m].sum() / total if total > 0 else np.nan
    p = pxx[(f >= 0.1) & (f <= 20)]
    p = p / p.sum() if p.sum() > 0 else p
    out[f"spec_{tag}_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p))) if len(p) > 1 and p.sum() > 0 else np.nan
    out[f"spec_{tag}_centroid_hz"] = float((f[(f >= 0.1) & (f <= 20)] * p).sum()) if p.sum() > 0 else np.nan
    return out


def wavelet_features(sig: np.ndarray, tag: str) -> dict:
    coeffs = pywt.wavedec(sig - sig.mean(), "db4", level=5)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    rel = energies / energies.sum() if energies.sum() > 0 else np.full(len(energies), np.nan)
    names = ["approx"] + [f"detail{lvl}" for lvl in range(5, 0, -1)]
    return {f"wav_{tag}_{n}_relenergy": float(v) for n, v in zip(names, rel)}


def autocorr(sig: np.ndarray, max_lag: int) -> np.ndarray:
    x = sig - sig.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return acf / acf[0] if acf[0] > 0 else np.full(max_lag + 1, np.nan)


def regularity_features(total_sum: np.ndarray) -> dict:
    """Step / stride regularity = height of first / second dominant autocorrelation peak."""
    out = {"step_regularity": np.nan, "stride_regularity": np.nan, "regularity_ratio": np.nan}
    acf = autocorr(total_sum, min(400, len(total_sum) // 2))
    if np.isnan(acf).any():
        return out
    peaks, _ = find_peaks(acf, distance=30, height=0.0)
    peaks = peaks[peaks > 30]
    if len(peaks) >= 2:
        out["step_regularity"], out["stride_regularity"] = float(acf[peaks[0]]), float(acf[peaks[1]])
        out["regularity_ratio"] = out["stride_regularity"] / out["step_regularity"] if out["step_regularity"] > 0 else np.nan
    elif len(peaks) == 1:
        out["step_regularity"] = float(acf[peaks[0]])
    return out


def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    x = np.asarray(x, float)
    if len(x) < m + 5 or x.std() == 0:
        return np.nan
    r = r_factor * x.std()

    def match_count(mm):
        emb = np.array([x[i:i + mm] for i in range(len(x) - mm)])
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(emb)) / 2
    b, a = match_count(m), match_count(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else np.nan


def stride_series(v2, force: np.ndarray) -> np.ndarray:
    phases = v2.detect_stance_phases_hysteresis(force)
    return np.array([(phases[i + 1][0] - phases[i][0]) / FS for i in range(len(phases) - 1)])


def stance_mask(v2, force: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(force), bool)
    for s, e in v2.detect_stance_phases_hysteresis(force):
        mask[s:e] = True
    return mask


def double_support_features(v2, df: pd.DataFrame) -> dict:
    ml, mr = stance_mask(v2, df["TotalL"].values), stance_mask(v2, df["TotalR"].values)
    either = (ml | mr).sum()
    out = {"double_support_pct": float((ml & mr).sum() / either) if either else np.nan}
    for side, m in (("L", ml), ("R", mr)):
        out[f"{side}_stance_pct_time"] = float(m.mean())
    return out


def load_demographics() -> pd.DataFrame:
    d = pd.read_excel(DEMO_PATH)
    d = d.rename(columns={"Height (meters)": "height", "Weight (kg)": "weight", "Speed_01 (m/sec)": "speed"})
    d["height"] = np.where(d["height"] > 3, d["height"] / 100.0, d["height"])
    d["sex_male"] = (d["Gender"].str.lower() == "male").astype(int)
    d["bmi"] = d["weight"] / d["height"] ** 2
    d = d.rename(columns={"Age": "age"})
    return d[["ID", "age", "sex_male", "height", "weight", "bmi", "speed"]].rename(columns={"ID": "subject_id"})


def assign_group(col: str) -> str:
    if col.startswith("spec_"): return "spectral"
    if col.startswith("wav_"): return "wavelet"
    if col in ("step_regularity", "stride_regularity", "regularity_ratio", "stride_time_sampen", "stride_time_sd"): return "regularity"
    if col in ("double_support_pct", "L_stance_pct_time", "R_stance_pct_time"): return "double_support"
    if col in ("age", "sex_male", "height", "weight", "bmi"): return "demographics"
    if col in ("speed_mps", "stride_length_est_m", "step_length_est_m", "stride_length_over_height"): return "speed"
    if "_cop_" in col: return "cop"
    if "symmetry" in col: return "symmetry"
    if col in ("cadence_steps_per_min",) or col.endswith(("_stance_time", "_swing_time", "_stride_time", "_stride_time_cv", "_step_time_cv")): return "timing"
    return "stats"


def main():
    v2 = load_v2_module()
    base = pd.read_csv(V2_PATH)
    demo = load_demographics()
    print(f"v2 features: {base.shape}; demographics rows: {len(demo)}")

    rows = []
    for _, r in base.iterrows():
        f = next(INTERIM_DIR.rglob(r["filename"]))
        df = pd.read_csv(f)
        row = {"filename": r["filename"]}
        for side in ("L", "R"):
            sig = df[f"Total{side}"].values
            row.update(spectral_features(sig, side))
            row.update(wavelet_features(sig, side))
        row.update(regularity_features((df["TotalL"] + df["TotalR"]).values))
        strides = np.concatenate([stride_series(v2, df["TotalL"].values), stride_series(v2, df["TotalR"].values)])
        row["stride_time_sd"] = float(strides.std()) if len(strides) > 1 else np.nan
        row["stride_time_sampen"] = sample_entropy(strides) if len(strides) > 10 else np.nan
        row.update(double_support_features(v2, df))
        rows.append(row)
    new = pd.DataFrame(rows)

    out = base.merge(new, on="filename").merge(demo, on="subject_id", how="left")
    out = out.rename(columns={"speed": "speed_mps"})
    stride_t = out[["L_stride_time", "R_stride_time"]].mean(axis=1)
    out["stride_length_est_m"] = out["speed_mps"] * stride_t
    out["step_length_est_m"] = out["stride_length_est_m"] / 2.0
    out["stride_length_over_height"] = out["stride_length_est_m"] / out["height"]

    id_cols = ["filename", "cohort", "group", "subject_id", "trial"]
    feat_cols = [c for c in out.columns if c not in id_cols]
    groups = pd.DataFrame({"feature": feat_cols, "group": [assign_group(c) for c in feat_cols]})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    groups.to_csv(GROUPS_PATH, index=False)
    print(f"Wrote {OUT_PATH} {out.shape} and {GROUPS_PATH}")
    print(groups.group.value_counts().to_string())
    print("\nMissing values in new columns (top):")
    print(out[feat_cols].isna().sum().sort_values(ascending=False).head(8).to_string())


if __name__ == "__main__":
    main()
