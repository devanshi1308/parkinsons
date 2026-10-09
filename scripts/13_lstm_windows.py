"""
13_lstm_windows.py
Optional sliding-window LSTM experiment (binary Healthy-vs-PD), to check whether a sequence
model can beat the tabular-feature models now that subject-grouped CV is used honestly. This
was expected going in to underperform at 165 subjects (see CLAUDE.md/rules.md) -- the point is
to report the result honestly either way, not to chase a win.

Design:
  - Each trial's 18-channel VGRF signal (16 sensors + TotalL + TotalR) is cut into overlapping
    5-second windows (500 samples, 50% overlap / 2.5s stride) -- turns 306 trials into ~11k
    windows, enough to train a small sequence model without the model collapsing to the
    majority class immediately.
  - CV grouping is by subject_id (not trial, not window), so every window from every trial of
    a given subject stays in one fold -- no subject ever split across train/test, same rule as
    everywhere else in this project.
  - Per-channel z-score normalization is fit on TRAINING windows only, inside each fold.
  - Light augmentation (Gaussian jitter + random amplitude scaling), applied only to the
    training windows of each fold, each epoch.
  - A small 1-layer LSTM (hidden=48) + dense classifier, trained with early stopping on a
    held-out slice of the training subjects (not the test fold).
  - Reported at both window level and trial level (mean window-probability per trial, since
    that's the unit the rest of the project evaluates at).

5-fold (not 10 -- fewer, larger folds keep the LSTM's per-fold training set usable) subject-
grouped CV, single seed (deep nets are expensive to repeat here; this is an exploratory
secondary result, not the primary reported number).

Outputs:
  data/processed/lstm_results.csv
  docs/figures/lstm_vs_tabular.png

Run from project root:
    python scripts/13_lstm_windows.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

INTERIM_DIR = Path("data/interim")
LABELS_PATH = Path("data/processed/subject_labels.csv")
V2_PATH = Path("data/processed/features_v2.csv")  # for filename/subject_id/cohort/group lookup
OUT_CSV = Path("data/processed/lstm_results.csv")
OUT_FIG = Path("docs/figures/lstm_vs_tabular.png")

FS = 100
WINDOW_S = 5.0
STRIDE_S = 2.5
WINDOW_LEN = int(WINDOW_S * FS)
STRIDE_LEN = int(STRIDE_S * FS)
CHANNELS = [f"L{i}" for i in range(1, 9)] + [f"R{i}" for i in range(1, 9)] + ["TotalL", "TotalR"]
SEED = 42
N_FOLDS = 5
MAX_EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 64
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def make_windows():
    """Cut every trial into overlapping windows; returns (X, y, groups, trial_ids)."""
    base = pd.read_csv(V2_PATH)
    labels = pd.read_csv(LABELS_PATH)
    df = base.merge(labels[["subject_id", "stage"]], on="subject_id", how="left").dropna(subset=["stage"])
    df = df.assign(label=np.where(df.stage == "Healthy", "Healthy", "PD"))

    X_list, y_list, group_list, trial_list = [], [], [], []
    for _, row in df.iterrows():
        f = next(INTERIM_DIR.rglob(row["filename"]))
        sig = pd.read_csv(f)[CHANNELS].to_numpy(dtype=np.float32)
        n = len(sig)
        for start in range(0, max(n - WINDOW_LEN, 0) + 1, STRIDE_LEN):
            X_list.append(sig[start:start + WINDOW_LEN])
            y_list.append(row["label"])
            group_list.append(row["subject_id"])
            trial_list.append(row["filename"])
    X = np.stack(X_list).astype(np.float32)
    y = LabelEncoder().fit_transform(np.array(y_list))  # Healthy=0, PD=1
    groups = np.array(group_list)
    trials = np.array(trial_list)
    print(f"Windows: {len(X)} from {df.shape[0]} trials, {df.subject_id.nunique()} subjects "
          f"(PD share {y.mean():.2f}); window shape {X.shape[1:]}")
    return X, y, groups, trials


def augment(batch_x):
    noise = torch.randn_like(batch_x) * 0.03
    scale = (0.9 + 0.2 * torch.rand(batch_x.size(0), 1, 1, device=batch_x.device))
    return batch_x * scale + noise


class SmallLSTM(nn.Module):
    def __init__(self, n_channels, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def train_one_fold(Xtr, ytr, Xval, yval, Xte, yte, device):
    mean, std = Xtr.mean(axis=(0, 1), keepdims=True), Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    norm = lambda a: (a - mean) / std
    Xtr_t = torch.tensor(norm(Xtr)); Xval_t = torch.tensor(norm(Xval)); Xte_t = torch.tensor(norm(Xte))
    ytr_t, yval_t, yte_t = torch.tensor(ytr, dtype=torch.float32), torch.tensor(yval, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32)

    model = SmallLSTM(Xtr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)], dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH_SIZE, shuffle=True)
    best_val, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = augment(xb)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = lossf(model(Xval_t.to(device)), yval_t.to(device)).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_prob = torch.sigmoid(model(Xte_t.to(device))).cpu().numpy()
    return te_prob, epoch + 1


def main():
    X, y, groups, trials = make_windows()
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    device = torch.device(DEVICE)
    print(f"Device: {DEVICE}")

    window_rows, trial_rows = [], []
    for fold, (tr_full, te) in enumerate(cv.split(X, y, groups)):
        # carve a validation slice (for early stopping) out of the training subjects only
        tr_groups = groups[tr_full]
        uniq_tr_subj = np.unique(tr_groups)
        rng = np.random.default_rng(SEED + fold)
        val_subj = set(rng.choice(uniq_tr_subj, size=max(1, len(uniq_tr_subj) // 6), replace=False))
        val_mask = np.isin(tr_groups, list(val_subj))
        tr, val = tr_full[~val_mask], tr_full[val_mask]

        prob, n_epochs = train_one_fold(X[tr], y[tr], X[val], y[val], X[te], y[te], device)
        pred = (prob >= 0.5).astype(int)
        acc = np.mean(pred == y[te])
        print(f"  fold {fold + 1}/{N_FOLDS}: train {len(tr)} windows, val {len(val)}, test {len(te)} "
              f"windows ({n_epochs} epochs) -> window-level acc {acc:.3f}")
        for i, idx in enumerate(te):
            window_rows.append({"fold": fold, "subject_id": groups[idx], "trial": trials[idx], "y_true": y[idx], "prob": prob[i]})

    wdf = pd.DataFrame(window_rows)
    window_acc = np.mean((wdf.prob >= 0.5).astype(int) == wdf.y_true)
    window_f1 = 2 * ((wdf.prob >= 0.5).astype(int) & wdf.y_true).sum() / (((wdf.prob >= 0.5).astype(int).sum()) + wdf.y_true.sum() + 1e-9)

    tdf = wdf.groupby(["trial", "subject_id"]).agg(y_true=("y_true", "first"), prob=("prob", "mean")).reset_index()
    tdf["pred"] = (tdf.prob >= 0.5).astype(int)
    trial_acc = np.mean(tdf.pred == tdf.y_true)
    from sklearn.metrics import f1_score, roc_auc_score
    trial_f1 = f1_score(tdf.y_true, tdf.pred, average="macro")
    trial_auc = roc_auc_score(tdf.y_true, tdf.prob)

    print(f"\nWindow-level: accuracy {window_acc:.3f}")
    print(f"Trial-level (mean window prob per trial): accuracy {trial_acc:.3f}, macro-F1 {trial_f1:.3f}, AUC {trial_auc:.3f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"level": "window", "n": len(wdf), "accuracy": window_acc},
        {"level": "trial (mean window prob)", "n": len(tdf), "accuracy": trial_acc, "f1_macro": trial_f1, "auc": trial_auc},
    ]).to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")

    # comparison figure against the tabular models already reported
    tabular = {"Naive Bayes": 0.726, "Decision Tree": 0.717, "SVM": 0.739, "Bagging": 0.782,
               "Nested-selected best\n(RF/HGB/XGB/LGBM)": 0.817}
    labels_ = list(tabular.keys()) + ["LSTM (sliding window,\ntrial-level)"]
    values = list(tabular.values()) + [trial_acc]
    colors = ["#9ec5e8"] * len(tabular) + ["#c0504d"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels_, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Binary accuracy (subject-grouped)")
    ax.set_title("Sliding-window LSTM vs tabular models (gait, binary Healthy vs PD)")
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v:.1%}", ha="center", fontsize=8.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    print(f"Saved -> {OUT_FIG}")


if __name__ == "__main__":
    main()
