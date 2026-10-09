"""
19_combined_leakage_figure.py
The paper's headline figure: gait and voice side by side, showing that trial/recording-level
(leaky) CV inflates accuracy relative to subject-grouped (honest) CV in BOTH modalities, with
the 20-seed, statistically-tested numbers from 18_statistical_robustness.py (not the earlier
5-seed estimates). Significance stars come from the paired Wilcoxon test (per classifier).

Output: docs/figures/combined_leakage_gap.png

Run from project root:
    python scripts/19_combined_leakage_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIG_PATH = Path("data/processed/leakage_significance.csv")
OUT_FIG = Path("docs/figures/combined_leakage_gap.png")
CLASSIFIER_ORDER = ["Bayes (GaussianNB)", "SVM", "Decision Tree", "Ensemble (Bagging)"]
SHORT = {"Bayes (GaussianNB)": "Naive\nBayes", "SVM": "SVM", "Decision Tree": "Decision\nTree", "Ensemble (Bagging)": "Bagging"}


def stars(p):
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def panel(ax, sig, scheme, title):
    sub = sig[sig.scheme == scheme].set_index("classifier").loc[CLASSIFIER_ORDER]
    x = np.arange(len(CLASSIFIER_ORDER))
    width = 0.38
    ax.bar(x - width / 2, sub.leaky_mean * 100, width, color="#c0504d", label="Trial/recording-level (leaky)")
    ax.bar(x + width / 2, sub.grouped_mean * 100, width, color="#4f81bd", label="Subject-grouped (honest)")
    for i, (_, row) in enumerate(sub.iterrows()):
        top = max(row.leaky_mean, row.grouped_mean) * 100
        ax.annotate("", xy=(i - width / 2, row.leaky_mean * 100 + 1.5), xytext=(i + width / 2, row.grouped_mean * 100 + 1.5),
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
        ax.text(i, top + 3.5, stars(row.wilcoxon_p), ha="center", fontsize=9)
        ax.text(i, top + 8, f"+{row.gap_pts:.0f}pts", ha="center", fontsize=7.5, color="#444")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CLASSIFIER_ORDER])
    ax.set_ylim(0, 108)
    ax.set_title(title, fontsize=11)


def main():
    sig = pd.read_csv(SIG_PATH)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    panel(axes[0], sig[sig.modality == "gait"], "4class", "Gait: 4-class")
    panel(axes[1], sig[sig.modality == "gait"], "3class", "Gait: 3-class")
    panel(axes[2], sig[sig.modality == "gait"], "binary", "Gait: binary")
    panel(axes[3], sig[sig.modality == "voice"], "binary", "Voice: binary")
    axes[0].set_ylabel("Accuracy (%, mean of 20 seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Leakage inflates accuracy in two independent modalities (gait: n=165 subjects, voice: n=40 subjects)\n"
                 "stars = Wilcoxon signed-rank test, paired by seed, leaky vs grouped (**** p<0.0001, *** p<0.001, ** p<0.01, * p<0.05)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    print(f"Saved -> {OUT_FIG}")


if __name__ == "__main__":
    main()
