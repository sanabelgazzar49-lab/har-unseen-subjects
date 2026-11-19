# explore_data.py: descriptive look at the UCI HAR data (no models).
# Usage (from the project root):  python src/explore_data.py
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import ROOT, SIGNALS, activity_names, load_split

SHORT = ["Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lay"]


def main():
    os.makedirs("results/figures", exist_ok=True)
    names = activity_names()
    Xtr, ytr, str_, sigtr = load_split("train")
    Xte, yte, ste, sigte = load_split("test")
    X = np.vstack([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    s = np.concatenate([str_, ste])
    sig = np.concatenate([sigtr, sigte])

    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    say("UCI HAR data summary (computed from the downloaded files)")
    say(f"Windows: {len(y)} (official train {len(ytr)}, official test {len(yte)})")
    say(f"Subjects: {len(np.unique(s))} (official train {len(np.unique(str_))}, official test {len(np.unique(ste))})")
    say(f"Engineered features per window: {X.shape[1]}")
    say(f"Raw signal tensor: {sig.shape} = windows x readings x channels")
    say("Window length: 128 readings (2.56 s at 50 Hz), consecutive windows overlap by 50% (dataset documentation)")
    feat = pd.read_csv(f"{ROOT}/features.txt", sep=r"\s+", header=None, names=["idx", "name"])
    say("First feature names: " + ", ".join(feat["name"].head(5).tolist()))
    say(f"Feature value range: min {X.min():.3f}, max {X.max():.3f}")
    say(f"Missing values in features / signals: {int(np.isnan(X).sum())} / {int(np.isnan(sig).sum())}")

    say("\nWindows per activity (official train | official test):")
    for i, n in enumerate(names, start=1):
        say(f"  {n:20s} {int((ytr == i).sum()):5d} | {int((yte == i).sum()):5d}")
    per_subject = pd.Series(s).value_counts()
    say(f"\nWindows per subject: min {per_subject.min()}, max {per_subject.max()}, median {int(per_subject.median())}")

    say("\nRaw signal channels (mean and standard deviation over all windows and readings):")
    flat = sig.reshape(-1, sig.shape[-1])
    for n, m, sd in zip(SIGNALS, flat.mean(axis=0), flat.std(axis=0)):
        say(f"  {n:12s} mean {m:8.3f}   std {sd:7.3f}")

    with open("results/data_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Figure 1: class balance
    w = 0.38
    xs = np.arange(6)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs - w / 2, [(ytr == i).sum() for i in range(1, 7)], w, label="Official train")
    ax.bar(xs + w / 2, [(yte == i).sum() for i in range(1, 7)], w, label="Official test")
    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("Number of windows")
    ax.set_title("Windows per activity")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/fig_data_classes.png", dpi=200)
    plt.close(fig)

    # Figure 2: windows per subject, coloured by the official split
    gids = np.unique(s)
    test_set = set(np.unique(ste))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(gids)), [(s == g).sum() for g in gids],
           color=["tab:orange" if g in test_set else "tab:blue" for g in gids])
    ax.set_xticks(range(len(gids)))
    ax.set_xticklabels(gids)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Number of windows")
    ax.set_title("Windows per subject (blue = official train subjects, orange = official test subjects)")
    fig.tight_layout()
    fig.savefig("results/figures/fig_data_subjects.png", dpi=200)
    plt.close(fig)

    # Figure 3: one example window per activity (total acceleration, 3 axes)
    t = np.arange(128) / 50.0
    first = int(gids[0])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for a, ax in zip(range(1, 7), axes.ravel()):
        i = np.where((y == a) & (s == first))[0][0]
        for c, lab in zip([6, 7, 8], ["x", "y", "z"]):
            ax.plot(t, sig[i, :, c], label=lab)
        ax.set_title(names[a - 1])
        ax.set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Total acceleration")
    axes[1, 0].set_ylabel("Total acceleration")
    axes[0, 0].legend()
    fig.suptitle(f"One 2.56 s window per activity (subject {first})")
    fig.tight_layout()
    fig.savefig("results/figures/fig_data_signals.png", dpi=200)
    plt.close(fig)

    # Figure 4: 2D PCA of the 561 features (visualization only, not used for modelling)
    Z = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(7, 6))
    for a in range(1, 7):
        m = y == a
        ax.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.5, color=plt.cm.tab10(a - 1), label=names[a - 1])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("2D PCA of the 561 features (for visualization only)")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/fig_data_pca.png", dpi=200)
    plt.close(fig)
    print("\nSaved results/data_summary.txt and 4 figures in results/figures/")


if __name__ == "__main__":
    main()
