# analyze_errors.py: error analysis from saved cross-validation predictions (no training).
# Usage (from the project root):
#   python src/analyze_errors.py [--pred results/cv_predictions.npz] [--out results]
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import LABELS, activity_names, cm_by_subject, macro_f1_from_cm

SHORT = ["Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lay"]


def main(pred_path, out):
    os.makedirs(f"{out}/figures", exist_ok=True)
    names = activity_names()
    z = np.load(pred_path)
    y, subj = z["y"], z["subject"]
    preds = {("RF", "group"): z["rf_group"][None, :], ("RF", "random"): z["rf_random"][None, :],
             ("CNN", "group"): z["cnn_group"], ("CNN", "random"): z["cnn_random"]}
    for k, v in preds.items():
        assert v.shape[1] == len(y) and set(np.unique(v)) <= set(LABELS), f"Unexpected predictions for {k}"
    gids = np.unique(subj)
    print("Windows:", len(y), "| subjects:", len(gids))

    cm = {k: np.array([cm_by_subject(y, pr, subj, gids) for pr in v]) for k, v in preds.items()}  # (runs, 30, 6, 6)
    cm_mean = {k: v.mean(axis=0) for k, v in cm.items()}       # averaged over CNN seeds
    pooled = {k: v.sum(axis=0) for k, v in cm_mean.items()}    # (6, 6)

    print("\nPooled macro-F1 over all 30 subjects:")
    for k in cm:
        val = np.mean([macro_f1_from_cm(cm[k][r].sum(axis=0)) for r in range(cm[k].shape[0])])
        print(f"  {k[0]:3s} {k[1]:6s}: {val:.4f}")

    # 1) Per-class recall
    def recall(c):
        return np.diag(c) / c.sum(axis=1)

    rec = pd.DataFrame({f"{m}_{s}": recall(pooled[(m, s)]) for m in ["RF", "CNN"] for s in ["group", "random"]},
                       index=names)
    rec["RF_gain_from_leak"] = rec["RF_random"] - rec["RF_group"]
    rec["CNN_gain_from_leak"] = rec["CNN_random"] - rec["CNN_group"]
    print("\n===== Per-class recall (group = unseen subjects, random = random window split) =====")
    print(rec.round(3).to_string())
    rec.to_csv(f"{out}/error_recall_table.csv")

    # 2) Pooled confusion matrices and most frequent confusions
    for m in ["RF", "CNN"]:
        pd.DataFrame(pooled[(m, "group")], index=["true_" + n for n in names],
                     columns=["pred_" + n for n in names]).round(1).to_csv(f"{out}/pooled_confusion_unseen_{m}.csv")

    def top_confusions(m, n=6):
        c, cr, cs = pooled[(m, "group")], pooled[(m, "random")], cm_mean[(m, "group")]
        rows = []
        for t in range(6):
            for q in range(6):
                if t == q:
                    continue
                subj_rate = cs[:, t, q] / np.maximum(cs[:, t, :].sum(axis=1), 1e-12)
                rows.append({"model": m, "true": names[t], "predicted_as": names[q],
                             "rate_unseen": c[t, q] / c[t].sum(),
                             "rate_random_split": cr[t, q] / cr[t].sum(),
                             "subjects_with_rate>=5%": int((subj_rate >= 0.05).sum()),
                             "subjects_with_rate>=20%": int((subj_rate >= 0.20).sum())})
        return pd.DataFrame(rows).sort_values("rate_unseen", ascending=False).head(n)

    top = pd.concat([top_confusions("RF"), top_confusions("CNN")], ignore_index=True)
    print("\n===== Most frequent confusions (rate = share of the true class predicted as the other class) =====")
    print("(subjects_with_rate columns: how many of the 30 subjects show this confusion at that level)")
    print(top.round(3).to_string(index=False))
    top.to_csv(f"{out}/top_confusions.csv", index=False)

    # 3) Per-subject scores and the effect of the random split
    per_f1 = {k: np.array([np.mean([macro_f1_from_cm(cm[k][r, i]) for r in range(cm[k].shape[0])])
                           for i in range(len(gids))]) for k in cm}
    infl = {m: per_f1[(m, "random")] - per_f1[(m, "group")] for m in ["RF", "CNN"]}
    per = pd.DataFrame({"subject": gids, "RF_unseen": per_f1[("RF", "group")], "CNN_unseen": per_f1[("CNN", "group")],
                        "RF_inflation": infl["RF"], "CNN_inflation": infl["CNN"]})
    per.to_csv(f"{out}/per_subject_inflation.csv", index=False)
    print("\n===== Effect of the random split, per subject =====")
    for m in ["RF", "CNN"]:
        rho, pval = spearmanr(per_f1[(m, "group")], infl[m])
        print(f"{m}: random split scored higher for {int((infl[m] > 0).sum())} of {len(gids)} subjects | "
              f"largest gain {infl[m].max():.3f} (subject {gids[np.argmax(infl[m])]}) | "
              f"Spearman rho between unseen-subject score and gain = {rho:.2f} (p = {pval:.3f})")
    print("Note: subjects with low unseen-subject scores have more room to gain, so a negative correlation "
          "is partly mechanical. It is descriptive, not a causal test.")

    # 4) Figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, m in zip(axes, ["RF", "CNN"]):
        c = pooled[(m, "group")]
        cn = c / c.sum(axis=1, keepdims=True)
        im = ax.imshow(cn, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(6))
        ax.set_xticklabels(SHORT, rotation=45, ha="right")
        ax.set_yticks(range(6))
        ax.set_yticklabels(SHORT)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{m}: unseen subjects, 30 subjects pooled")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cn[i, j] * 100:.0f}", ha="center", va="center",
                        color="white" if cn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.03, label="Share of true class")
    fig.suptitle("Row-normalized confusion matrices (%), CNN averaged over 3 seeds")
    fig.savefig(f"{out}/figures/fig_confusion_unseen.png", dpi=200)
    plt.close(fig)

    order = np.argsort(per_f1[("RF", "group")] + per_f1[("CNN", "group")])
    x = np.arange(len(gids))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for xi, i in zip(x, order):
        ax.plot([xi, xi], [per_f1[("RF", "group")][i], per_f1[("CNN", "group")][i]], color="gray", lw=0.8)
    ax.plot(x, per_f1[("RF", "group")][order], "o", color="tab:blue", label="Random Forest")
    ax.plot(x, per_f1[("CNN", "group")][order], "s", color="tab:orange", label="1D CNN")
    ax.axhline(0.90, ls="--", color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(gids[order], rotation=90)
    ax.set_xlabel("Subject ID (sorted by mean macro-F1 of the two models)")
    ax.set_ylabel("Macro-F1 on that subject")
    ax.set_title("Performance on each unseen subject (GroupKFold, 5 folds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/figures/fig_per_subject.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(per_f1[("RF", "group")], infl["RF"], marker="o", color="tab:blue", alpha=0.8, label="Random Forest")
    ax.scatter(per_f1[("CNN", "group")], infl["CNN"], marker="s", color="tab:orange", alpha=0.8, label="1D CNN")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Macro-F1 when the subject is unseen")
    ax.set_ylabel("Gain under random split (random minus unseen)")
    ax.set_title("How much each subject benefits from the random split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/figures/fig_inflation_vs_difficulty.png", dpi=200)
    plt.close(fig)
    print(f"\nTables and 3 figures written to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/cv_predictions.npz")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.pred, a.out)
