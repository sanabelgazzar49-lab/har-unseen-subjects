# accuracy_ci.py: الدقة على الأشخاص الثلاثين مع فاصل ثقة بإعادة سحب الأشخاص.
# الاستخدام (من جذر المشروع):  python src/accuracy_ci.py [--pred results/cv_predictions.npz] [--out results]
import argparse
import os

import numpy as np
import pandas as pd

SEED, B = 2026, 5000


def main(pred_path, out):
    os.makedirs(out, exist_ok=True)
    z = np.load(pred_path)
    y, subj = z["y"], z["subject"]
    gids = np.unique(subj)
    total = np.array([(subj == g).sum() for g in gids])       # عدد نوافذ كل شخص
    assert total.sum() == len(y) and len(gids) == 30

    def correct_per_subject(pred):
        # عدد النوافذ المصنَّفة صحيحاً لكل شخص
        return np.array([(pred[subj == g] == y[subj == g]).sum() for g in gids])

    rf = {k: correct_per_subject(z[f"rf_{k}"]) for k in ["group", "random"]}
    cnn = {k: np.array([correct_per_subject(p) for p in z[f"cnn_{k}"]]) for k in ["group", "random"]}

    # الأشخاص المسحوبون أنفسهم لكل الكميات (مزدوجة)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(gids), size=(B, len(gids)))

    def acc_point(c):
        c = np.atleast_2d(c)
        return float(np.mean(c.sum(axis=1) / total.sum()))

    def acc_boot(c):
        c = np.atleast_2d(c)
        return np.mean([ci[draws].sum(axis=1) / total[draws].sum(axis=1) for ci in c], axis=0)

    pt = {("RF", "group"): acc_point(rf["group"]), ("CNN", "group"): acc_point(cnn["group"]),
          ("RF", "random"): acc_point(rf["random"]), ("CNN", "random"): acc_point(cnn["random"])}
    bt = {("RF", "group"): acc_boot(rf["group"]), ("CNN", "group"): acc_boot(cnn["group"]),
          ("RF", "random"): acc_boot(rf["random"]), ("CNN", "random"): acc_boot(cnn["random"])}

    def row(name, p, b):
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {"quantity": name, "estimate": p, "ci_low": lo, "ci_high": hi}

    table = pd.DataFrame([
        row("RF accuracy, unseen subjects", pt[("RF", "group")], bt[("RF", "group")]),
        row("CNN accuracy, unseen subjects", pt[("CNN", "group")], bt[("CNN", "group")]),
        row("CNN minus RF, unseen subjects", pt[("CNN", "group")] - pt[("RF", "group")], bt[("CNN", "group")] - bt[("RF", "group")]),
        row("RF accuracy, random 5-fold", pt[("RF", "random")], bt[("RF", "random")]),
        row("CNN accuracy, random 5-fold", pt[("CNN", "random")], bt[("CNN", "random")]),
        row("Inflation RF (random minus unseen)", pt[("RF", "random")] - pt[("RF", "group")], bt[("RF", "random")] - bt[("RF", "group")]),
        row("Inflation CNN (random minus unseen)", pt[("CNN", "random")] - pt[("CNN", "group")], bt[("CNN", "random")] - bt[("CNN", "group")]),
    ])
    table.to_csv(f"{out}/accuracy_ci.csv", index=False)
    print(f"Accuracy over all 30 subjects (subject-level bootstrap, 95% interval, B={B})")
    print(table.round(4).to_string(index=False))
    print(f"Saved: {out}/accuracy_ci.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/cv_predictions.npz")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.pred, a.out)
