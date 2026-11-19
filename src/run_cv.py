# run_cv.py: GroupKFold (unseen subjects) versus stratified random 5-fold over all 30 subjects.
# Every window is tested exactly once under each scheme. Random Forest and the 1D CNN (3 seeds) use the same folds.
# Usage (from the project root):  python src/run_cv.py [--out results]
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import CNN_SEEDS, SEED, cm_by_subject, load_all, macro_f1_from_cm, run_cnn

K, B = 5, 5000
N_VAL_SUBJECTS, N_VAL_WINDOWS = 4, 1400


def main(out):
    os.makedirs(out, exist_ok=True)
    X, y, s, sig = load_all()
    N = len(y)
    gids = np.unique(s)
    print("Windows:", N, "| subjects:", len(gids), "| signal tensor:", sig.shape)

    group_folds = list(GroupKFold(n_splits=K).split(X, y, groups=s))
    random_folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED).split(X, y))

    fold_info = {}
    for i, (tr, te) in enumerate(group_folds):
        assert not set(s[tr]) & set(s[te]), "subject leakage in GroupKFold"
        fold_info[f"fold{i}"] = {"test_subjects": [int(v) for v in np.unique(s[te])], "n_test_windows": int(len(te))}
        print(f"GroupKFold fold {i}: test subjects {fold_info[f'fold{i}']['test_subjects']} | test windows {len(te)}")
    with open(f"{out}/groupkfold_folds.json", "w") as f:
        json.dump(fold_info, f, indent=2)

    def make_val(train_idx, scheme, fold_i):
        rng = np.random.default_rng(SEED + fold_i)
        if scheme == "group":
            val_subs = rng.choice(np.unique(s[train_idx]), size=N_VAL_SUBJECTS, replace=False)
            vm = np.isin(s[train_idx], val_subs)
        else:
            vm = np.zeros(len(train_idx), dtype=bool)
            vm[rng.choice(len(train_idx), size=N_VAL_WINDOWS, replace=False)] = True
        return train_idx[~vm], train_idx[vm]

    rf_pred = {k: np.zeros(N, dtype=int) for k in ["group", "random"]}
    cnn_pred = {k: np.zeros((len(CNN_SEEDS), N), dtype=int) for k in ["group", "random"]}
    fold_rows, t_start = [], time.time()
    for scheme, folds in [("group", group_folds), ("random", random_folds)]:
        for i, (tr_idx, te_idx) in enumerate(folds):
            fit_idx, val_idx = make_val(tr_idx, scheme, i)
            if scheme == "group":
                assert not set(s[fit_idx]) & set(s[val_idx])
                assert not set(s[fit_idx]) & set(s[te_idx])
            rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[fit_idx], y[fit_idx])
            rf_pred[scheme][te_idx] = rf.predict(X[te_idx])
            cnn_f1s, eps = [], []
            for k, seed in enumerate(CNN_SEEDS):
                p, n_ep, _ = run_cnn(sig[fit_idx], y[fit_idx], sig[val_idx], y[val_idx], sig[te_idx], seed)
                cnn_pred[scheme][k, te_idx] = p
                cnn_f1s.append(f1_score(y[te_idx], p, average="macro"))
                eps.append(n_ep)
            fold_rows.append({"scheme": scheme, "fold": i, "n_test_windows": len(te_idx),
                              "RF_macroF1": f1_score(y[te_idx], rf_pred[scheme][te_idx], average="macro"),
                              "CNN_macroF1": float(np.mean(cnn_f1s)), "CNN_epochs": eps})
            r = fold_rows[-1]
            print(f"{scheme} fold {i} | RF {r['RF_macroF1']:.4f} | CNN {r['CNN_macroF1']:.4f} | "
                  f"CNN epochs {eps} | elapsed {time.time() - t_start:.0f}s")
    for scheme in ["group", "random"]:
        assert (rf_pred[scheme] > 0).all() and (cnn_pred[scheme] > 0).all(), "some windows were never tested"

    pd.DataFrame(fold_rows).to_csv(f"{out}/cv_fold_results.csv", index=False)
    np.savez(f"{out}/cv_predictions.npz", y=y, subject=s, rf_group=rf_pred["group"], rf_random=rf_pred["random"],
             cnn_group=cnn_pred["group"], cnn_random=cnn_pred["random"])

    cms = {("RF", sc): cm_by_subject(y, rf_pred[sc], s, gids) for sc in ["group", "random"]}
    cms.update({("CNN", sc): np.array([cm_by_subject(y, p, s, gids) for p in cnn_pred[sc]]) for sc in ["group", "random"]})

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(gids), size=(B, len(gids)))   # same subject draws for every quantity (paired)

    def point(c):
        return float(np.mean(macro_f1_from_cm(c.sum(axis=-3))))

    def boot(c):
        if c.ndim == 3:
            c = c[None]
        return np.mean([macro_f1_from_cm(cs[draws].sum(axis=1)) for cs in c], axis=0)

    pt = {k: point(v) for k, v in cms.items()}
    bt = {k: boot(v) for k, v in cms.items()}

    def row(name, p, b):
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {"quantity": name, "estimate": p, "ci_low": lo, "ci_high": hi}

    ci = pd.DataFrame([
        row("RF, unseen subjects (GroupKFold)", pt[("RF", "group")], bt[("RF", "group")]),
        row("CNN, unseen subjects (GroupKFold)", pt[("CNN", "group")], bt[("CNN", "group")]),
        row("CNN minus RF, unseen subjects", pt[("CNN", "group")] - pt[("RF", "group")], bt[("CNN", "group")] - bt[("RF", "group")]),
        row("RF, random 5-fold", pt[("RF", "random")], bt[("RF", "random")]),
        row("CNN, random 5-fold", pt[("CNN", "random")], bt[("CNN", "random")]),
        row("Inflation RF (random minus group), paired", pt[("RF", "random")] - pt[("RF", "group")], bt[("RF", "random")] - bt[("RF", "group")]),
        row("Inflation CNN (random minus group), paired", pt[("CNN", "random")] - pt[("CNN", "group")], bt[("CNN", "random")] - bt[("CNN", "group")]),
    ])
    ci["interval_excludes_0"] = (ci["ci_low"] > 0) | (ci["ci_high"] < 0)
    ci.to_csv(f"{out}/cv_bootstrap_ci.csv", index=False)
    print(f"\n===== Pooled macro-F1 over all 30 subjects, subject-level bootstrap 95% (B={B}) =====")
    print(ci.round(4).to_string(index=False))
    print("(interval_excludes_0 is only meaningful for the 'minus' and 'Inflation' rows)")

    per = pd.DataFrame({"subject": gids,
                        "RF_macroF1": macro_f1_from_cm(cms[("RF", "group")]),
                        "CNN_macroF1": macro_f1_from_cm(cms[("CNN", "group")]).mean(axis=0),
                        "RF_random_macroF1": macro_f1_from_cm(cms[("RF", "random")]),
                        "CNN_random_macroF1": macro_f1_from_cm(cms[("CNN", "random")]).mean(axis=0)})
    per.to_csv(f"{out}/cv_per_subject.csv", index=False)
    for m in ["RF", "CNN"]:
        v = per[f"{m}_macroF1"]
        print(f"{m} unseen-subject macro-F1: min {v.min():.3f} | max {v.max():.3f} | subjects below 0.90: {(v < 0.90).sum()} of {len(v)}")
    print("\nTotal time: %.0f s | outputs written to %s/" % (time.time() - t_start, out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    main(ap.parse_args().out)
