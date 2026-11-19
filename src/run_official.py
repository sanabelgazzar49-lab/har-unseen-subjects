# run_official.py: the official subject-independent split (17 fit / 4 validation / 9 test subjects)
# versus a random window split of the same sizes, for Random Forest and the 1D CNN,
# with subject-level bootstrap intervals.
# Usage (from the project root):  python src/run_official.py [--out results]
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split

from common import (CNN_SEEDS, LABELS, SEED, activity_names, cm_by_subject, load_split,
                    macro_f1_from_cm, run_cnn)

B = 5000


def main(out):
    os.makedirs(out, exist_ok=True)
    names = activity_names()
    Xtr_all, ytr_all, str_all, sigtr_all = load_split("train")
    Xte, yte, ste, sigte = load_split("test")

    # ---- Subject-independent split: validation subjects are taken from the training subjects ----
    rng = np.random.default_rng(SEED)
    train_subjects = np.unique(str_all)
    val_subjects = np.sort(rng.choice(train_subjects, size=4, replace=False))
    fit_subjects = np.setdiff1d(train_subjects, val_subjects)
    test_subjects = np.unique(ste)
    assert not set(fit_subjects) & set(val_subjects)
    assert not set(fit_subjects) & set(test_subjects)
    assert not set(val_subjects) & set(test_subjects)
    fit_mask = np.isin(str_all, fit_subjects)
    val_mask = np.isin(str_all, val_subjects)
    X_fit, y_fit, sig_fit = Xtr_all[fit_mask], ytr_all[fit_mask], sigtr_all[fit_mask]
    X_val, y_val, sig_val = Xtr_all[val_mask], ytr_all[val_mask], sigtr_all[val_mask]
    split_info = {"seed": SEED, "fit_subjects": [int(x) for x in fit_subjects],
                  "val_subjects": [int(x) for x in val_subjects],
                  "test_subjects": [int(x) for x in test_subjects]}
    if os.path.exists("results/subject_splits.json"):
        with open("results/subject_splits.json") as f:
            assert json.load(f) == split_info, "the subject split differs from the saved one"
    with open(f"{out}/subject_splits.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print("Fit subjects:", split_info["fit_subjects"])
    print("Validation subjects:", split_info["val_subjects"])
    print("Test subjects:", split_info["test_subjects"])
    print(f"Windows -> fit {len(y_fit)} | validation {len(y_val)} | test {len(yte)}")

    # ---- Random window split with the same sizes (deliberately leaky reference) ----
    X_all = np.vstack([Xtr_all, Xte])
    y_all = np.concatenate([ytr_all, yte])
    s_all = np.concatenate([str_all, ste])
    sig_all = np.concatenate([sigtr_all, sigte])
    idx = np.arange(len(y_all))
    idx_tr, idx_te = train_test_split(idx, train_size=len(y_fit), test_size=len(yte),
                                      stratify=y_all, random_state=SEED)
    rest = np.setdiff1d(idx, np.concatenate([idx_tr, idx_te]))   # validation windows for the CNN
    if os.path.exists("results/random_split_indices.npz"):
        saved = np.load("results/random_split_indices.npz")
        assert (saved["train"] == idx_tr).all() and (saved["test"] == idx_te).all(), \
            "the random split differs from the saved one"
    np.savez(f"{out}/random_split_indices.npz", train=idx_tr, test=idx_te)

    # ---- Random Forest (fixed settings, no tuning) ----
    def make_rf():
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)

    t0 = time.time()
    rf_s = make_rf().fit(X_fit, y_fit)
    rf_pred_val = rf_s.predict(X_val)
    rf_pred_subject = rf_s.predict(Xte)
    rf_r = make_rf().fit(X_all[idx_tr], y_all[idx_tr])
    rf_pred_random = rf_r.predict(X_all[idx_te])
    y_te_r = y_all[idx_te]
    print("Random Forest done in %.0f s" % (time.time() - t0))

    summary = pd.DataFrame([
        {"setting": "Subject-independent", "evaluated_on": "validation subjects",
         "macro_f1": f1_score(y_val, rf_pred_val, average="macro"), "accuracy": accuracy_score(y_val, rf_pred_val)},
        {"setting": "Subject-independent", "evaluated_on": "test subjects (PRIMARY)",
         "macro_f1": f1_score(yte, rf_pred_subject, average="macro"), "accuracy": accuracy_score(yte, rf_pred_subject)},
        {"setting": "Random window split", "evaluated_on": "random test windows",
         "macro_f1": f1_score(y_te_r, rf_pred_random, average="macro"), "accuracy": accuracy_score(y_te_r, rf_pred_random)},
    ])
    summary.to_csv(f"{out}/rf_summary.csv", index=False)
    print("\n===== Random Forest =====")
    print(summary.round(4).to_string(index=False))
    pd.DataFrame({"subject_independent_test": recall_score(yte, rf_pred_subject, average=None, labels=LABELS),
                  "random_split_test": recall_score(y_te_r, rf_pred_random, average=None, labels=LABELS)},
                 index=names).to_csv(f"{out}/rf_per_class_recall.csv")
    np.savez(f"{out}/rf_predictions.npz", pred_subject=rf_pred_subject, pred_random=rf_pred_random)

    # ---- 1D CNN: 3 seeds in each setting ----
    settings = {
        "subject_independent": (sig_fit, y_fit, sig_val, y_val, sigte, yte),
        "random_window": (sig_all[idx_tr], y_all[idx_tr], sig_all[rest], y_all[rest], sig_all[idx_te], y_te_r),
    }
    rows, preds, truths = [], {}, {}
    print("\n===== 1D CNN =====")
    for setting, (sf, yf, sv, yv, st, yt) in settings.items():
        truths[setting], preds[setting] = yt, []
        for seed in CNN_SEEDS:
            t0 = time.time()
            p, n_ep, best_val = run_cnn(sf, yf, sv, yv, st, seed)
            preds[setting].append(p)
            rows.append({"setting": setting, "seed": seed, "epochs": n_ep, "val_macro_f1": best_val,
                         "test_macro_f1": f1_score(yt, p, average="macro"),
                         "test_accuracy": accuracy_score(yt, p), "seconds": time.time() - t0})
            r = rows[-1]
            print(f"{setting} | seed {seed} | epochs {n_ep} | val F1 {best_val:.4f} | "
                  f"test F1 {r['test_macro_f1']:.4f} | test acc {r['test_accuracy']:.4f}")
    runs = pd.DataFrame(rows)
    runs.to_csv(f"{out}/cnn_runs.csv", index=False)
    cnn_sub = np.array(preds["subject_independent"])
    cnn_rnd = np.array(preds["random_window"])
    np.savez(f"{out}/cnn_predictions.npz", y_test_subject=yte, pred_subject=cnn_sub,
             y_test_random=y_te_r, pred_random=cnn_rnd, idx_te_random=idx_te)
    print("\nCNN mean test macro-F1 over 3 seeds:")
    print(runs.groupby("setting")["test_macro_f1"].agg(["mean", "std"]).round(4).to_string())
    rec = {k: np.mean([recall_score(truths[k], p, average=None, labels=LABELS) for p in preds[k]], axis=0)
           for k in preds}
    pd.DataFrame(rec, index=names).to_csv(f"{out}/cnn_per_class_recall.csv")
    cm0 = confusion_matrix(yte, cnn_sub[0], labels=LABELS)
    pd.DataFrame(cm0, index=["true_" + n for n in names],
                 columns=["pred_" + n for n in names]).to_csv(f"{out}/cnn_confusion_subject_seed0.csv")

    # ---- Per-subject results (9 test subjects) ----
    gids = np.unique(ste)
    cm_rf = cm_by_subject(yte, rf_pred_subject, ste, gids)
    cm_cnn = np.array([cm_by_subject(yte, p, ste, gids) for p in cnn_sub])
    per = pd.DataFrame({"subject": gids, "n_windows": [int((ste == g).sum()) for g in gids],
                        "RF_macroF1": macro_f1_from_cm(cm_rf),
                        "CNN_macroF1": macro_f1_from_cm(cm_cnn).mean(axis=0)})
    per.to_csv(f"{out}/per_subject_results.csv", index=False)

    # ---- Subject-level bootstrap ----
    rng = np.random.default_rng(SEED)
    G = len(gids)
    draws = rng.integers(0, G, size=(B, G))

    def boot(cm_g, d):
        return macro_f1_from_cm(cm_g[d].sum(axis=1))

    b_rf = boot(cm_rf, draws)
    b_cnn = np.mean([boot(c, draws) for c in cm_cnn], axis=0)
    pt_rf = float(macro_f1_from_cm(cm_rf.sum(axis=0)))
    pt_cnn = float(macro_f1_from_cm(cm_cnn.sum(axis=1)).mean())

    s_te_r = s_all[idx_te]
    gids_all = np.unique(s_all)
    cm_rf_r = cm_by_subject(y_te_r, rf_pred_random, s_te_r, gids_all)
    cm_cnn_r = np.array([cm_by_subject(y_te_r, p, s_te_r, gids_all) for p in cnn_rnd])
    draws_r = rng.integers(0, len(gids_all), size=(B, len(gids_all)))
    b_rf_r = boot(cm_rf_r, draws_r)
    b_cnn_r = np.mean([boot(c, draws_r) for c in cm_cnn_r], axis=0)
    pt_rf_r = float(macro_f1_from_cm(cm_rf_r.sum(axis=0)))
    pt_cnn_r = float(macro_f1_from_cm(cm_cnn_r.sum(axis=1)).mean())

    def row(name, point, samples):
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return {"quantity": name, "estimate": point, "ci_low": lo, "ci_high": hi,
                "ci_excludes_0": bool(lo > 0 or hi < 0)}

    ci = pd.DataFrame([
        row("RF macro-F1, unseen subjects", pt_rf, b_rf),
        row("CNN macro-F1, unseen subjects", pt_cnn, b_cnn),
        row("CNN minus RF, unseen subjects", pt_cnn - pt_rf, b_cnn - b_rf),
        row("RF macro-F1, random window split", pt_rf_r, b_rf_r),
        row("CNN macro-F1, random window split", pt_cnn_r, b_cnn_r),
        row("Inflation for RF (random minus unseen)", pt_rf_r - pt_rf, b_rf_r - b_rf),
        row("Inflation for CNN (random minus unseen)", pt_cnn_r - pt_cnn, b_cnn_r - b_cnn),
    ])
    ci.to_csv(f"{out}/subject_bootstrap_ci.csv", index=False)
    print("\n===== Subject-level bootstrap, 95% intervals (official 9 test subjects) =====")
    print(ci.round(4).to_string(index=False))
    print("Note: inflation intervals treat the two evaluations as independent (approximation).")
    print(f"\nAll outputs written to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    main(ap.parse_args().out)
