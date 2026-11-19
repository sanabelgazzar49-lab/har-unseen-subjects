# stat_tests.py: الاختبارات الإحصائية المسجّلة مسبقاً في CONTRIBUTIONS.md (وحدة التحليل: الشخص).
# H1: CNN ناقص RF على أشخاص جدد. H2: RF عشوائي ناقص أشخاص جدد. H3: الشيء نفسه لـ CNN.
# H4 استكشافية: مبالغة RF ناقص مبالغة CNN (بلا تصحيح).
# الاستخدام (من جذر المشروع):  python src/stat_tests.py [--scores results/cv_per_subject.csv] [--out results]
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED, B, N_FLIPS = 2026, 5000, 200000


def holm(ps):
    # تصحيح Holm للمقارنات المتعددة
    ps = np.asarray(ps, float)
    m, adj, run = len(ps), np.empty(len(ps)), 0.0
    for rank, idx in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (m - rank) * ps[idx]))
        adj[idx] = run
    return adj


def main(scores_path, out):
    os.makedirs(out, exist_ok=True)
    per = pd.read_csv(scores_path)
    assert len(per) == 30 and per["subject"].is_unique

    # الفروق الأربعة لكل شخص
    h2 = per["RF_random_macroF1"] - per["RF_macroF1"]
    h3 = per["CNN_random_macroF1"] - per["CNN_macroF1"]
    D = {
        "H1": ("CNN minus RF, unseen subjects", per["CNN_macroF1"] - per["RF_macroF1"]),
        "H2": ("RF: random minus unseen", h2),
        "H3": ("CNN: random minus unseen", h3),
        "H4": ("RF inflation minus CNN inflation (exploratory)", h2 - h3),
    }

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, 30, size=(B, 30))     # الأشخاص المسحوبون أنفسهم لكل الفرضيات
    flip_rng = np.random.default_rng(SEED)

    def perm_p(v):
        # اختبار التبديل: نقلب إشارة كل فرق عشوائياً ونعدّ كم مرة يبلغ متوسط القيمة المطلقة حجم المرصود أو أكثر
        obs = abs(v.mean())
        count, done = 0, 0
        while done < N_FLIPS:
            n = min(50000, N_FLIPS - done)
            signs = flip_rng.integers(0, 2, size=(n, len(v))) * 2 - 1
            count += int((np.abs((signs * v).mean(axis=1)) >= obs - 1e-12).sum())
            done += n
        return (count + 1) / (N_FLIPS + 1)

    rows = []
    for key, (desc, s) in D.items():
        v = s.to_numpy(float)
        w = wilcoxon(v, zero_method="wilcox", alternative="two-sided")
        bm = v[draws].mean(axis=1)
        bd = np.median(v[draws], axis=1)
        rows.append({
            "hyp": key, "description": desc, "n_nonzero": int((v != 0).sum()),
            "mean_diff": v.mean(), "mean_ci_low": np.percentile(bm, 2.5), "mean_ci_high": np.percentile(bm, 97.5),
            "median_diff": np.median(v), "median_ci_low": np.percentile(bd, 2.5), "median_ci_high": np.percentile(bd, 97.5),
            "subjects_positive": int((v > 0).sum()),
            "p_wilcoxon": float(w.pvalue), "p_permutation": perm_p(v),
        })
    res = pd.DataFrame(rows)

    # تصحيح Holm على H1 إلى H3 فقط؛ و H4 استكشافية بلا تصحيح
    main_rows = res["hyp"].isin(["H1", "H2", "H3"])
    res["p_wilcoxon_holm"] = np.nan
    res["p_permutation_holm"] = np.nan
    res.loc[main_rows, "p_wilcoxon_holm"] = holm(res.loc[main_rows, "p_wilcoxon"])
    res.loc[main_rows, "p_permutation_holm"] = holm(res.loc[main_rows, "p_permutation"])
    res.to_csv(f"{out}/stat_tests.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"Effect sizes (per-subject differences, subject-level bootstrap 95% intervals, B={B})")
    print(res[["hyp", "description", "n_nonzero", "mean_diff", "mean_ci_low", "mean_ci_high",
               "median_diff", "median_ci_low", "median_ci_high", "subjects_positive"]].round(4).to_string(index=False))
    print("\nTests (two-sided). Holm correction covers H1-H3 only; H4 is exploratory and uncorrected.")
    print(res[["hyp", "p_wilcoxon", "p_wilcoxon_holm", "p_permutation", "p_permutation_holm"]].round(5).to_string(index=False))
    print(f"Saved: {out}/stat_tests.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/cv_per_subject.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.scores, a.out)
