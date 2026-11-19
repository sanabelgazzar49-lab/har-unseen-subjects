# Contribution Log

## 2026-11-12
- Chose project idea 4 from the course list (human activity recognition with smartphone sensors).
- Reorganized and archived earlier work (project_pm25_archive).
- Created the project structure and wrote the README with the fixed decisions.

## 2026-11-14
- Trained the 1D CNN (3 seeds) and ran a subject-level bootstrap on the official split.
- Ran GroupKFold vs random 5-fold over all 30 subjects (robustness check planned in the README).

## 2026-11-14
- Error analysis on all 30 subjects from saved predictions (no retraining): per-class recall, pooled confusion matrices, per-subject scores, three figures.

## 2026-11-14
- Started moving code to src/: common.py, download_data.py, requirements.txt.

## 2026-11-15
- Added explore_data.py and run_official.py to src/ (outputs of the check run in results_check/).
- Framework: PyTorch (chosen without asking the instructor; to be confirmed).

## 2026-11-15
- Added run_cv.py to src/. The check run reproduced the notebook predictions (Random Forest identical, CNN within GPU noise). The script crashed at the very end because of a print-format bug (fixed); the per-subject table will be regenerated in the final full pipeline run.

## 2026-11-15
- Added analyze_errors.py to src/. Its tables reproduced the notebook error analysis exactly.

## 2026-11-15
- Wrote README, LICENSE, .gitignore, run_all.py and make_summary.py.
- Ran the full pipeline from src/ into a clean results/ folder; the notebook-era results are in archive/results_notebook (not deleted). The numbers used in the paper come from the new results/.

## 2026-11-17
- Replaced the placeholder name in README.md and LICENSE; corrected the run-time statement in the README (the full pipeline took about 13 minutes in our run).

## 2026-11-17
- Keras project run: this folder is a separate project in which the CNN is implemented in Keras (the instructor prefers Keras). Earlier entries above describe work done before the switch.
- Ran the full pipeline from src/ in an empty results/ folder (about 22 minutes). The numbers for the paper come from this run, whatever they are.

## 2026-11-19 11:34: Pre-registered statistical analysis (written before any p-value or accuracy interval was computed)
- Unit of analysis: the subject (30 subjects). Scores: per-subject macro-F1 under GroupKFold (unseen subjects) and under the random 5-fold split, from results/cv_per_subject.csv. CNN scores are averages over its 3 seeds.
- H1: the 1D CNN and the Random Forest differ in per-subject macro-F1 on unseen subjects (CNN minus RF).
- H2: for the Random Forest, per-subject macro-F1 under the random split differs from that on unseen subjects (random minus unseen).
- H3: the same as H2 for the 1D CNN.
- H4 (exploratory, not corrected): the inflation differs between the Random Forest and the CNN (RF inflation minus CNN inflation).
- Primary test: two-sided Wilcoxon signed-rank test on the 30 paired per-subject differences. Confirmation: sign-flip permutation test on the mean difference (200,000 random sign flips, seed 2026). Multiple testing: Holm correction over H1 to H3 (family-wise level 0.05). H4 is reported separately as exploratory.
- Effect sizes: mean and median difference with a subject-level bootstrap 95% interval, and the share of subjects with a positive difference.
- Accuracy over all 30 subjects will also be reported with a subject-level bootstrap interval, because the instructor's list of evaluation outputs includes accuracy.
- No other tests will be added after seeing the results. Limitations to state: per-subject scores come from folds that share training data, so p-values may be slightly optimistic; a non-significant result does not show that two models are equivalent.

## 2026-11-19
- Added accuracy_ci.py and stat_tests.py to src/ and to run_all.py, and extended make_summary.py and the README. Run on the saved predictions and per-subject scores, the scripts reproduced the earlier notebook tables exactly.
- The full run_all.py was not repeated end to end after this change, because CNN training is not bit-for-bit reproducible on a GPU and a new run would change the CNN numbers slightly. The numbers in the paper are those saved in results/.
