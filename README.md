# Human Activity Recognition with Smartphone Sensors: Performance on Unseen Subjects

**Student:** Sanabel al jazzar
**Course:** Introduction to Deep Learning, MSc in AI and Data Analytics, Palestine Ahliya University

## Research question
How well can a model recognize everyday activities when it is evaluated on people who were not represented in its training data?

Additional experiment: how much does a conventional random split of windows overestimate performance compared with a subject-independent split?

## Quick start (3 steps)
Tested with Python 3 on Google Colab with a GPU runtime. A GPU is recommended for the CNN. The full pipeline took about 22 minutes in our run.

1. Install the requirements:

        pip install -r requirements.txt

2. Download the dataset (about 60 MB):

        python src/download_data.py

3. Run the whole pipeline from the project root:

        python src/run_all.py

Tables are written to results/ and figures to results/figures/. A short summary of the main results is written to results/results_summary.md.

## What each script does
| Script | Purpose |
|---|---|
| src/download_data.py | Downloads and unpacks the UCI HAR dataset into data/raw/uci_har |
| src/common.py | Shared settings, data loading, the 1D CNN and metric helpers |
| src/explore_data.py | Descriptive summary and four figures of the data (no models) |
| src/run_official.py | Official split: 17 fit, 4 validation and 9 test subjects, versus a random window split of the same sizes. Random Forest and 1D CNN, subject-level bootstrap |
| src/run_cv.py | GroupKFold (unseen subjects) versus stratified random 5-fold over all 30 subjects, Random Forest and 1D CNN with 3 seeds, paired subject-level bootstrap |
| src/analyze_errors.py | Per-class recall, pooled confusion matrices, per-subject scores, three figures (uses saved predictions, no training) |
| src/accuracy_ci.py | Accuracy over all 30 subjects with subject-level bootstrap intervals (uses saved predictions) |
| src/stat_tests.py | Pre-registered tests on per-subject macro-F1 differences: Wilcoxon signed-rank and permutation tests, Holm correction, effect sizes (uses saved scores) |
| src/make_summary.py | Builds results/results_summary.md from the result tables |
| src/run_all.py | Runs the scripts above in order |

## Design decisions fixed before looking at test results
- Dataset: UCI Human Activity Recognition Using Smartphones (CC BY 4.0).
- Simple baseline: Random Forest (300 trees, fixed settings) on the 561 provided features.
- Deep learning model: small 1D CNN (Keras, TensorFlow backend) on the raw signals (128 readings x 9 channels), 3 random seeds, early stopping on validation macro-F1.
- Primary metric: macro-F1 on subjects not seen in training. Secondary: accuracy, per-class recall, confusion matrices.
- Validation data always comes from training subjects only; windows of one subject never appear in two different sets under subject-independent splits.
- Uncertainty: bootstrap resampling at the subject level.
- Robustness check planned in advance: GroupKFold over all 30 subjects.
- No hyperparameter tuning was done for either model.
- The statistical tests (H1 to H4) were recorded in CONTRIBUTIONS.md before any p-value was computed. H4 is exploratory and uncorrected.

## Main results
The pipeline writes the main tables to results/results_summary.md (macro-F1 and accuracy with 95% subject-level bootstrap intervals, per-class recall, statistical tests, official-split results).

## Data
- Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K
- Link: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- License: CC BY 4.0
- Download date: 2026-11-12
- 30 volunteers, 6 activities, waist-mounted smartphone, 50 Hz accelerometer and gyroscope, labels created manually from video. 2.56 s windows with 50% overlap (10,299 windows). Raw files are not modified; everything is computed in code.

## Known limitations
- Only 30 subjects, one phone model and body position, ages 19 to 48, scripted activities. The results must not be read as evidence about real-world performance.
- The two models differ in both model family and input (engineered features versus raw signals), so a difference cannot be attributed to the model alone.
- Neither model was tuned. GPU training is not bit-for-bit deterministic: CNN scores can vary slightly between runs.
- Subject-level bootstrap intervals with 30 subjects are approximate. Intervals for the inflation under the official split treat the two evaluations as independent.
- Per-subject scores come from folds that share training data, so p-values may be slightly optimistic. A non-significant result (for example the CNN versus the Random Forest) does not show that two models are equivalent.
- Consecutive windows overlap by 50%, which is why random window splits leak information.

## Ethics and privacy
The dataset contains anonymous subject IDs. Movement data can in principle identify people, and the consent procedure was not verified from the sources consulted here. No personal data was added.

## AI assistance disclosure
The code and the analysis were developed with the help of an AI assistant (Claude, by Anthropic). The student ran all experiments and is responsible for the results. Adapt this statement to the course AI policy.

## License and citation
Code: MIT License (see LICENSE). If you use this work, cite the dataset above.
