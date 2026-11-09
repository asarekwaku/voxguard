# VoxGuard experiments

Experiments were run on November 8, 2026. The compact numeric record is [results/metrics.json](../results/metrics.json). Complete outputs and trained models remain in the [original experiment snapshot](https://github.com/asarekwaku/voxguard/tree/c181261e298390a767e71c56cd09669d43d50b86/runs).

## First baseline

The first pilot sampled 1,200 training, 400 development, and 400 evaluation recordings directly from the official ASVspoof 2019 Logical Access archive, using seed 475. Each partition was balanced between genuine and synthetic speech. Synthetic recordings were approximately balanced across attack IDs. The official partitions were retained, and the loader checked for overlapping utterance IDs, speakers, and audio hashes.

Audio was converted to mono and 16 kHz, peak-normalized, and limited to its first four seconds. Valid shorter clips used their available frames without duration padding. Twenty MFCCs and first temporal differences were summarized by means and standard deviations, producing 80 features. Files shorter than 0.1 seconds and silent recordings were rejected.

Candidate settings were Logistic Regression C in {0.1, 1, 10}; RBF SVM C in {0.1, 1, 10}, gamma=scale; and Random Forest with 200 trees and depth in {8, unlimited}. StandardScaler was fit only on training data. Hyperparameters and model choice used clean development balanced accuracy. Default classifier thresholds were retained.

| Model | Development accuracy | First evaluation accuracy | Evaluation ROC-AUC | Evaluation EER |
|---|---:|---:|---:|---:|
| Trivial reference | 50.00% | 50.00% | 0.5000 | 50.0% |
| Logistic Regression | 82.50% | 84.50% | 0.8908 | 18.5% |
| RBF SVM | 89.75% | 87.00% | 0.9303 | 14.5% |
| Random Forest | 87.75% | 83.25% | 0.8964 | 18.5% |

Accuracy equals balanced accuracy in these balanced samples. The trivial classifier chose genuine speech after a training-class tie. EER is interpolated from the ROC crossing; it does not select an evaluation-set operating threshold.

The SVM was selected using development data. On the first evaluation sample, it incorrectly flagged 21/200 genuine recordings and missed 31/200 synthetic recordings. Added Gaussian white noise at 20 dB and 10 dB SNR caused it to label every recording synthetic, reducing accuracy to 50%.

## Improved acoustic model

We compared MFCC, LFCC, and combined representations with Logistic Regression, RBF SVM, and Random Forest, for 15 candidate configurations. Each of the same 1,200 source training recordings contributed a clean view plus two independently seeded white-noise views, with SNR drawn uniformly from 8 to 35 dB. This produced 3,600 training views.

The LFCC representation uses 128 triangular linear-frequency bands, a 1,024-point FFT, a 400-sample window, a 160-sample hop, and 40 cepstral coefficients. Mean and standard deviation of coefficients and temporal differences produce 160 LFCC features. Concatenation with the original 80 MFCC features produces 240 features. This is our implementation, not an exact reproduction of the official LFCC challenge baselines.

Candidates were compared using the same 400 development recordings under clean, 20 dB, and 10 dB conditions. Selection maximized mean balanced accuracy across the three conditions. One score threshold was chosen from pooled development scores. Evaluation data did not select settings or thresholds.

The selected model was the combined-feature RBF SVM, C=1, gamma=scale, threshold -0.029748856495285136. Its mean development balanced accuracy was 85.75% across the three conditions.

## Fresh held-out comparison

After selection was frozen, we sampled 1,000 fresh evaluation recordings using seed 1475: 500 genuine and 500 synthetic recordings, with synthetic examples approximately balanced across A07–A19. All 400 previously inspected evaluation recordings were excluded by ID and hash. Evaluation IDs, speaker IDs, and hashes were also checked against each model's training data.

Both models used the same source training recordings. They were tested on identical held-out audio and identical seeded noise realizations, with settings and thresholds fixed before scoring.

| Condition | Original accuracy | Improved accuracy | Original ROC-AUC | Improved ROC-AUC | Original EER | Improved EER |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 86.4% | 89.4% | 0.9258 | 0.9552 | 14.0% | 10.8% |
| White noise at 20 dB | 50.0% | 83.8% | 0.4938 | 0.9138 | 51.6% | 16.0% |
| White noise at 10 dB | 50.0% | 76.7% | 0.5853 | 0.8373 | 43.4% | 23.6% |

The original model's 86.4% here and its earlier 87.0% were measured on different evaluation samples. The matched comparison above is the appropriate measure of improvement.

On clean audio, genuine false alarms decreased from 55/500 to 31/500 and missed synthetic recordings decreased from 81/500 to 75/500. Improvements were not uniform across attacks: A19 recall rose from 14/38 to 38/38, while A17 fell from 10/38 to 5/38 and A18 fell from 16/38 to 10/38. Per-recording and per-attack records remain in the original experiment snapshot.

## CNN and Colab

The Colab notebook uses 6,000 training, 1,200 development, and 2,000 evaluation recordings by default. It excludes all 1,400 locally inspected evaluation recordings, retrains an MFCC baseline in its current environment, trains augmented acoustic models, and trains a log-Mel CNN.

The CNN has three convolution blocks, group normalization, padding-aware temporal pooling, and a binary output. It uses 64-band log-Mel spectrograms, training-only normalization with padding excluded from statistics, noise augmentation, and training-only frequency/time masks. Early stopping and best-epoch selection use average development balanced accuracy across clean, 20 dB, and 10 dB conditions.

Models and epoch checkpoints are saved to Drive, with data and caches on Colab's temporary disk. Interrupted CNN training can resume with matching training settings and data. GPU operations may not be bit-for-bit deterministic across hardware.

The CNN passed forward/backward tests, a one-epoch CPU training check, checkpoint recovery through epoch two, and saved-checkpoint inference. The smoke run used only 26 training and 26 development recordings. It validates functionality, not detection performance. No substantive CNN GPU result is claimed yet.

## Software testing

All 29 tests passed locally with PyTorch installed. Tests cover audio conversion, stereo input, feature stability, cropping, invalid input rejection, metric orientation and EER edge cases, partition leakage, class coverage, serialization, training-only scaling, seeded sampling, LFCC dimensions, requested noise SNR, threshold orientation, inspected-evaluation exclusion, and CNN forward/backward behavior. Neural tests are skipped without the optional dependency.

Separate real-data checks exercised downloading, training, evaluation, noisy evaluation, saved-model single-file inference, and resumed CNN inference. Evaluation summaries were independently recomputed from saved per-recording predictions and matched. The original experiment snapshot contains machine-readable test logs and check records.

## Limits and next work

These are class-balanced and approximately attack-balanced subset results, not full-benchmark or deployment estimates. They use one held-out sampling seed without confidence intervals or statistical significance tests. Gaussian white noise does not represent all environments. Audio beyond the first four seconds is discarded, and some attack types remain difficult. A different attack ID does not necessarily mean a wholly unfamiliar generation method.

The improved experiment changed features, augmentation, the selection objective, and threshold selection together. Ablation experiments are needed to attribute gains to individual changes. Compression, recorded background noise, newer generators, multilingual speech, cross-dataset transfer, and the web interface remain future work.

The inspected-ID list in `protocols/inspected_evaluation.csv` is required for fresh evaluations. Later experiments must also exclude any newly inspected evaluation recordings rather than tune on them.
