# VoxGuard

VoxGuard detects genuine versus AI-generated speech using acoustic classifiers and a spectrogram CNN. This is a CSCI 475 research prototype with preliminary results on ASVspoof 2019 Logical Access recordings.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asarekwaku/voxguard/blob/main/notebooks/VoxGuard_Colab.ipynb)

## Run in Colab

Open the notebook, select **Runtime → Change runtime type → GPU**, and run the cells in order. It downloads real recordings, trains the baseline and improved classifiers, trains the CNN, and exports evaluation results. Checkpoints and reports are saved to Google Drive.

Defaults are 6,000 training, 1,200 development, and 2,000 evaluation recordings. For a quick functional run, use 1,200 / 400 / 400 and three CNN epochs. The notebook excludes all 1,400 evaluation recordings inspected in our local experiments.

## Results so far

Both models below were tested on the same fresh, balanced subset of 1,000 recordings after development-only selection.

| Condition | Original MFCC SVM | Augmented MFCC/LFCC SVM |
|---|---:|---:|
| Clean | 86.4% | 89.4% |
| White noise at 20 dB SNR | 50.0% | 83.8% |
| White noise at 10 dB SNR | 50.0% | 76.7% |

These are subset results, not full-benchmark or deployment estimates. Some synthesis methods remain difficult. The CNN has passed local software and small training/resume checks; its substantive GPU experiment is pending.

See [experiment details](docs/experiments.md) and [machine-readable metrics](results/metrics.json).

## Repository layout

```text
voxguard/      Audio processing, models, training, and evaluation
notebooks/     Google Colab workflow
scripts/       Official-dataset subset downloader
tests/         Software tests
protocols/     Previously inspected evaluation IDs to exclude
results/       Compact recorded experiment metrics
docs/          Experiment methods, findings, and limitations
```

Generated models, predictions, caches, and logs stay outside version control. The exclusion list under `protocols/` is retained because it protects future evaluations from reusing inspected recordings.

## Local setup

Python 3.12 was used.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-neural.txt
python -m pytest -q
```

Use `requirements.txt` if you only need the classical models. The current suite has 29 tests; two neural tests are skipped when PyTorch is absent.

## Trained models and detailed outputs

[View the original experiment outputs](https://github.com/asarekwaku/voxguard/tree/c181261e298390a767e71c56cd09669d43d50b86/runs) for trained models, per-recording predictions, plots, test logs, frozen settings, and manifests. They remain available in Git history without cluttering the current source tree. The snapshot also preserves the original package lock file and experiment notes.

After training your own acoustic model, predict an audio file with:

```bash
python -m voxguard.cli predict --model runs/my_experiment/selected.joblib --audio /path/to/recording.wav
```

The model argument must point to a trained acoustic model bundle. Scores are not calibrated confidence, and predictions are not proof of authenticity.

## Dataset attribution

[ASVspoof 2019](https://doi.org/10.7488/ds/2555), University of Edinburgh DataShare, Open Data Commons Attribution License (ODC-By).

M. Todisco et al., “ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection,” Interspeech 2019, pp. 1008–1012, doi:10.21437/Interspeech.2019-2249.
