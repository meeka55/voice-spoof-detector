# Model card: voice-spoof-detector baselines (v0.1)

## Intended use
Research and portfolio demonstration of synthetic-speech detection. Produces a score and a bona fide / spoof
label for a single audio clip. **Not** validated for production fraud decisions.

## Models
LFCC or MFCC (20 coefficients + deltas) with a GMM log-likelihood-ratio, logistic regression, or RBF SVM back-end.

## Training data
ASVspoof 2019 Logical Access `train` split (and `dev` for threshold selection). In-the-Wild or custom data can be
plugged in through a CSV manifest. State here which you actually used: _fill in_.

## Evaluation
Equal error rate (EER), AUC, false accept / false reject rate at the dev-chosen threshold, and per-attack EER.
Results: _fill in from results/*/metrics.json_.

## Known limitations
* Attack coverage is limited to the generators in the training corpus; new TTS/VC systems may evade detection.
* Clean read speech only; telephony codecs, background noise, and re-recording change the acoustics.
* Corpus bias: speaker, language (English) and recording conditions are narrow.
* Threshold set on dev is dataset-specific; recalibrate for any new deployment domain.

## Ethical considerations
False rejects can disadvantage genuine speakers (accents, speech disorders, poor microphones). Do not use the
score as the sole basis for denying someone service. No personal data is stored by the API; audio is processed in memory.
