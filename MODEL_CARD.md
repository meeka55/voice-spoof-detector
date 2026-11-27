# Model card: voice-spoof-detector baselines (v0.1)

## Intended use
Research and portfolio demonstration of synthetic-speech detection. Produces a score and a bona fide / spoof
label for a single audio clip. **Not** validated for production fraud decisions.

## Models
LFCC or MFCC (20 coefficients + deltas, 60 dimensions) with a two-class GMM log-likelihood-ratio back-end
(64 diagonal-covariance components per class). Logistic-regression and SVM back-ends exist in the code but are not
part of the reported results.

## Training data
ASVspoof 2019 Logical Access `train` split (25,380 utterances: 2,580 bona fide, 22,800 spoofed, attacks A01-A06).
A 3,000-utterance sample of `dev` is used to choose the decision threshold.

## Evaluation
ASVspoof 2019 LA `eval` split, scored once: 71,237 utterances (7,355 bona fide, 63,882 spoofed, attacks A07-A19).

| Model | Eval EER | Eval AUC |
|-------|---------:|---------:|
| LFCC + GMM | 6.02% | 0.985 |
| MFCC + GMM | 13.38% | 0.934 |

Per-attack results and plots are in the README and `docs/results/`.

## Known limitations
* Per-attack EER varies widely: for LFCC + GMM it ranges from about 0.03% (A11) to 18.7% (A17); MFCC + GMM reaches
  34.9% on A19. New TTS/VC systems may evade detection.
* The threshold chosen on dev accepted 26.3% of eval spoofs (LFCC + GMM): it does not transfer to unseen attacks.
* Clean read English speech from one corpus; telephony codecs, background noise and re-recording are untested.
* Single run per configuration, no confidence intervals; the MFCC GMM fit did not fully converge.

## Ethical considerations
False rejects can disadvantage genuine speakers (accents, speech disorders, poor microphones). Do not use the
score as the sole basis for denying someone service. No personal data is stored by the API; audio is processed in memory.
