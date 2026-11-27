# Recorded evaluation results

These numbers are copied from the console output of the full evaluation runs on the ASVspoof 2019 LA
`eval` split (71,237 utterances: 7,355 bona fide, 63,882 spoofed). The original `metrics.json`, `scores.csv`
and ROC plots from these runs were not retained, so the per-attack charts in this folder were **redrawn from
the recorded numbers**. ROC curves will be added when the runs are repeated.

Settings: 20 coefficients + deltas (60 dims), 25 ms window, 10 ms hop, 64-component diagonal GMMs,
seed 0, trained on all 25,380 `train` utterances, threshold chosen on a 3,000-utterance `dev` sample.

| Run | Eval EER | Eval AUC | Dev threshold | EER-optimal eval threshold | Real speech rejected @ dev thr | Spoofs accepted @ dev thr | Dev EER |
|-----|---------:|---------:|--------------:|---------------------------:|-------------------------------:|--------------------------:|--------:|
| LFCC + GMM | 6.02% | 0.9851 | 0.7464 | 1.5064 | 0.18% | 26.26% | not recorded |
| MFCC + GMM | 13.38% | 0.9343 | 1.2086 | 1.4075 | 7.93% | 17.73% | 10.76% |

Per-attack EER (%, bona fide vs a single attack):

| Run | A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LFCC + GMM | 0.40 | 0.35 | 0.10 | 8.79 | 0.03 | 3.69 | 10.71 | 4.48 | 2.54 | 0.69 | 18.74 | 3.83 | 1.81 |
| MFCC + GMM | 12.98 | 2.91 | 0.07 | 15.17 | 3.57 | 1.08 | 6.88 | 4.52 | 9.53 | 12.72 | 32.47 | 11.48 | 34.88 |

Note: the MFCC GMM fit emitted a scikit-learn convergence warning (EM iteration limit reached).
