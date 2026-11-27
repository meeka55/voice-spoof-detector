# Roadmap

- [x] v0.1 LFCC/MFCC + GMM/logreg/SVM, EER + per-attack EER, API, Docker, CI
- [x] Bootstrap confidence intervals for every reported EER
- [ ] Run real ASVspoof 2019 LA experiments; fill README results table and analyse failure cases
- [ ] Cross-dataset test on In-the-Wild
- [ ] Robustness: add codec/noise augmentation (e.g. MP3, 8 kHz telephony) and report the degradation
- [ ] v0.2 wav2vec2 (frozen features + small head, then partial fine-tune); compare with baselines
- [x] Score calibration + a "reject / needs review" band instead of a hard threshold
- [ ] Small Gradio/Streamlit demo
