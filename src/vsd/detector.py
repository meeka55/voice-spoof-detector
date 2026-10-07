"""Detector: feature front-end + back-end behind one small interface.

Backends
--------
gmm     Two diagonal-covariance GMMs (bona fide / spoof) on frame features; the score is the
        mean per-frame log-likelihood ratio. This mirrors the official ASVspoof baselines.
logreg  Logistic regression on utterance mean/std statistics.
svm     RBF SVM on utterance mean/std statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import BinaryIO

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FeatureConfig, frame_features, load_audio, utterance_stats

BACKENDS = ("gmm", "logreg", "svm")


def _extract_train(path: str, cfg: FeatureConfig, frames_per_utt: int, seed: int):
    y = load_audio(path, cfg.sr, cfg.max_seconds)
    frames = frame_features(y, cfg)
    stats = utterance_stats(frames)
    if len(frames) > frames_per_utt:
        idx = np.random.default_rng(seed).choice(len(frames), frames_per_utt, replace=False)
        frames = frames[idx]
    return frames, stats


class Detector:
    def __init__(
        self,
        feature_cfg: FeatureConfig | None = None,
        backend: str = "gmm",
        n_components: int = 64,
        seed: int = 0,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.feature_cfg = feature_cfg or FeatureConfig()
        self.backend = backend
        self.n_components = n_components
        self.seed = seed
        self.threshold: float = 0.0  # decision threshold; set from the dev set after training
        self.meta: dict = {}
        self._gmm_bona = None
        self._gmm_spoof = None
        self._clf = None

    # ---------------------------------------------------------------- training
    def fit(
        self,
        paths: Sequence[str],
        labels: Sequence[int],
        n_jobs: int = -1,
        frames_per_utt: int = 60,
    ) -> Detector:
        labels = np.asarray(labels, dtype=int)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_extract_train)(p, self.feature_cfg, frames_per_utt, self.seed + i)
            for i, p in enumerate(paths)
        )
        if self.backend == "gmm":
            bona = np.concatenate([r[0] for r, y in zip(results, labels) if y == 1])
            spoof = np.concatenate([r[0] for r, y in zip(results, labels) if y == 0])
            k = self.n_components
            self._gmm_bona = GaussianMixture(
                k, covariance_type="diag", max_iter=100, random_state=self.seed
            ).fit(bona)
            self._gmm_spoof = GaussianMixture(
                k, covariance_type="diag", max_iter=100, random_state=self.seed
            ).fit(spoof)
        else:
            X = np.stack([r[1] for r in results])
            if self.backend == "logreg":
                model = LogisticRegression(max_iter=3000, class_weight="balanced")
            else:
                model = SVC(kernel="rbf", class_weight="balanced")
            self._clf = make_pipeline(StandardScaler(), model).fit(X, labels)
        return self

    # ----------------------------------------------------------------- scoring
    def _score_frames(self, frames: np.ndarray) -> float:
        if self.backend == "gmm":
            return float(
                np.mean(self._gmm_bona.score_samples(frames) - self._gmm_spoof.score_samples(frames))
            )
        stats = utterance_stats(frames)[None, :]
        return float(self._clf.decision_function(stats)[0])

    def score_audio(self, y: np.ndarray) -> float:
        return self._score_frames(frame_features(y, self.feature_cfg))

    def score_file(self, src: str | BinaryIO) -> float:
        return self.score_audio(load_audio(src, self.feature_cfg.sr, self.feature_cfg.max_seconds))

    def score_many(self, paths: Sequence[str], n_jobs: int = -1) -> np.ndarray:
        scores = Parallel(n_jobs=n_jobs)(delayed(self.score_file)(p) for p in paths)
        return np.asarray(scores, dtype=float)

    def predict_label(self, score: float) -> str:
        return "bonafide" if score >= self.threshold else "spoof"

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str) -> Detector:
        obj = joblib.load(path)
        if not isinstance(obj, Detector):
            raise TypeError("file does not contain a Detector")
        return obj
