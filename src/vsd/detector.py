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
    # Class-level defaults so models pickled before these fields existed still load.
    accept_threshold: float | None = None
    reject_threshold: float | None = None

    def __init__(
        self,
        feature_cfg: FeatureConfig | None = None,
        backend: str = "gmm",
        n_components: int = 64,
        seed: int = 0,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if isinstance(n_components, bool) or not isinstance(n_components, int):
            raise TypeError(f"n_components must be an integer, got {n_components!r}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
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
        y = np.asarray(labels, dtype=int)
        paths = list(paths)
        if len(paths) != y.size:
            raise ValueError(
                f"paths and labels differ in length ({len(paths)} vs {y.size}); "
                "a shorter label list would silently drop utterances"
            )
        if not paths:
            raise ValueError("cannot fit on an empty set of paths")
        if isinstance(frames_per_utt, bool) or not isinstance(frames_per_utt, int):
            raise TypeError(f"frames_per_utt must be an integer, got {frames_per_utt!r}")
        if frames_per_utt < 1:
            raise ValueError(f"frames_per_utt must be >= 1, got {frames_per_utt}")
        n_bona, n_spoof = int((y == 1).sum()), int((y == 0).sum())
        if n_bona == 0 or n_spoof == 0:
            missing = "bona fide" if n_bona == 0 else "spoof"
            raise ValueError(f"training needs both classes; no {missing} utterances in the input")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_extract_train)(p, self.feature_cfg, frames_per_utt, self.seed + i)
            for i, p in enumerate(paths)
        )

        if self.backend == "gmm":
            bona = np.concatenate([r[0] for r, lab in zip(results, y) if lab == 1])
            spoof = np.concatenate([r[0] for r, lab in zip(results, y) if lab == 0])
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
            self._clf = make_pipeline(StandardScaler(), model).fit(X, y)
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

    @property
    def decision_threshold(self) -> float:
        """The threshold that actually decides ``bonafide``: the calibrated accept
        threshold when an operating point is set, the dev EER threshold otherwise."""
        return self.accept_threshold if self.accept_threshold is not None else self.threshold

    def decide(self, score: float) -> str:
        """``bonafide`` / ``spoof``, or ``review`` for scores inside a calibrated band."""
        if self.accept_threshold is not None:
            if score >= self.accept_threshold:
                return "bonafide"
            if self.reject_threshold is not None and score <= self.reject_threshold:
                return "spoof"
            return "review"
        return self.predict_label(score)

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str) -> Detector:
        obj = joblib.load(path)
        if not isinstance(obj, Detector):
            raise TypeError("file does not contain a Detector")
        return obj
