"""Acoustic front-ends: MFCC and LFCC with deltas, plus utterance-level statistics.

LFCC (linear-frequency cepstral coefficients) keeps full resolution in the high
band, where vocoder artefacts tend to live, which is why it is the classic
anti-spoofing baseline front-end alongside CQCC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf
from scipy.fft import dct

MIN_SECONDS = 0.5


@dataclass(frozen=True)
class FeatureConfig:
    kind: str = "lfcc"  # "lfcc" | "mfcc"
    sr: int = 16000
    n_coeffs: int = 20
    n_filters: int = 40
    n_fft: int = 512
    win_length: int = 400  # 25 ms
    hop_length: int = 160  # 10 ms
    max_seconds: float = 8.0

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio(src: str | BinaryIO, sr: int = 16000, max_seconds: float | None = None) -> np.ndarray:
    """Load mono float32 audio from a path or file-like object, resampled to `sr`."""
    y, file_sr = sf.read(src, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    if max_seconds is not None:
        y = y[: int(max_seconds * sr)]
    return y


def _linear_filterbank(sr: int, n_fft: int, n_filters: int) -> np.ndarray:
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = np.linspace(0, sr / 2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def _lfcc(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    power = (
        np.abs(
            librosa.stft(
                y, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length
            )
        )
        ** 2
    )
    fb = _linear_filterbank(cfg.sr, cfg.n_fft, cfg.n_filters)
    log_energy = np.log(fb @ power + 1e-10)
    return dct(log_energy, type=2, axis=0, norm="ortho")[: cfg.n_coeffs]


def _mfcc(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y,
        sr=cfg.sr,
        n_mfcc=cfg.n_coeffs,
        n_mels=cfg.n_filters,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
    )


def frame_features(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Return a (T, 3 * n_coeffs) matrix: static + delta + delta-delta per frame."""
    min_len = int(MIN_SECONDS * cfg.sr)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)), mode="wrap" if len(y) > 0 else "constant")
    if cfg.kind == "lfcc":
        static = _lfcc(y, cfg)
    elif cfg.kind == "mfcc":
        static = _mfcc(y, cfg)
    else:
        raise ValueError(f"unknown feature kind: {cfg.kind!r}")
    d1 = librosa.feature.delta(static, order=1)
    d2 = librosa.feature.delta(static, order=2)
    return np.vstack([static, d1, d2]).T.astype(np.float32)


def utterance_stats(frames: np.ndarray) -> np.ndarray:
    """Mean and standard deviation over time: a fixed-length utterance vector."""
    return np.concatenate([frames.mean(axis=0), frames.std(axis=0)])
