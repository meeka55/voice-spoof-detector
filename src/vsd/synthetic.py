"""Tiny synthetic corpus in the ASVspoof 2019 LA layout.

THIS IS NOT SPEECH AND PRODUCES NO MEANINGFUL RESULTS. It exists only so that the data
loaders, feature extraction, training, evaluation and API can be smoke-tested in CI
without downloading gigabytes of audio. Real numbers must come from real datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000


def _voiced(rng: np.random.Generator, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = 120 + 40 * np.cumsum(rng.normal(0, 0.002, n))
    f0 = np.clip(f0, 80, 220)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    y = sum(np.sin(h * phase) / h for h in range(1, 20))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t + rng.uniform(0, 6))
    y = y * env + 0.02 * rng.normal(size=n)
    return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)


def _apply_attack(y: np.ndarray, attack: str, rng: np.random.Generator) -> np.ndarray:
    if attack in ("A01", "A07"):  # band-limited, like a low-rate vocoder
        b, a = signal.butter(6, 3800 / (SR / 2))
        return signal.lfilter(b, a, y).astype(np.float32)
    if attack in ("A02", "A08"):  # coarse quantisation
        return (np.round(y * 40) / 40).astype(np.float32)
    # A03 / A09: noise-excited high band
    hp = signal.lfilter(*signal.butter(4, 5000 / (SR / 2), btype="high"), rng.normal(size=len(y)))
    return (y + 0.05 * hp).astype(np.float32)


def make_synthetic_asvspoof(root: str | Path, n_per_split: int = 24, seed: int = 0) -> Path:
    """Write a miniature corpus under ``root`` and return the root."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    (root / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True, exist_ok=True)
    plan = {
        "train": ("ASVspoof2019.LA.cm.train.trn.txt", "LA_T", ["A01", "A02", "A03"]),
        "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "LA_D", ["A01", "A02", "A03"]),
        "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "LA_E", ["A07", "A08", "A09"]),
    }
    for split, (protocol, prefix, attacks) in plan.items():
        audio_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
        audio_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(n_per_split):
            utt = f"{prefix}_{i:07d}"
            base = _voiced(rng, seconds=float(rng.uniform(1.0, 2.0)))
            if i % 2 == 0:
                y, label, attack = base, "bonafide", "-"
            else:
                attack = attacks[(i // 2) % len(attacks)]
                y, label = _apply_attack(base, attack, rng), "spoof"
            sf.write(audio_dir / f"{utt}.flac", y, SR, format="FLAC")
            lines.append(f"LA_{i:04d} {utt} - {attack} {label}")
        (root / "ASVspoof2019_LA_cm_protocols" / protocol).write_text("\n".join(lines) + "\n")
    return root
