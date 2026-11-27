"""Dataset loading. Two sources are supported:

* ``asvspoof2019la``: the official ASVspoof 2019 Logical Access layout and protocols.
* ``manifest``: any CSV with columns ``path,label`` (and optionally ``attack,split``),
  which is how you plug in In-the-Wild or your own recordings.

Everything is returned as one DataFrame with columns
``path, label, attack, split`` where label is 1 for bona fide and 0 for spoof.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ASVSPOOF_SPLITS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
}
PROTOCOL_DIR = "ASVspoof2019_LA_cm_protocols"
BONAFIDE = {"bonafide", "real", "genuine", "1", "1.0", "true", "yes", "human"}
SPOOF = {"spoof", "spoofed", "fake", "synthetic", "deepfake", "0", "0.0", "false", "no", "attack"}


def parse_asvspoof_protocol(protocol_path: Path, audio_dir: Path, split: str) -> pd.DataFrame:
    """Protocol lines look like: ``LA_0079 LA_T_1138215 - - bonafide`` or
    ``LA_0079 LA_T_1271820 - A01 spoof``."""
    rows = []
    with open(protocol_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, utt, _, attack, label = parts[:5]
            rows.append(
                {
                    "path": str(audio_dir / f"{utt}.flac"),
                    "label": 1 if label == "bonafide" else 0,
                    "attack": "-" if label == "bonafide" else attack,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def load_asvspoof2019_la(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    frames = []
    for split, (folder, protocol) in ASVSPOOF_SPLITS.items():
        proto = root / PROTOCOL_DIR / protocol
        if not proto.exists():
            continue
        frames.append(parse_asvspoof_protocol(proto, root / folder / "flac", split))
    if not frames:
        raise FileNotFoundError(
            f"No ASVspoof 2019 LA protocols found under {root}. Expected "
            f"{root / PROTOCOL_DIR}/<protocol files>. See docs/DATASETS.md."
        )
    return pd.concat(frames, ignore_index=True)


def load_manifest(csv_path: str | Path, base_dir: str | Path | None = None, seed: int = 0) -> pd.DataFrame:
    """Load a generic manifest. Missing ``split`` is filled with a random 70/15/15 split.

    Labels are matched after normalising case, surrounding spaces and ``-``/``_``
    separators, so ``bona fide``, ``Bona-Fide`` and ``bonafide`` all mean the same
    thing. Anything that matches neither the bona fide nor the spoof vocabulary raises
    instead of being silently counted as spoof: a typo in one label would otherwise
    flip a whole class and train a model on the opposite of what the manifest says.
    """
    df = pd.read_csv(csv_path)
    if not {"path", "label"} <= set(df.columns):
        raise ValueError("manifest needs at least the columns: path,label")
    df = df.copy()
    raw = df["label"].astype(str).str.strip().str.lower().str.replace(r"[\s\-_]+", "", regex=True)
    unknown = sorted(set(raw) - BONAFIDE - SPOOF)
    if unknown:
        raise ValueError(
            f"unrecognised label values in {csv_path}: {unknown[:5]}. "
            f"Known bona fide: {sorted(BONAFIDE)}; known spoof: {sorted(SPOOF)}"
        )
    df["label"] = raw.isin(BONAFIDE).astype(int)
    if "attack" not in df.columns:
        df["attack"] = np.where(df["label"] == 1, "-", "spoof")
    if base_dir is not None:
        df["path"] = df["path"].map(lambda p: str(Path(base_dir) / p))
    if "split" not in df.columns:
        rng = np.random.default_rng(seed)
        r = rng.random(len(df))
        df["split"] = np.where(r < 0.70, "train", np.where(r < 0.85, "dev", "eval"))
    return df[["path", "label", "attack", "split"]]


def subsample(df: pd.DataFrame, max_utts: int | None, seed: int = 0) -> pd.DataFrame:
    """Stratified (by label) random subsample, used to make quick experiments cheap.

    ``None`` means "keep everything". Zero and negative values are refused: ``0`` used
    to return the full frame, so a quick run silently trained and evaluated on the whole
    dataset while reporting the reduced size.
    """
    if max_utts is None:
        return df
    if isinstance(max_utts, bool) or not isinstance(max_utts, (int, np.integer)):
        raise TypeError(f"max_utts must be an integer or None, got {max_utts!r}")
    if max_utts < 1:
        raise ValueError(f"max_utts must be >= 1 or None, got {max_utts}")
    if len(df) <= max_utts or len(df) == 0:
        return df
    parts = []
    for _, g in df.groupby("label"):
        k = max(1, round(max_utts * len(g) / len(df)))
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
