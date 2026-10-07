"""Evaluation metrics. Convention everywhere: label 1 = bona fide, 0 = spoof,
and a higher score means "more likely bona fide"."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the score threshold at which it occurs."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        raise ValueError("EER needs both bona fide and spoof samples")
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False-accept rate (spoof accepted) and false-reject rate (bona fide rejected)."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return fpr, 1.0 - tpr


def per_attack_eer(
    scores: np.ndarray, labels: np.ndarray, attacks: np.ndarray
) -> dict[str, float]:
    """EER of bona fide vs each individual attack. Shows which attacks generalise badly."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    attacks = np.asarray(attacks)
    bona = labels == 1
    out: dict[str, float] = {}
    for attack in sorted(set(attacks[~bona])):
        if attack in ("-", "", "nan"):
            continue
        mask = bona | (attacks == attack)
        out[str(attack)] = compute_eer(scores[mask], labels[mask])[0]
    return out


def summarize(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    eer, thr = compute_eer(scores, labels)
    return {
        "eer": eer,
        "eer_threshold": thr,
        "auc": float(roc_auc_score(labels, scores)),
        "n_bonafide": int((np.asarray(labels) == 1).sum()),
        "n_spoof": int((np.asarray(labels) == 0).sum()),
    }
