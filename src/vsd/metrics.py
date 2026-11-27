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


def eer_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the EER.

    Every replicate resamples the bona fide and the spoof scores separately, keeping
    each class at its original size: the EER is a ratio of two error rates, so a
    replicate that drops a class would not be a rate at all. Returns
    ``(lower, upper)`` at level ``1 - alpha``. A single point estimate on a few
    thousand utterances hides how much it can move; this makes the uncertainty part
    of the report.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    for name, value in (("n_boot", n_boot), ("alpha", alpha)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {value!r}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if scores.size != labels.size:
        raise ValueError(f"scores and labels differ in length ({scores.size} vs {labels.size})")
    bona = scores[labels == 1]
    spoof = scores[labels == 0]
    if bona.size == 0 or spoof.size == 0:
        raise ValueError("EER confidence interval needs both bona fide and spoof samples")

    rng = np.random.default_rng(seed)
    repl = np.empty(int(n_boot))
    rep_labels = np.concatenate([np.ones(bona.size, dtype=int), np.zeros(spoof.size, dtype=int)])
    for i in range(int(n_boot)):
        draw = np.concatenate(
            [
                bona[rng.integers(0, bona.size, bona.size)],
                spoof[rng.integers(0, spoof.size, spoof.size)],
            ]
        )
        repl[i] = compute_eer(draw, rep_labels)[0]
    lower, upper = np.quantile(repl, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)
