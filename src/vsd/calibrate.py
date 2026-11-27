"""Operating-point calibration: pick decision thresholds from labelled scores.

Convention (the same as everywhere else in this package): a **higher score means more
likely bona fide**, label 1 = bona fide, label 0 = spoof.

A single threshold cannot say "I do not know". ``calibrate`` therefore fits two
thresholds and leaves the scores in between for human review:

* ``score >= accept`` -> ``bonafide``
* ``score <= reject`` -> ``spoof``
* otherwise -> ``review``

Both thresholds are chosen from a *held-out* split with a false-accept and a
false-reject budget, so the numbers you get back are the rates you actually bought
on that split, not a hope. The ASVspoof protocol puts ``dev`` in that role: attacks
there are the ones seen in training, so an operating point picked on ``dev`` is
optimistic for unseen attacks. Reporting it on ``eval`` is what shows the gap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

LABELS = ("bonafide", "spoof", "review")


def _validate_targets(target_far: float, target_frr: float) -> tuple[float, float]:
    out = []
    for name, value in (("target_far", target_far), ("target_frr", target_frr)):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number, got {value!r}") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
        out.append(value)
    return out[0], out[1]


def _far_at(thresholds: np.ndarray, spoof_sorted: np.ndarray) -> np.ndarray:
    """Share of spoof utterances accepted at each threshold (score >= threshold)."""
    n_ge = len(spoof_sorted) - np.searchsorted(spoof_sorted, thresholds, side="left")
    return n_ge / len(spoof_sorted)


def _frr_at(thresholds: np.ndarray, bona_sorted: np.ndarray) -> np.ndarray:
    """Share of bona fide utterances rejected at each threshold (score <= threshold)."""
    n_le = np.searchsorted(bona_sorted, thresholds, side="right")
    return n_le / len(bona_sorted)


@dataclass(frozen=True)
class OperatingPoint:
    """Two thresholds plus the error rates they achieve on the calibration split."""

    accept: float
    reject: float
    target_far: float
    target_frr: float
    far: float
    frr: float
    review_rate: float
    n_bona: int
    n_spoof: int
    band_empty: bool

    def decide(self, score: float) -> str:
        """Map one score to ``bonafide`` / ``spoof`` / ``review``."""
        if score >= self.accept:
            return "bonafide"
        if score <= self.reject:
            return "spoof"
        return "review"

    def decide_many(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        out = np.full(scores.shape, "review", dtype=object)
        out[scores >= self.accept] = "bonafide"
        out[scores <= self.reject] = "spoof"
        return out.astype(str)

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    target_far: float = 0.05,
    target_frr: float = 0.05,
) -> OperatingPoint:
    """Pick thresholds such that at most ``target_far`` of spoofs are accepted and at
    most ``target_frr`` of bona fide utterances are rejected on this split.

    ``accept`` is the *lowest* score that satisfies the false-accept budget and
    ``reject`` the *highest* score that satisfies the false-reject budget. If the two
    budgets cannot be met with an ordered pair of thresholds (a model with little
    separation), ``accept`` is raised to ``reject``, ``band_empty`` is set, and
    ``review_rate`` is 0: the caller is told the band collapsed instead of silently
    getting a review band that can never trigger.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    target_far, target_frr = _validate_targets(target_far, target_frr)
    if scores.size != labels.size:
        raise ValueError(f"scores and labels differ in length ({scores.size} vs {labels.size})")
    if scores.size == 0:
        raise ValueError("cannot calibrate on an empty score set")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite; NaN or inf found")
    bona_sorted = np.sort(scores[labels == 1])
    spoof_sorted = np.sort(scores[labels == 0])
    if bona_sorted.size == 0 or spoof_sorted.size == 0:
        missing = "bona fide" if bona_sorted.size == 0 else "spoof"
        raise ValueError(f"calibration needs both classes; no {missing} samples")

    candidates = np.unique(scores)
    far_curve = _far_at(candidates, spoof_sorted)
    frr_curve = _frr_at(candidates, bona_sorted)

    feasible_far = np.nonzero(far_curve <= target_far)[0]
    if target_far >= 1.0:
        accept = float(scores.min())
    elif feasible_far.size:
        accept = float(candidates[feasible_far[0]])
    else:  # budget below 1/n_spoof: the only way to honour it is to accept nothing
        accept = float(np.nextafter(spoof_sorted[-1], np.inf))

    feasible_frr = np.nonzero(frr_curve <= target_frr)[0]
    if target_frr >= 1.0:
        reject = float(scores.max())
    elif feasible_frr.size:
        reject = float(candidates[feasible_frr[-1]])
    else:  # budget below 1/n_bona: reject nothing
        reject = float(np.nextafter(bona_sorted[0], -np.inf))

    band_empty = reject >= accept
    if band_empty:
        # The two budgets cannot be met with an ordered pair of thresholds (the model
        # separates little at these targets). Raise the accept side to the reject side:
        # that keeps both budgets and can only let *fewer* spoofs through, whereas
        # lowering reject would silently widen acceptance while reporting no review.
        accept = reject

    far = float(_far_at(np.array([accept]), spoof_sorted)[0])
    frr = float(_frr_at(np.array([reject]), bona_sorted)[0])
    review_rate = float(np.mean((scores > reject) & (scores < accept)))
    return OperatingPoint(
        accept=accept,
        reject=reject,
        target_far=target_far,
        target_frr=target_frr,
        far=far,
        frr=frr,
        review_rate=review_rate,
        n_bona=int(bona_sorted.size),
        n_spoof=int(spoof_sorted.size),
        band_empty=bool(band_empty),
    )
