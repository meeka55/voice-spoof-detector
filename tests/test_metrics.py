import numpy as np
import pytest

from vsd.metrics import compute_eer, eer_ci, per_attack_eer


def test_eer_perfect_separation():
    scores = np.array([2.0, 1.5, 1.0, -1.0, -1.5, -2.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    eer, _ = compute_eer(scores, labels)
    assert eer == pytest.approx(0.0)


def test_eer_random_is_about_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000)
    labels = rng.integers(0, 2, size=4000)
    eer, _ = compute_eer(scores, labels)
    assert 0.45 < eer < 0.55


def test_eer_needs_both_classes():
    with pytest.raises(ValueError):
        compute_eer(np.array([1.0, 2.0]), np.array([1, 1]))


def test_per_attack_eer_flags_the_hard_attack():
    scores = np.array([3, 2.5, 2, -3, -2.5, 2.2, 1.8, 2.4])
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    attacks = np.array(["-", "-", "-", "A01", "A01", "A02", "A02", "A02"])
    res = per_attack_eer(scores, labels, attacks)
    assert res["A01"] < res["A02"]


def _overlapping(n=2000, seed=3):
    rng = np.random.default_rng(seed)
    bona = rng.normal(1.0, 1.0, n)
    spoof = rng.normal(0.0, 1.0, n)
    return np.concatenate([bona, spoof]), np.concatenate([np.ones(n), np.zeros(n)])


def test_eer_ci_brackets_the_point_estimate():
    scores, labels = _overlapping()
    eer, _ = compute_eer(scores, labels)
    lo, hi = eer_ci(scores, labels, n_boot=100, seed=0)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= eer <= hi
    assert hi - lo > 0.0


def test_eer_ci_is_reproducible_and_collapses_on_perfect_separation():
    scores = np.array([2.0, 1.5, 1.0, -1.0, -1.5, -2.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert eer_ci(scores, labels, n_boot=50, seed=0) == (0.0, 0.0)
    noisy_scores, noisy_labels = _overlapping(n=300)
    first = eer_ci(noisy_scores, noisy_labels, n_boot=50, seed=7)
    second = eer_ci(noisy_scores, noisy_labels, n_boot=50, seed=7)
    assert first == second


def test_eer_ci_validates_its_arguments():
    scores, labels = _overlapping(n=100)
    with pytest.raises(ValueError, match="n_boot"):
        eer_ci(scores, labels, n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        eer_ci(scores, labels, alpha=0.0)
    with pytest.raises(ValueError, match="differ in length"):
        eer_ci(scores[:-1], labels)
    with pytest.raises(ValueError, match="both"):
        eer_ci(np.array([1.0, 2.0]), np.array([1, 1]))
