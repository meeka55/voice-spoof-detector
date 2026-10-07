import numpy as np
import pytest

from vsd.metrics import compute_eer, per_attack_eer


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
