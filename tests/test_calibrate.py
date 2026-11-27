import numpy as np
import pytest

from vsd.calibrate import calibrate


def _scores(n_bona: int = 1500, n_spoof: int = 6000, seed: int = 1):
    rng = np.random.default_rng(seed)
    bona = rng.normal(1.0, 1.0, n_bona)
    spoof = rng.normal(-0.5, 1.0, n_spoof)
    return (
        np.concatenate([bona, spoof]),
        np.concatenate([np.ones(n_bona, dtype=int), np.zeros(n_spoof, dtype=int)]),
    )


def test_reported_rates_are_the_measured_ones():
    scores, labels = _scores()
    op = calibrate(scores, labels, target_far=0.05, target_frr=0.05)
    assert op.far <= 0.05 and op.frr <= 0.05
    assert op.far == pytest.approx(float(np.mean(scores[labels == 0] >= op.accept)))
    assert op.frr == pytest.approx(float(np.mean(scores[labels == 1] <= op.reject)))
    assert op.n_bona == 1500 and op.n_spoof == 6000
    assert 0.0 < op.review_rate < 1.0


def test_tighter_budget_costs_acceptance_not_just_review():
    scores, labels = _scores()
    loose = calibrate(scores, labels, target_far=0.10, target_frr=0.10)
    tight = calibrate(scores, labels, target_far=0.01, target_frr=0.01)
    assert tight.accept > loose.accept
    assert tight.reject < loose.reject
    assert tight.far <= loose.far
    assert tight.review_rate > loose.review_rate


def test_zero_budget_accepts_no_spoof():
    scores, labels = _scores()
    op = calibrate(scores, labels, target_far=0.0, target_frr=0.0)
    assert op.far == 0.0
    assert op.frr == 0.0
    assert op.accept > scores[labels == 0].max()


def test_permissive_budget_collapses_the_band():
    scores, labels = _scores()
    op = calibrate(scores, labels, target_far=0.5, target_frr=0.5)
    assert op.band_empty is True
    assert op.reject == op.accept
    assert op.review_rate == 0.0


def test_decide_maps_to_the_three_labels():
    scores, labels = _scores()
    op = calibrate(scores, labels, target_far=0.05, target_frr=0.05)
    assert op.decide(op.accept) == "bonafide"
    assert op.decide(op.reject) == "spoof"
    assert op.decide((op.accept + op.reject) / 2.0) == "review"
    decided = op.decide_many(np.array([op.accept, op.reject, (op.accept + op.reject) / 2.0]))
    assert list(decided) == ["bonafide", "spoof", "review"]


def test_as_dict_round_trips_to_json():
    import json

    scores, labels = _scores(n_bona=200, n_spoof=200)
    op = calibrate(scores, labels, target_far=0.1, target_frr=0.1)
    payload = json.loads(json.dumps(op.as_dict()))
    assert payload["accept"] == op.accept and payload["target_far"] == 0.1


def test_identical_scores_land_entirely_in_review():
    scores = np.zeros(100)
    labels = np.array([1] * 50 + [0] * 50)
    op = calibrate(scores, labels, target_far=0.05, target_frr=0.05)
    # Zero separation: no score clears the accept threshold, and the "reject" threshold
    # sits below every score, so every utterance is sent for review instead of being
    # silently counted as a correct decision.
    assert op.decide(0.0) == "review"
    assert op.far == 0.0 and op.frr == 0.0
    assert op.review_rate == 1.0


def test_needs_both_classes():
    with pytest.raises(ValueError, match="no spoof"):
        calibrate(np.array([1.0, 2.0]), np.array([1, 1]), 0.1, 0.1)


def test_rejects_bad_arguments():
    scores, labels = _scores(n_bona=50, n_spoof=50)
    with pytest.raises(ValueError, match="target_far"):
        calibrate(scores, labels, target_far=-0.1, target_frr=0.1)
    with pytest.raises(ValueError, match="target_frr"):
        calibrate(scores, labels, target_far=0.1, target_frr=1.5)
    with pytest.raises(ValueError, match="differ in length"):
        calibrate(scores[:-1], labels, 0.1, 0.1)
    with pytest.raises(ValueError, match="empty"):
        calibrate(np.array([]), np.array([]), 0.1, 0.1)
    broken = scores.copy()
    broken[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calibrate(broken, labels, 0.1, 0.1)
