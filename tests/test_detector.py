"""The interface contract of Detector: what it refuses and how it decides."""

import numpy as np
import pytest

from vsd.detector import Detector
from vsd.features import FeatureConfig


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="backend"):
        Detector(backend="knn")


def test_component_count_must_be_a_positive_integer():
    with pytest.raises(ValueError, match="n_components"):
        Detector(n_components=0)
    with pytest.raises(TypeError, match="n_components"):
        Detector(n_components="4")


def test_decision_threshold_prefers_the_calibrated_accept_threshold():
    det = Detector(backend="logreg")
    det.threshold = 0.5
    assert det.decision_threshold == pytest.approx(0.5)
    det.accept_threshold = 1.5
    det.reject_threshold = -1.5
    assert det.decision_threshold == pytest.approx(1.5)


def test_decide_without_a_band_matches_predict_label():
    det = Detector(backend="logreg")
    det.threshold = 0.0
    assert det.decide(0.5) == det.predict_label(0.5) == "bonafide"
    assert det.decide(-0.5) == det.predict_label(-0.5) == "spoof"


def test_decide_reports_review_inside_the_band():
    det = Detector(backend="logreg")
    det.threshold = 0.0
    det.accept_threshold = 1.0
    det.reject_threshold = -1.0
    assert det.decide(2.0) == "bonafide"
    assert det.decide(0.0) == "review"
    assert det.decide(-2.0) == "spoof"


def test_fit_refuses_a_label_list_of_the_wrong_length():
    det = Detector(backend="logreg")
    with pytest.raises(ValueError, match="differ in length"):
        det.fit(["a.wav", "b.wav"], [1])


def test_fit_refuses_an_empty_input():
    det = Detector(backend="logreg")
    with pytest.raises(ValueError, match="empty"):
        det.fit([], [])


def test_fit_refuses_a_single_class():
    det = Detector(backend="logreg")
    with pytest.raises(ValueError, match="no spoof"):
        det.fit(["a.wav"], [1])


def test_fit_refuses_a_zero_frame_budget():
    det = Detector(backend="logreg")
    with pytest.raises(ValueError, match="frames_per_utt"):
        det.fit(["a.wav"], [1], frames_per_utt=0)


def test_score_many_returns_float_scores(synthetic_root):
    from vsd.data import load_asvspoof2019_la

    df = load_asvspoof2019_la(synthetic_root)
    train = df[df.split == "train"]
    det = Detector(FeatureConfig(kind="lfcc"), backend="logreg").fit(
        train.path.tolist(), train.label.tolist(), n_jobs=1
    )
    scores = det.score_many(train.path.tolist()[:4], n_jobs=1)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()


def test_models_saved_before_the_band_existed_still_load(tmp_path):
    """A pickle written by an older version has no accept/reject attribute at all."""
    import joblib

    det = Detector(backend="logreg")
    det.threshold = 0.0
    det.__dict__.pop("accept_threshold", None)
    det.__dict__.pop("reject_threshold", None)
    path = tmp_path / "old.joblib"
    joblib.dump(det, path)
    loaded = Detector.load(str(path))
    assert loaded.accept_threshold is None and loaded.reject_threshold is None
    assert loaded.decide(0.5) == "bonafide"
    assert loaded.decision_threshold == pytest.approx(0.0)
