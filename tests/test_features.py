import numpy as np
import pytest

from vsd.features import FeatureConfig, frame_features, utterance_stats


@pytest.mark.parametrize("kind", ["lfcc", "mfcc"])
def test_frame_features_shape_and_finite(kind):
    cfg = FeatureConfig(kind=kind, n_coeffs=20)
    y = np.random.default_rng(0).normal(size=16000 * 2).astype(np.float32)
    f = frame_features(y, cfg)
    assert f.shape[1] == 60
    assert f.shape[0] > 150
    assert np.isfinite(f).all()


def test_short_audio_is_padded_not_crashing():
    cfg = FeatureConfig()
    f = frame_features(np.zeros(100, dtype=np.float32), cfg)
    assert f.shape[1] == 60 and np.isfinite(f).all()


def test_utterance_stats_length():
    f = np.random.default_rng(0).normal(size=(100, 60)).astype(np.float32)
    assert utterance_stats(f).shape == (120,)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        frame_features(np.zeros(16000, dtype=np.float32), FeatureConfig(kind="nope"))
