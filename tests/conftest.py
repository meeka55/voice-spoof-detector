import pytest

from vsd.synthetic import make_synthetic_asvspoof


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    return make_synthetic_asvspoof(tmp_path_factory.mktemp("asvspoof_synth"), n_per_split=24)
