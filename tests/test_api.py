import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from vsd import api, train


@pytest.fixture(scope="module")
def client(synthetic_root, tmp_path_factory):
    model = tmp_path_factory.mktemp("m") / "m.joblib"
    train.main(["--data-root", str(synthetic_root), "--backend", "logreg",
                "--n-jobs", "1", "--out", str(model)])
    import os

    os.environ["VSD_MODEL_PATH"] = str(model)
    with TestClient(api.app) as c:
        yield c


def _wav_bytes(seconds=1.5):
    y = (0.3 * np.sin(2 * np.pi * 150 * np.arange(int(16000 * seconds)) / 16000)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, y, 16000, format="WAV")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_detect_ok(client):
    r = client.post("/v1/detect", files={"file": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"bonafide", "spoof"}
    assert body["duration_s"] == pytest.approx(1.5, abs=0.01)


def test_detect_rejects_garbage(client):
    r = client.post("/v1/detect", files={"file": ("a.wav", b"not audio", "audio/wav")})
    assert r.status_code == 422


def test_detect_rejects_too_short(client):
    r = client.post("/v1/detect", files={"file": ("a.wav", _wav_bytes(0.1), "audio/wav")})
    assert r.status_code == 422
