"""FastAPI service.

    VSD_MODEL_PATH=models/lfcc_gmm.joblib uvicorn vsd.api:app --port 8000
"""

from __future__ import annotations

import io
import math
import os
from contextlib import asynccontextmanager
from typing import Annotated

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import __version__
from .detector import Detector
from .features import load_audio

MAX_UPLOAD_BYTES = int(os.getenv("VSD_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MIN_SECONDS = 0.5

state: dict = {}


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, naming the variable when it is unusable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("VSD_MODEL_PATH", "models/model.joblib")
    if os.path.exists(path):
        state["detector"] = Detector.load(path)
        if os.getenv("VSD_THRESHOLD") is not None:
            det = state["detector"]
            det.threshold = _env_float("VSD_THRESHOLD", det.threshold)
            # An explicit threshold replaces a calibrated band, otherwise the override
            # would be accepted and then ignored by decide().
            det.accept_threshold = None
            det.reject_threshold = None
    yield
    state.clear()


app = FastAPI(
    title="Voice Spoof Detector",
    version=__version__,
    description="Scores an audio clip for the likelihood that it is synthetic or replayed speech.",
    lifespan=lifespan,
)


class Detection(BaseModel):
    label: str
    score: float
    threshold: float
    duration_s: float
    model: dict
    accept_threshold: float | None = None
    reject_threshold: float | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "detector" in state, "version": __version__}


@app.post("/v1/detect", response_model=Detection)
async def detect(file: Annotated[UploadFile, File()]) -> Detection:
    det: Detector | None = state.get("detector")
    if det is None:
        raise HTTPException(503, "No model loaded. Set VSD_MODEL_PATH and restart.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES} bytes")
    try:
        y = load_audio(io.BytesIO(data), det.feature_cfg.sr, det.feature_cfg.max_seconds)
    except (RuntimeError, sf.LibsndfileError, ValueError) as exc:
        raise HTTPException(422, f"Could not decode audio: {exc}") from exc
    duration = len(y) / det.feature_cfg.sr
    if duration < MIN_SECONDS:
        raise HTTPException(422, f"Audio shorter than {MIN_SECONDS}s")
    score = det.score_audio(y)
    return Detection(
        label=det.decide(score),
        score=score,
        threshold=det.decision_threshold,
        duration_s=round(duration, 3),
        model=det.meta,
        accept_threshold=det.accept_threshold,
        reject_threshold=det.reject_threshold,
    )
