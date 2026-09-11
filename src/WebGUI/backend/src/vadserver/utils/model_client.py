"""HTTP client for the GPU model service. Keeps Django and the workers CPU-only."""
from __future__ import annotations

import os
from typing import Dict, Optional

import requests

VAD_MODEL_URL = os.environ.get("VAD_MODEL_URL", "http://vad_model:5000")
TIMEOUT_ANALYZE = float(os.environ.get("VAD_ANALYZE_TIMEOUT", "3600"))
TIMEOUT_FRAME = float(os.environ.get("VAD_FRAME_TIMEOUT", "30"))


class ModelServiceError(RuntimeError):
    pass


def _post(path: str, timeout: float, **kw) -> Dict:
    try:
        r = requests.post(f"{VAD_MODEL_URL}{path}", timeout=timeout, **kw)
    except requests.RequestException as e:
        raise ModelServiceError(f"model service unreachable: {e}") from e
    if r.status_code >= 400:
        detail = r.json().get("error", r.text) if "json" in r.headers.get(
            "content-type", "") else r.text
        raise ModelServiceError(f"model service returned {r.status_code}: {detail}")
    return r.json()


def health() -> Dict:
    try:
        r = requests.get(f"{VAD_MODEL_URL}/health", timeout=10)
        return r.json()
    except requests.RequestException as e:
        return {"status": "unreachable", "error": str(e)}


def analyze(path: str, threshold: float = 0.5, min_duration_sec: float = 1.0,
            smooth_snippets: int = 3) -> Dict:
    return _post("/analyze", TIMEOUT_ANALYZE, json={
        "path": path, "threshold": threshold,
        "min_duration_sec": min_duration_sec, "smooth_snippets": smooth_snippets})


def stream_open(fps: float = 30.0, threshold: float = 0.5, context: int = 64) -> str:
    return _post("/stream/open", 30, json={
        "fps": fps, "threshold": threshold, "context": context})["session_id"]


def stream_frame(sid: str, jpeg_bytes: bytes) -> Dict:
    return _post(f"/stream/{sid}/frame", TIMEOUT_FRAME,
                 files={"frame": ("f.jpg", jpeg_bytes, "image/jpeg")})


def stream_close(sid: str) -> Dict:
    return _post(f"/stream/{sid}/close", 60)
