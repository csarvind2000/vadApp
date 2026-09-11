"""
Flask model service: the only container that touches the GPU.

The Django backend and its RQ workers stay CPU-only and call this over HTTP, so
the model is loaded once and shared rather than re-instantiated per job. This
bounds GPU memory to a single copy of the network however many workers run, and
keeps the web tier deployable on hosts without a GPU.

Endpoints
---------
GET  /health                 liveness + which checkpoint is loaded
POST /analyze                {"path": "/media/videos/x.mp4", ...} -> scores + incidents
POST /stream/open            {"fps":..., "threshold":...} -> {"session_id":...}
POST /stream/<sid>/frame     JPEG bytes (or base64) -> per-snippet score when ready
POST /stream/<sid>/close     -> full session summary, then frees the session
"""
from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from typing import Dict

import cv2
import numpy as np
from flask import Flask, jsonify, request

from utils.engine import VADEngine

app = Flask(__name__)

CHECKPOINT = os.environ.get("VAD_CHECKPOINT", "/models/best.pt")
BACKBONE = os.environ.get("VAD_BACKBONE", "i3d_r50")
DEFAULT_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
STREAM_TTL_SEC = float(os.environ.get("VAD_STREAM_TTL", "900"))

_engine: VADEngine | None = None
_engine_lock = threading.Lock()
_sessions: Dict[str, dict] = {}
_sessions_lock = threading.Lock()


def engine() -> VADEngine:
    """Load lazily so the container starts even if the checkpoint appears later."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = VADEngine(CHECKPOINT, backbone=BACKBONE)
    return _engine


def _reap_sessions() -> None:
    """Drop abandoned stream sessions so a dropped browser cannot leak GPU memory."""
    now = time.time()
    with _sessions_lock:
        for sid in [s for s, v in _sessions.items() if now - v["seen"] > STREAM_TTL_SEC]:
            _sessions.pop(sid, None)


@app.get("/health")
def health():
    try:
        e = engine()
        return jsonify({"status": "ok", "model": e.model_kind, "backbone": e.backbone_name,
                        "feature_dim": e.feat_dim, "device": str(e.device),
                        "checkpoint": CHECKPOINT, "trained_epoch": e.trained_epoch,
                        "val_auc": e.val_auc, "active_streams": len(_sessions),
                        "category_head": e.category_head is not None,
                        "category_top1_accuracy": e.category_top1})
    except Exception as exc:
        # Report unhealthy rather than crashing: the stack should come up and
        # show a clear reason even when no checkpoint has been trained yet.
        return jsonify({"status": "model_unavailable", "error": str(exc),
                        "checkpoint": CHECKPOINT}), 503


@app.post("/analyze")
def analyze():
    body = request.get_json(force=True, silent=True) or {}
    path = body.get("path")
    if not path:
        return jsonify({"error": "missing 'path'"}), 400
    if not os.path.isfile(path):
        return jsonify({"error": f"file not found: {path}"}), 404
    try:
        result = engine().score_video(
            path,
            batch_size=int(body.get("batch_size", 8)),
            threshold=float(body.get("threshold", DEFAULT_THRESHOLD)),
            min_duration_sec=float(body.get("min_duration_sec", 1.0)),
            smooth_snippets=int(body.get("smooth_snippets", 3)),
        )
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify(result)


@app.post("/stream/open")
def stream_open():
    _reap_sessions()
    body = request.get_json(force=True, silent=True) or {}
    sid = uuid.uuid4().hex
    sess = engine().stream_session(
        context=int(body.get("context", 64)),
        fps=float(body.get("fps", 30.0)),
        threshold=float(body.get("threshold", DEFAULT_THRESHOLD)),
    )
    with _sessions_lock:
        _sessions[sid] = {"session": sess, "seen": time.time()}
    return jsonify({"session_id": sid})


def _decode_frame(req) -> np.ndarray | None:
    """Accept either a raw file upload or base64 JSON, whichever the client sends."""
    if req.files.get("frame"):
        raw = np.frombuffer(req.files["frame"].read(), np.uint8)
    else:
        body = req.get_json(force=True, silent=True) or {}
        b64 = body.get("frame_b64")
        if not b64:
            return None
        if "," in b64[:64]:                      # strip a data: URL prefix
            b64 = b64.split(",", 1)[1]
        raw = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


@app.post("/stream/<sid>/frame")
def stream_frame(sid: str):
    with _sessions_lock:
        entry = _sessions.get(sid)
        if entry:
            entry["seen"] = time.time()
    if entry is None:
        return jsonify({"error": "unknown or expired session"}), 404

    frame = _decode_frame(request)
    if frame is None:
        return jsonify({"error": "could not decode frame"}), 400
    try:
        res = entry["session"].push_frame(frame)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    # None means the snippet buffer is not full yet -- normal, not an error.
    return jsonify(res if res is not None else {"buffering": True})


@app.post("/stream/<sid>/close")
def stream_close(sid: str):
    with _sessions_lock:
        entry = _sessions.pop(sid, None)
    if entry is None:
        return jsonify({"error": "unknown or expired session"}), 404
    return jsonify(entry["session"].finish())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
