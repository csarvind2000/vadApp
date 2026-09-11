"""
Object detection for the review overlay.

A YOLO detector run over sampled frames, giving person/vehicle/object boxes that
are drawn on the player. This is presentation only: the anomaly score comes from
the I3D + ATA-HS pipeline and is not influenced by these boxes. Keeping the two
separate matters, because the detector is a COCO model that has no notion of
what is anomalous -- it reports what is present, not what is wrong.

Boxes are normalised to [0,1] against frame width and height so the frontend can
scale them to whatever size the video element is rendered at.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

# Classes worth surfacing in surveillance footage. The COCO model knows 80;
# reporting all of them adds noise (a detected "tie" or "potted plant" tells a
# reviewer nothing) without adding information.
SURVEILLANCE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "backpack", "handbag", "suitcase", "knife", "baseball bat", "cell phone",
    "laptop", "dog", "fire hydrant", "bench",
}


class ObjectDetector:
    """Wraps a YOLO model; loads lazily so the service starts without weights."""

    def __init__(self, weights: str = "yolo11n.pt", device: Optional[str] = None,
                 conf: float = 0.3, restrict: bool = True):
        self.weights = weights
        self.conf = conf
        self.restrict = restrict
        self.device = device if device is not None else (0 if _cuda() else "cpu")
        self._model = None
        self.names: Dict[int, str] = {}

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            self.names = self._model.names
        return self._model

    @property
    def available(self) -> bool:
        try:
            _ = self.model
            return True
        except Exception:
            return False

    def detect_frames(self, frames: Sequence[np.ndarray], frame_indices: Sequence[int],
                      batch: int = 16) -> List[Dict]:
        """
        frames: BGR images. Returns one record per frame that has any detection:
            {"frame": int, "boxes": [{"cls","conf","x","y","w","h"}, ...]}
        with x/y/w/h normalised to [0,1].
        """
        if not frames:
            return []
        out: List[Dict] = []
        for s in range(0, len(frames), batch):
            chunk = list(frames[s:s + batch])
            res = self.model.predict(chunk, device=self.device, verbose=False,
                                     conf=self.conf)
            for r, fi, fr in zip(res, frame_indices[s:s + batch], chunk):
                h, w = fr.shape[:2]
                boxes = []
                for b in r.boxes:
                    name = r.names[int(b.cls)]
                    if self.restrict and name not in SURVEILLANCE_CLASSES:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    boxes.append({
                        "cls": name, "conf": round(float(b.conf), 3),
                        "x": round(x1 / w, 4), "y": round(y1 / h, 4),
                        "w": round((x2 - x1) / w, 4), "h": round((y2 - y1) / h, 4),
                    })
                if boxes:
                    out.append({"frame": int(fi), "boxes": boxes})
        return out

    def detect_video(self, path: str, fps_sample: float = 5.0,
                     intervals: Optional[Sequence[tuple]] = None,
                     max_frames: int = 1200) -> Dict:
        """
        Sample a video and detect objects.

        ``intervals`` restricts sampling to (start_sec, end_sec) spans, so a long
        video is only scanned where the detector flagged something. Without it the
        whole video is sampled.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps / max(fps_sample, 0.1))))

        wanted = range(0, n, step)
        if intervals:
            keep = set()
            for a, b in intervals:
                lo, hi = int(a * fps), int(b * fps)
                keep.update(i for i in wanted if lo <= i <= hi)
            wanted = sorted(keep)
        wanted = list(wanted)[:max_frames]

        frames, idxs = [], []
        try:
            for i in wanted:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, fr = cap.read()
                if ok:
                    frames.append(fr); idxs.append(i)
        finally:
            cap.release()

        dets = self.detect_frames(frames, idxs)
        counts: Dict[str, int] = {}
        for d in dets:
            for b in d["boxes"]:
                counts[b["cls"]] = counts.get(b["cls"], 0) + 1
        return {
            "fps": fps, "sample_step": step, "sampled_frames": len(frames),
            "detections": dets,
            # Per-class totals over sampled frames; an object present across many
            # frames counts many times, so this indicates prevalence, not identity.
            "class_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        }


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
