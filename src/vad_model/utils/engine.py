"""
VAD inference engine: I3D snippet features -> ATA-HS anomaly scores.

Two modes share one loaded model:

``score_video``   offline analysis of an uploaded file. Decodes the whole video,
                  extracts snippet features, scores every snippet, expands to
                  frame level.
``StreamSession``  live camera. Frames arrive in batches; the session keeps a
                  rolling buffer of snippet features so the temporal model still
                  sees context rather than scoring each snippet in isolation.

The frame/snippet conventions here MUST match those used at training time
(16-frame snippets, short side 256, 224 centre crop, Kinetics normalisation) or
the features the model sees at inference differ from the ones it was trained on.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

# The research package is mounted into the image so the app and the experiments
# share one definition of the model -- no re-implementation to drift out of sync.
_RESEARCH_ROOT = os.environ.get("RESEARCH_ROOT", "/research")
if _RESEARCH_ROOT not in sys.path:
    sys.path.insert(0, _RESEARCH_ROOT)

from src.ucfcrime.ata_hs import ATAHS                     # noqa: E402
from src.ucfcrime.classify import CategoryHead            # noqa: E402
from src.ucfcrime.extract_features import (CROP, MEAN, STD, _crops,   # noqa: E402
                                           _resize_short, build_backbone)
from src.ucfcrime.models import SultaniMIL                # noqa: E402

from .detect import ObjectDetector                        # noqa: E402

SNIPPET_LEN = 16


@dataclass
class Incident:
    """A contiguous run of frames scored above threshold."""
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    peak_score: float
    mean_score: float
    peak_frame: int

    def to_dict(self) -> Dict:
        # Cast explicitly: these originate from numpy indexing, and numpy ints are
        # not JSON-serializable -- which fails only at the HTTP boundary, long
        # after the values themselves look correct.
        return {
            "start_frame": int(self.start_frame), "end_frame": int(self.end_frame),
            "start_sec": round(self.start_sec, 2), "end_sec": round(self.end_sec, 2),
            "duration_sec": round(self.end_sec - self.start_sec, 2),
            "peak_score": round(float(self.peak_score), 4),
            "mean_score": round(float(self.mean_score), 4),
            "peak_frame": int(self.peak_frame),
        }


def find_incidents(
    frame_scores: np.ndarray,
    fps: float,
    threshold: float = 0.5,
    min_duration_sec: float = 1.0,
    merge_gap_sec: float = 1.0,
) -> List[Incident]:
    """
    Turn a frame-score curve into discrete incidents.

    Runs above ``threshold`` are merged when separated by less than
    ``merge_gap_sec`` (a brief dip mid-event should not split one incident in
    two), then runs shorter than ``min_duration_sec`` are dropped as noise.
    """
    s = np.asarray(frame_scores, dtype=np.float64).ravel()
    if s.size == 0:
        return []
    above = s > threshold
    if not above.any():
        return []

    edges = np.diff(above.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(s.size)

    merge_gap = max(1, int(round(merge_gap_sec * fps)))
    merged: List[List[int]] = []
    for a, b in zip(starts, ends):
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    min_len = max(1, int(round(min_duration_sec * fps)))
    out: List[Incident] = []
    for a, b in merged:
        if b - a < min_len:
            continue
        seg = s[a:b]
        peak = int(a + int(np.argmax(seg)))
        out.append(Incident(int(a), int(b - 1), float(a / fps), float((b - 1) / fps),
                            float(seg.max()), float(seg.mean()), int(peak)))
    return out


class VADEngine:
    """Loads the backbone and the trained detector once, serves both modes."""

    def __init__(
        self,
        checkpoint: str,
        backbone: str = "i3d_r50",
        device: Optional[str] = None,
        snippet_len: int = SNIPPET_LEN,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.snippet_len = snippet_len
        self.backbone_name = backbone
        self.backbone, self.feat_dim = build_backbone(backbone, self.device)

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {})
        self.model_kind = args.get("model", "atahs")
        in_dim = ckpt.get("in_dim", self.feat_dim)
        if in_dim != self.feat_dim:
            raise ValueError(
                f"checkpoint expects {in_dim}-d features but backbone {backbone!r} "
                f"produces {self.feat_dim}-d. Use the backbone the model was trained on."
            )

        if self.model_kind == "sultani":
            self.detector = SultaniMIL(in_dim, dropout=args.get("dropout", 0.6))
        else:
            self.detector = ATAHS(
                in_dim,
                d_model=args.get("d_model", 512),
                dropout=args.get("dropout", 0.3),
                attn=args.get("attn", "entmax"),
                use_multiscale=not args.get("no_multiscale", False),
                adaptive_budget=not args.get("no_adaptive_budget", False),
                heads=tuple(args.get("heads", "mil,mag,tmp").split(",")),
                adaptive_fusion=not args.get("no_adaptive_fusion", False),
            )
        self.detector.load_state_dict(ckpt["model"])
        self.detector.to(self.device).eval()
        self.l2_normalise = not args.get("no_l2", False)
        self.trained_epoch = ckpt.get("epoch")
        self.val_auc = ckpt.get("val_auc")

        # Optional category head. Absent by default: the detector works without
        # it, and a missing classifier must not prevent the service starting.
        self.category_head = None
        self.category_classes: List[str] = []
        self.category_top1 = None
        # Named distinctly from self.detector, which is the anomaly model.
        self.object_detector = ObjectDetector(
            weights=os.environ.get("VAD_YOLO_WEIGHTS", "yolo11n.pt"),
            conf=float(os.environ.get("VAD_YOLO_CONF", "0.3")))

        cat_path = os.environ.get("VAD_CATEGORY_CHECKPOINT", "/models/category.pt")
        if os.path.isfile(cat_path):
            cc = torch.load(cat_path, map_location="cpu", weights_only=False)
            if cc.get("in_dim") == in_dim:
                self.category_head = CategoryHead(in_dim)
                self.category_head.load_state_dict(cc["model"])
                self.category_head.to(self.device).eval()
                self.category_classes = cc.get("classes", [])
                self.category_top1 = cc.get("top1")

    # -- feature extraction ------------------------------------------------
    @torch.no_grad()
    def _features_from_clips(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        """clips: list of [T,H,W,3] float32 in [0,1] -> [n, D] float32."""
        if not clips:
            return np.zeros((0, self.feat_dim), dtype=np.float32)
        arr = np.stack([_crops(c, 1) for c in clips], axis=0)          # [B,1,T,H,W,3]
        b = arr.shape[0]
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
        t = (t - torch.tensor(MEAN, device=self.device)) / torch.tensor(STD, device=self.device)
        t = t.permute(0, 1, 5, 2, 3, 4).reshape(b, 3, self.snippet_len, CROP, CROP)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            f = self.backbone(t)
        return f.float().cpu().numpy()

    def _prep(self, feats: np.ndarray) -> torch.Tensor:
        f = np.asarray(feats, dtype=np.float32)
        if self.l2_normalise:
            f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
        return torch.from_numpy(f).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def classify(self, feats: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Rank anomaly categories for a whole video.

        Returned as a ranked list with probabilities rather than a single label:
        top-1 accuracy is 0.38 over 13 classes, so a bare label would be wrong
        more often than right. The caller is expected to present the ranking and
        the confidence, not just the argmax.
        """
        if self.category_head is None or feats.shape[0] == 0:
            return []
        logits = self.category_head(self._prep(feats))
        p = torch.softmax(logits, dim=1)[0]
        k = min(top_k, p.numel())
        vals, idx = p.topk(k)
        return [{"category": self.category_classes[int(j)], "probability": round(float(v), 4)}
                for v, j in zip(vals, idx)]

    @torch.no_grad()
    def _score_features(self, feats: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """[n,D] -> (snippet scores [n], attention weights [n] or None)."""
        if feats.shape[0] == 0:
            return np.zeros(0, dtype=np.float32), None
        x = self._prep(feats)
        if self.model_kind == "sultani":
            return self.detector(x).squeeze(0).cpu().numpy(), None
        out = self.detector(x)
        return (out["score"].squeeze(0).cpu().numpy(),
                out["attn"].squeeze(0).cpu().numpy())

    # -- offline video -----------------------------------------------------
    @torch.no_grad()
    def score_video(
        self,
        path: str,
        batch_size: int = 8,
        threshold: float = 0.5,
        min_duration_sec: float = 1.0,
        smooth_snippets: int = 3,
        detect_objects: bool = True,
        object_fps: float = 5.0,
        progress=None,
    ) -> Dict:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        feats: List[np.ndarray] = []
        buf: List[np.ndarray] = []
        pending: List[np.ndarray] = []
        n_frames = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                n_frames += 1
                rgb = _resize_short(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buf.append(rgb.astype(np.float32) / 255.0)
                if len(buf) == self.snippet_len:
                    pending.append(np.stack(buf, 0)); buf = []
                    if len(pending) >= batch_size:
                        feats.append(self._features_from_clips(pending)); pending = []
                        if progress and n_total:
                            progress(min(0.95, n_frames / max(n_total, 1)))
            if pending:
                feats.append(self._features_from_clips(pending))
        finally:
            cap.release()

        if not feats:
            raise RuntimeError("video too short: no complete 16-frame snippet")
        F = np.concatenate(feats, 0)
        snippet_scores, attn = self._score_features(F)

        if smooth_snippets > 1 and snippet_scores.size >= smooth_snippets:
            k = np.ones(smooth_snippets) / smooth_snippets
            pad = smooth_snippets // 2
            snippet_scores = np.convolve(
                np.pad(snippet_scores, (pad, pad), mode="edge"), k, mode="valid"
            )[:F.shape[0]]

        frame_scores = np.repeat(snippet_scores, self.snippet_len)
        if frame_scores.size < n_frames:
            frame_scores = np.concatenate(
                [frame_scores, np.full(n_frames - frame_scores.size, frame_scores[-1])])
        frame_scores = frame_scores[:n_frames]

        incidents = find_incidents(frame_scores, fps, threshold, min_duration_sec)
        categories = self.classify(F) if incidents else []

        # Classify each incident from its own snippets. A video can contain more
        # than one kind of event, and a single video-level label cannot express
        # that; scoring the interval alone also removes the surrounding ordinary
        # footage that otherwise dominates the pooled descriptor.
        inc_out = []
        for inc in incidents:
            lo = max(0, int(inc.start_frame // self.snippet_len))
            hi = min(F.shape[0], int(inc.end_frame // self.snippet_len) + 1)
            d = inc.to_dict()
            d["categories"] = self.classify(F[lo:hi]) if hi > lo else []
            inc_out.append(d)
        # Object detection for the review overlay, sampled inside the detected
        # intervals only so a long video is not scanned end to end.
        objects = None
        if detect_objects and incidents:
            try:
                objects = self.object_detector.detect_video(
                    path, fps_sample=object_fps,
                    intervals=[(i.start_sec, i.end_sec) for i in incidents])
            except Exception as exc:
                # Overlay data is optional; a detector failure must not lose the
                # anomaly result that has already been computed.
                objects = {"error": f"{type(exc).__name__}: {exc}"}

        if progress:
            progress(1.0)
        return {
            "fps": float(fps),
            "n_frames": int(n_frames),
            "n_snippets": int(F.shape[0]),
            "snippet_scores": [float(x) for x in snippet_scores],
            "frame_scores": [float(x) for x in frame_scores],
            "attention": None if attn is None else [float(x) for x in attn],
            "max_score": float(snippet_scores.max()),
            "mean_score": float(snippet_scores.mean()),
            "incidents": inc_out,
            "threshold": threshold,
            "model": self.model_kind,
            "backbone": self.backbone_name,
            # Ranked category guesses for the video as a whole, empty when no
            # incident was found or no category head is loaded.
            "categories": categories,
            "category_top1_accuracy": self.category_top1,
            "objects": objects,
        }

    def stream_session(self, **kw) -> "StreamSession":
        return StreamSession(self, **kw)


class StreamSession:
    """
    Live-camera session.

    Frames are pushed in; every ``snippet_len`` frames a feature is computed and
    appended to a rolling context window. The detector then scores the whole
    window, so the temporal model still sees history instead of judging each
    snippet alone -- the score returned is for the newest snippet.
    """

    def __init__(self, engine: VADEngine, context: int = 64, fps: float = 30.0,
                 threshold: float = 0.5):
        self.engine = engine
        self.context = context
        self.fps = fps
        self.threshold = threshold
        self.frame_buf: List[np.ndarray] = []
        self.feat_buf: deque = deque(maxlen=context)
        self.snippet_scores: List[float] = []
        self.n_frames = 0
        self._active: Optional[Dict] = None
        self.incidents: List[Dict] = []

    def push_frame(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """Add one frame. Returns a result dict when a snippet completes."""
        self.n_frames += 1
        rgb = _resize_short(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.frame_buf.append(rgb.astype(np.float32) / 255.0)
        if len(self.frame_buf) < self.engine.snippet_len:
            return None

        clip = np.stack(self.frame_buf, 0)
        self.frame_buf = []
        f = self.engine._features_from_clips([clip])[0]
        self.feat_buf.append(f)

        scores, _ = self.engine._score_features(np.stack(list(self.feat_buf), 0))
        score = float(scores[-1])
        self.snippet_scores.append(score)
        idx = len(self.snippet_scores) - 1
        t = (idx + 1) * self.engine.snippet_len / self.fps

        alert = score > self.threshold
        self._update_incident(alert, score, t, idx)
        return {"snippet_index": idx, "score": score, "time_sec": round(t, 2),
                "alert": alert, "n_incidents": len(self.incidents)}

    def _update_incident(self, alert: bool, score: float, t: float, idx: int) -> None:
        if alert:
            if self._active is None:
                self._active = {"start_sec": t, "peak_score": score, "peak_time": t,
                                "start_snippet": idx, "scores": [score]}
            else:
                self._active["scores"].append(score)
                if score > self._active["peak_score"]:
                    self._active.update(peak_score=score, peak_time=t)
        elif self._active is not None:
            self._close(t, idx)

    def _close(self, t: float, idx: int) -> None:
        a = self._active
        self._active = None
        if a is None:
            return
        self.incidents.append({
            "start_sec": round(a["start_sec"], 2), "end_sec": round(t, 2),
            "duration_sec": round(t - a["start_sec"], 2),
            "peak_score": round(a["peak_score"], 4),
            "mean_score": round(float(np.mean(a["scores"])), 4),
            "peak_time_sec": round(a["peak_time"], 2),
            "start_snippet": a["start_snippet"], "end_snippet": idx,
        })

    def finish(self) -> Dict:
        if self._active is not None:
            t = len(self.snippet_scores) * self.engine.snippet_len / self.fps
            self._close(t, len(self.snippet_scores) - 1)
        return {"n_frames": self.n_frames, "n_snippets": len(self.snippet_scores),
                "snippet_scores": self.snippet_scores, "incidents": self.incidents}
