"""
Background jobs run by the RQ workers.

Analysis is queued rather than done in the request so a long video cannot block
a web worker, and so progress is visible while it runs.
"""
from __future__ import annotations

import os
import traceback

import cv2
from django.utils import timezone

from ..models import AnalysisJob, Incident
from . import model_client


def _save_thumbnail(incident: Incident, video_path: str, frame_idx: int) -> None:
    """Grab the peak frame so the incident list is scannable at a glance."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = cap.read()
        if not ok:
            return
        h, w = frame.shape[:2]
        scale = 320 / max(w, 1)
        if scale < 1:
            frame = cv2.resize(frame, (320, max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            from django.core.files.base import ContentFile
            incident.thumbnail.save(f"peak_{frame_idx}.jpg",
                                    ContentFile(buf.tobytes()), save=True)
    finally:
        cap.release()


def run_analysis(job_id: str) -> str:
    """Analyse one video and persist its incidents. Called by an RQ worker."""
    job = AnalysisJob.objects.select_related("video").get(id=job_id)
    job.status = AnalysisJob.Status.RUNNING
    job.progress = 0.0
    job.save(update_fields=["status", "progress"])

    try:
        path = job.video.file.path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"video file missing: {path}")

        result = model_client.analyze(
            path, threshold=job.threshold, min_duration_sec=job.min_duration_sec)

        video = job.video
        video.fps = result.get("fps")
        video.n_frames = result.get("n_frames")
        if video.fps:
            video.duration_sec = (video.n_frames or 0) / video.fps
        video.save(update_fields=["fps", "n_frames", "duration_sec"])

        job.snippet_scores = result.get("snippet_scores")
        job.attention = result.get("attention")
        job.max_score = result.get("max_score")
        job.mean_score = result.get("mean_score")
        job.model_name = result.get("model", "")
        job.backbone = result.get("backbone", "")
        job.categories = result.get("categories") or None
        job.detections = result.get("objects") or None

        # Replace incidents for this job so a re-run is idempotent.
        job.incidents.all().delete()
        for inc in result.get("incidents", []):
            obj = Incident.objects.create(
                job=job, video=video,
                start_sec=inc["start_sec"], end_sec=inc["end_sec"],
                duration_sec=inc["duration_sec"],
                peak_score=inc["peak_score"], mean_score=inc["mean_score"],
                peak_frame=inc.get("peak_frame"),
                categories=inc.get("categories") or None,
            )
            if obj.peak_frame is not None:
                _save_thumbnail(obj, path, obj.peak_frame)

        job.status = AnalysisJob.Status.DONE
        job.progress = 1.0
        job.finished_at = timezone.now()
        job.save()
        return f"ok: {len(result.get('incidents', []))} incidents"

    except Exception as exc:
        job.status = AnalysisJob.Status.FAILED
        job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "error", "finished_at"])
        raise
