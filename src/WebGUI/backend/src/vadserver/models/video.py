"""Uploaded videos and the analysis jobs run over them."""
from __future__ import annotations

import uuid

from django.contrib.auth import get_user_model
from django.db import models


def video_upload_path(instance, filename: str) -> str:
    return f"videos/{instance.id}/{filename}"


class Video(models.Model):
    """A source video: either uploaded by a user or recorded from a camera."""

    class Source(models.TextChoices):
        UPLOAD = "upload", "Upload"
        CAMERA = "camera", "Camera"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to=video_upload_path)
    source = models.CharField(max_length=16, choices=Source.choices, default=Source.UPLOAD)
    owner = models.ForeignKey(get_user_model(), on_delete=models.CASCADE,
                              related_name="videos", null=True, blank=True)

    # Filled in once the file has been probed.
    duration_sec = models.FloatField(null=True, blank=True)
    fps = models.FloatField(null=True, blank=True)
    n_frames = models.IntegerField(null=True, blank=True)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    size_bytes = models.BigIntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.id})"


class AnalysisJob(models.Model):
    """
    One run of the detector over one video.

    Kept separate from Video so a video can be re-analysed with a different
    threshold or a newer checkpoint without destroying earlier results -- which
    is what makes the incident history auditable.
    """

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="jobs")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    progress = models.FloatField(default=0.0)
    error = models.TextField(blank=True, default="")

    # Parameters, recorded so a result can always be traced to how it was made.
    threshold = models.FloatField(default=0.5)
    min_duration_sec = models.FloatField(default=1.0)
    model_name = models.CharField(max_length=64, blank=True, default="")
    backbone = models.CharField(max_length=64, blank=True, default="")

    # Score curve for plotting. Stored at snippet resolution, not frame, which is
    # ~16x smaller and is the resolution the model actually predicts at.
    snippet_scores = models.JSONField(null=True, blank=True)
    attention = models.JSONField(null=True, blank=True)
    max_score = models.FloatField(null=True, blank=True)
    mean_score = models.FloatField(null=True, blank=True)

    # Ranked anomaly-category guesses, e.g.
    #   [{"category": "Shooting", "probability": 0.41}, ...]
    # Stored as a ranking rather than a single label because top-1 accuracy is
    # 0.38 over 13 classes; the UI must show the confidence alongside.
    categories = models.JSONField(null=True, blank=True)

    # Object-detection overlay: sampled frames with normalised boxes, plus per
    # class totals. Presentation only -- it does not affect the anomaly score.
    # Named `detections` rather than `objects`, which Django reserves for the
    # default model Manager.
    detections = models.JSONField(null=True, blank=True)

    rq_job_id = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"job {self.id} [{self.status}] on {self.video_id}"
