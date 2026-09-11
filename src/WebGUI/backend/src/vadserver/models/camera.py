"""Live camera sources and their streaming sessions."""
from __future__ import annotations

import uuid

from django.db import models


class Camera(models.Model):
    """A named live source: a browser webcam, or an RTSP/HTTP URL."""

    class Kind(models.TextChoices):
        BROWSER = "browser", "Browser webcam"
        RTSP = "rtsp", "RTSP / HTTP stream"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=128)
    kind = models.CharField(max_length=16, choices=Kind.choices, default=Kind.BROWSER)
    url = models.CharField(max_length=512, blank=True, default="")
    location = models.CharField(max_length=255, blank=True, default="")
    threshold = models.FloatField(default=0.5)
    enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class StreamSession(models.Model):
    """
    One live monitoring run.

    Holds the id issued by the model service so the backend can forward frames
    and close the session cleanly when the browser goes away.
    """

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        CLOSED = "closed", "Closed"
        ERROR = "error", "Error"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE,
                               related_name="sessions", null=True, blank=True)
    engine_session_id = models.CharField(max_length=64, blank=True, default="")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.OPEN)
    threshold = models.FloatField(default=0.5)
    fps = models.FloatField(default=30.0)

    n_frames = models.IntegerField(default=0)
    n_snippets = models.IntegerField(default=0)
    snippet_scores = models.JSONField(null=True, blank=True)

    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"stream {self.id} [{self.status}]"
