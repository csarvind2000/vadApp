"""Detected anomaly incidents."""
from __future__ import annotations

import uuid

from django.db import models


def thumb_path(instance, filename: str) -> str:
    return f"incidents/{instance.job_id}/{instance.id}_{filename}"


class Incident(models.Model):
    """
    A contiguous stretch of video the detector scored above threshold.

    This is the record a user actually cares about, so it is persisted as a row
    (not merely derived from the score curve on the fly) with a thumbnail at the
    peak frame and a reviewer verdict for audit.
    """

    class Review(models.TextChoices):
        UNREVIEWED = "unreviewed", "Unreviewed"
        CONFIRMED = "confirmed", "Confirmed"
        FALSE_POSITIVE = "false_positive", "False positive"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey("vadserver.AnalysisJob", on_delete=models.CASCADE,
                            related_name="incidents", null=True, blank=True)
    video = models.ForeignKey("vadserver.Video", on_delete=models.CASCADE,
                              related_name="incidents", null=True, blank=True)
    camera = models.ForeignKey("vadserver.Camera", on_delete=models.SET_NULL,
                               related_name="incidents", null=True, blank=True)

    start_sec = models.FloatField()
    end_sec = models.FloatField()
    duration_sec = models.FloatField()
    peak_score = models.FloatField()
    mean_score = models.FloatField()
    peak_frame = models.IntegerField(null=True, blank=True)

    # Category ranking for this interval alone. A video may contain more than
    # one kind of event, which a single video-level label cannot express.
    categories = models.JSONField(null=True, blank=True)

    thumbnail = models.ImageField(upload_to=thumb_path, null=True, blank=True)
    clip = models.FileField(upload_to="incident_clips/", null=True, blank=True)

    review = models.CharField(max_length=20, choices=Review.choices,
                              default=Review.UNREVIEWED)
    note = models.TextField(blank=True, default="")

    detected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-peak_score"]
        indexes = [
            models.Index(fields=["-detected_at"]),
            models.Index(fields=["review"]),
        ]

    def __str__(self) -> str:
        return f"incident {self.start_sec:.1f}-{self.end_sec:.1f}s score={self.peak_score:.2f}"
