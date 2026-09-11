"""DRF serializers."""
from rest_framework import serializers

from .models import AnalysisJob, Camera, Incident, StreamSession, Video


class IncidentSerializer(serializers.ModelSerializer):
    thumbnail_url = serializers.SerializerMethodField()
    video_name = serializers.CharField(source="video.name", read_only=True)
    camera_name = serializers.CharField(source="camera.name", read_only=True, default=None)

    class Meta:
        model = Incident
        fields = ["id", "job", "video", "video_name", "camera", "camera_name",
                  "start_sec", "end_sec", "duration_sec", "peak_score", "mean_score",
                  "peak_frame", "categories", "thumbnail_url", "review", "note",
                  "detected_at"]
        read_only_fields = ["id", "detected_at", "duration_sec"]

    def get_thumbnail_url(self, obj):
        if not obj.thumbnail:
            return None
        request = self.context.get("request")
        url = obj.thumbnail.url
        return request.build_absolute_uri(url) if request else url


class AnalysisJobSerializer(serializers.ModelSerializer):
    n_incidents = serializers.IntegerField(source="incidents.count", read_only=True)

    class Meta:
        model = AnalysisJob
        # snippet_scores is deliberately excluded from the list payload: it is a
        # long array and would dominate the response. Fetch it via the detail route.
        fields = ["id", "video", "status", "progress", "error", "threshold",
                  "min_duration_sec", "model_name", "backbone", "max_score",
                  "mean_score", "categories", "n_incidents", "created_at",
                  "finished_at"]
        read_only_fields = fields


class AnalysisJobDetailSerializer(AnalysisJobSerializer):
    incidents = IncidentSerializer(many=True, read_only=True)

    class Meta(AnalysisJobSerializer.Meta):
        fields = AnalysisJobSerializer.Meta.fields + [
            "snippet_scores", "attention", "detections", "incidents"]
        read_only_fields = fields


class VideoSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()
    latest_job = serializers.SerializerMethodField()
    n_incidents = serializers.IntegerField(source="incidents.count", read_only=True)

    class Meta:
        model = Video
        fields = ["id", "name", "file", "file_url", "source", "duration_sec", "fps",
                  "n_frames", "width", "height", "size_bytes", "created_at",
                  "latest_job", "n_incidents"]
        read_only_fields = ["id", "created_at", "duration_sec", "fps", "n_frames",
                            "width", "height", "size_bytes"]
        extra_kwargs = {"file": {"write_only": True}}

    def get_file_url(self, obj):
        if not obj.file:
            return None
        request = self.context.get("request")
        return request.build_absolute_uri(obj.file.url) if request else obj.file.url

    def get_latest_job(self, obj):
        job = obj.jobs.first()          # Meta.ordering puts the newest first
        return AnalysisJobSerializer(job).data if job else None


class CameraSerializer(serializers.ModelSerializer):
    n_incidents = serializers.IntegerField(source="incidents.count", read_only=True)

    class Meta:
        model = Camera
        fields = ["id", "name", "kind", "url", "location", "threshold", "enabled",
                  "created_at", "n_incidents"]
        read_only_fields = ["id", "created_at"]


class StreamSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = StreamSession
        fields = ["id", "camera", "status", "threshold", "fps", "n_frames",
                  "n_snippets", "started_at", "ended_at"]
        read_only_fields = ["id", "started_at", "ended_at", "n_frames", "n_snippets"]
