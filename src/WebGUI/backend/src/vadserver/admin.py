from django.contrib import admin

from .models import AnalysisJob, Camera, Incident, StreamSession, Video


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ("name", "source", "duration_sec", "fps", "created_at")
    list_filter = ("source", "created_at")
    search_fields = ("name",)


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ("id", "video", "status", "progress", "max_score", "created_at")
    list_filter = ("status", "model_name")


@admin.register(Incident)
class IncidentAdmin(admin.ModelAdmin):
    list_display = ("id", "video", "camera", "start_sec", "end_sec",
                    "peak_score", "review", "detected_at")
    list_filter = ("review", "detected_at")


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "kind", "location", "threshold", "enabled")
    list_filter = ("kind", "enabled")


@admin.register(StreamSession)
class StreamSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "camera", "status", "n_frames", "n_snippets", "started_at")
    list_filter = ("status",)
