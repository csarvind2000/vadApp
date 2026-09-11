"""REST API."""
from __future__ import annotations

import os

import cv2
import django_rq
from django.core.files.base import ContentFile
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .models import AnalysisJob, Camera, Incident, StreamSession, Video
from .paginations import StandardPagination
from .serializers import (AnalysisJobDetailSerializer, AnalysisJobSerializer,
                          CameraSerializer, IncidentSerializer,
                          StreamSessionSerializer, VideoSerializer)
from .utils import model_client
from .utils.jobs import run_analysis


@api_view(["GET"])
@permission_classes([AllowAny])   # a liveness probe must answer before auth
def health(request):
    """Liveness for the whole stack, including whether a model is actually loaded."""
    return Response({"backend": "ok", "model_service": model_client.health()})


def _probe_into(video: Video) -> None:
    """Read basic properties so the UI can show duration before analysis runs."""
    try:
        path = video.file.path
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video.fps = fps
            video.n_frames = n
            video.duration_sec = n / fps if fps else None
            video.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        video.size_bytes = os.path.getsize(path)
        video.save()
    except Exception:
        # Probing is best-effort; analysis re-reads these values anyway.
        pass


class VideoViewSet(viewsets.ModelViewSet):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer
    pagination_class = StandardPagination
    # Multipart for the file upload; JSON because the `analyze` action shares
    # these parsers and receives a JSON body. Omitting JSONParser makes that
    # action reject every request with 415.
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def perform_create(self, serializer):
        video = serializer.save(
            owner=self.request.user if self.request.user.is_authenticated else None)
        _probe_into(video)

    @action(detail=True, methods=["post"])
    def analyze(self, request, pk=None):
        """Queue an analysis job. Returns immediately with the job id to poll."""
        video = self.get_object()
        job = AnalysisJob.objects.create(
            video=video,
            threshold=float(request.data.get("threshold", 0.5)),
            min_duration_sec=float(request.data.get("min_duration_sec", 1.0)),
        )
        rq_job = django_rq.get_queue("default").enqueue(
            run_analysis, str(job.id), job_timeout=3600)
        job.rq_job_id = rq_job.id
        job.save(update_fields=["rq_job_id"])
        return Response(AnalysisJobSerializer(job).data, status=status.HTTP_202_ACCEPTED)


class AnalysisJobViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AnalysisJob.objects.select_related("video").prefetch_related("incidents")
    pagination_class = StandardPagination

    def get_serializer_class(self):
        return AnalysisJobDetailSerializer if self.action == "retrieve" else AnalysisJobSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        video = self.request.query_params.get("video")
        return qs.filter(video_id=video) if video else qs


class IncidentViewSet(viewsets.ModelViewSet):
    queryset = Incident.objects.select_related("video", "camera")
    serializer_class = IncidentSerializer
    pagination_class = StandardPagination

    def get_queryset(self):
        qs = super().get_queryset()
        p = self.request.query_params
        if p.get("video"):
            qs = qs.filter(video_id=p["video"])
        if p.get("camera"):
            qs = qs.filter(camera_id=p["camera"])
        if p.get("review"):
            qs = qs.filter(review=p["review"])
        if p.get("min_score"):
            qs = qs.filter(peak_score__gte=float(p["min_score"]))
        order = p.get("ordering")
        return qs.order_by(order) if order else qs.order_by("-detected_at")


class CameraViewSet(viewsets.ModelViewSet):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer
    pagination_class = StandardPagination


class StreamSessionViewSet(viewsets.ModelViewSet):
    """
    Live monitoring.

    The browser opens a session, posts JPEG frames, and closes it. Frames are
    relayed to the model service, which keeps the rolling temporal context.
    """
    queryset = StreamSession.objects.select_related("camera")
    serializer_class = StreamSessionSerializer
    pagination_class = StandardPagination
    # `create` and `close` send JSON; `frame` sends a multipart JPEG.
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def create(self, request, *args, **kwargs):
        camera_id = request.data.get("camera")
        camera = Camera.objects.filter(id=camera_id).first() if camera_id else None
        threshold = float(request.data.get(
            "threshold", camera.threshold if camera else 0.5))
        fps = float(request.data.get("fps", 30.0))
        try:
            engine_sid = model_client.stream_open(fps=fps, threshold=threshold)
        except model_client.ModelServiceError as e:
            return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        session = StreamSession.objects.create(
            camera=camera, engine_session_id=engine_sid, threshold=threshold, fps=fps)
        return Response(self.get_serializer(session).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["post"])
    def frame(self, request, pk=None):
        session = self.get_object()
        if session.status != StreamSession.Status.OPEN:
            return Response({"error": "session is not open"},
                            status=status.HTTP_400_BAD_REQUEST)
        f = request.FILES.get("frame")
        if f is None:
            return Response({"error": "missing 'frame'"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            res = model_client.stream_frame(session.engine_session_id, f.read())
        except model_client.ModelServiceError as e:
            session.status = StreamSession.Status.ERROR
            session.save(update_fields=["status"])
            return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        session.n_frames += 1
        update = ["n_frames"]
        if not res.get("buffering"):
            session.n_snippets = res.get("snippet_index", session.n_snippets) + 1
            update.append("n_snippets")
        session.save(update_fields=update)
        return Response(res)

    @action(detail=True, methods=["post"])
    def close(self, request, pk=None):
        session = self.get_object()
        if session.status != StreamSession.Status.OPEN:
            return Response(self.get_serializer(session).data)
        try:
            summary = model_client.stream_close(session.engine_session_id)
        except model_client.ModelServiceError as e:
            session.status = StreamSession.Status.ERROR
            session.save(update_fields=["status"])
            return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        session.snippet_scores = summary.get("snippet_scores")
        session.n_snippets = summary.get("n_snippets", session.n_snippets)
        session.status = StreamSession.Status.CLOSED
        session.ended_at = timezone.now()
        session.save()

        # Persist live detections as incidents so live and offline runs land in
        # one reviewable table.
        for inc in summary.get("incidents", []):
            Incident.objects.create(
                camera=session.camera,
                start_sec=inc["start_sec"], end_sec=inc["end_sec"],
                duration_sec=inc["duration_sec"],
                peak_score=inc["peak_score"], mean_score=inc["mean_score"],
            )
        data = self.get_serializer(session).data
        data["incidents_created"] = len(summary.get("incidents", []))
        return Response(data)
