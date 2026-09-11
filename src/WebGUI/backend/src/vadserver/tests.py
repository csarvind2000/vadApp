"""
Contract tests: every endpoint is exercised with the content type the frontend
actually sends.

These exist because a viewset that declares `parser_classes` applies them to its
`@action` methods too. Declaring only multipart parsers for a file-upload viewset
silently breaks any JSON action on it with 415 -- which is invisible until the
browser hits it.
"""
import io
import json
from unittest import mock

from django.test import TestCase
from rest_framework.test import APIClient

from .models import AnalysisJob, Camera, Incident, StreamSession, Video


def _fake_video(name="t.mp4"):
    # Content is never decoded in these tests; only the request path is checked.
    return io.BytesIO(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 512), name


class EndpointContractTests(TestCase):
    def setUp(self):
        self.c = APIClient()

    # -- reads ------------------------------------------------------------
    def test_health_is_public_and_reports_model(self):
        with mock.patch("vadserver.views.model_client.health",
                        return_value={"status": "ok"}):
            r = self.c.get("/api/health/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["backend"], "ok")

    def test_list_endpoints_return_200(self):
        for ep in ["videos", "jobs", "incidents", "cameras", "streams"]:
            with self.subTest(ep=ep):
                self.assertEqual(self.c.get(f"/api/{ep}/").status_code, 200)

    # -- writes, each with the content type the frontend uses -------------
    def test_video_upload_accepts_multipart(self):
        buf, name = _fake_video()
        with mock.patch("vadserver.views._probe_into"):
            r = self.c.post("/api/videos/", {"file": buf, "name": "clip"},
                            format="multipart")
        self.assertEqual(r.status_code, 201, r.content[:300])

    def test_analyze_action_accepts_json(self):
        """Regression: this returned 415 when the viewset had no JSONParser."""
        buf, name = _fake_video()
        with mock.patch("vadserver.views._probe_into"):
            v = self.c.post("/api/videos/", {"file": buf, "name": "clip"},
                            format="multipart").json()
        with mock.patch("vadserver.views.django_rq.get_queue") as q:
            q.return_value.enqueue.return_value = mock.Mock(id="rq-1")
            r = self.c.post(f"/api/videos/{v['id']}/analyze/",
                            {"threshold": 0.6, "min_duration_sec": 1.0},
                            format="json")
        self.assertEqual(r.status_code, 202, r.content[:300])
        self.assertEqual(AnalysisJob.objects.get().threshold, 0.6)

    def test_camera_create_accepts_json(self):
        r = self.c.post("/api/cameras/",
                        {"name": "Cam 1", "kind": "browser", "threshold": 0.6},
                        format="json")
        self.assertEqual(r.status_code, 201, r.content[:300])

    def test_stream_open_accepts_json(self):
        """Regression: Live Monitor hit the same 415 as the analyze action."""
        with mock.patch("vadserver.views.model_client.stream_open",
                        return_value="engine-sid"):
            r = self.c.post("/api/streams/", {"threshold": 0.5, "fps": 10},
                            format="json")
        self.assertEqual(r.status_code, 201, r.content[:300])
        self.assertEqual(StreamSession.objects.get().engine_session_id, "engine-sid")

    def test_stream_frame_accepts_multipart(self):
        with mock.patch("vadserver.views.model_client.stream_open",
                        return_value="engine-sid"):
            sid = self.c.post("/api/streams/", {"threshold": 0.5, "fps": 10},
                              format="json").json()["id"]
        with mock.patch("vadserver.views.model_client.stream_frame",
                        return_value={"snippet_index": 0, "score": 0.9, "alert": True}):
            r = self.c.post(f"/api/streams/{sid}/frame/",
                            {"frame": io.BytesIO(b"\xff\xd8\xff\xe0jpeg")},
                            format="multipart")
        self.assertEqual(r.status_code, 200, r.content[:300])

    def test_stream_close_persists_incidents(self):
        with mock.patch("vadserver.views.model_client.stream_open",
                        return_value="engine-sid"):
            sid = self.c.post("/api/streams/", {"threshold": 0.5, "fps": 10},
                              format="json").json()["id"]
        summary = {"n_snippets": 3, "snippet_scores": [0.1, 0.9, 0.2],
                   "incidents": [{"start_sec": 1.0, "end_sec": 3.0, "duration_sec": 2.0,
                                  "peak_score": 0.9, "mean_score": 0.7}]}
        with mock.patch("vadserver.views.model_client.stream_close", return_value=summary):
            r = self.c.post(f"/api/streams/{sid}/close/")
        self.assertEqual(r.status_code, 200, r.content[:300])
        self.assertEqual(Incident.objects.count(), 1)
        self.assertEqual(StreamSession.objects.get().status, "closed")

    def test_incident_review_accepts_patch_json(self):
        inc = Incident.objects.create(start_sec=1, end_sec=2, duration_sec=1,
                                      peak_score=0.9, mean_score=0.8)
        r = self.c.patch(f"/api/incidents/{inc.id}/",
                         {"review": "confirmed"}, format="json")
        self.assertEqual(r.status_code, 200, r.content[:300])
        inc.refresh_from_db()
        self.assertEqual(inc.review, "confirmed")

    # -- failure propagation ---------------------------------------------
    def test_stream_open_reports_model_service_failure(self):
        from vadserver.utils.model_client import ModelServiceError
        with mock.patch("vadserver.views.model_client.stream_open",
                        side_effect=ModelServiceError("unreachable")):
            r = self.c.post("/api/streams/", {"threshold": 0.5}, format="json")
        self.assertEqual(r.status_code, 503)
