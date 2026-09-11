from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from vadserver import views

router = DefaultRouter()
router.register(r"videos", views.VideoViewSet)
router.register(r"jobs", views.AnalysisJobViewSet)
router.register(r"incidents", views.IncidentViewSet)
router.register(r"cameras", views.CameraViewSet)
router.register(r"streams", views.StreamSessionViewSet)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include(router.urls)),
    path("api/health/", views.health, name="health"),
    path("api/auth/", include("rest_framework.urls")),
    path("django-rq/", include("django_rq.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
