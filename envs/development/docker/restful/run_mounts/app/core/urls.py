from rest_framework import routers

from.views import UploadViewSet

router = routers.DefaultRouter()
router.register(r'upload', UploadViewSet)

urlpatterns = router.urls
