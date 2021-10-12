from rest_framework.viewsets import ModelViewSet

from .models import Upload
from .serializers import UploadSerializer


class UploadViewSet(ModelViewSet):
    serializer_class = UploadSerializer
    queryset = Upload.objects.all().order_by('-created_at')
