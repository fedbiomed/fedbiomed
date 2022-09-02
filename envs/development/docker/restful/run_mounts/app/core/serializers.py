from rest_framework.serializers import HyperlinkedModelSerializer

from .models import Upload


class UploadSerializer(HyperlinkedModelSerializer):
    class Meta:
        model = Upload
        fields = '__all__'
