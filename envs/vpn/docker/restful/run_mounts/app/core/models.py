from django.db import models

import uuid

# defining database model

class Upload(models.Model):
    # Django Database model is defined as follow:
    # - an unique id (as primary key)
    # - a file path
    # - a timestamp
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(upload_to='uploads/%Y/%m/%d', null=False)

    created_at = models.DateTimeField(auto_now=True, blank=False, editable=False)
