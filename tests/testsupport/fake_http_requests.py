from unittest.mock import MagicMock


class FakeRequest:
    file_name = None
    content = b"some content"  # return fake content
    
    def __init__(self, *args, **kwargs  ):

        self.file_name = kwargs.get('files')
        self.status_code = 200
        self.request = MagicMock(return_value="some http requests")
        
    def raise_for_status(self):
        return None

    def json(self):
        return self.file_name