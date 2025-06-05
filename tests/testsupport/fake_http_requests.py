from unittest.mock import MagicMock


class FakeRequest:
    """Mimicks the behaviour of the result of a `requests` object
    (from requests package). It is made generic for both HTTP POST 
    and HTTP GET methods. 
    For example, it mimicks the result of `requests.post` 
    object
    """
    file_name = None
    content = b"some content"  # return fake content

    def __init__(self, *args, **kwargs  ):
        """
        Constructor of the HTTP request. 
        kwargs takes `files` argument for more in-depth tests (if provided).
        Please see `requests` package documentation for further information
        (https://docs.python-requests.org/en/latest/user/quickstart)
        
        Details:
        - returns status_code by default = 200
        - returns "some http request" when calling `requests.request.method`
        """
        self.file_name = kwargs.get('files')
        self.status_code = 200
        self.request = MagicMock(method="some http requests")

    def raise_for_status(self):
        """Simulates `raise_for_status` method (see
        https://docs.python-requests.org/en/latest/api/#requests.Response.raise_for_status)"""
        return None

    def json(self):
        """Simualates `json` requests method
        (see https://docs.python-requests.org/en/latest/user/quickstart/#json-response-content)
        """
        return self.file_name
