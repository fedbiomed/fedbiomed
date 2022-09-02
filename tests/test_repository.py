import os
from typing import Callable
import requests
import builtins
from json import JSONDecodeError
import unittest
from unittest.mock import MagicMock, patch

from testsupport.fake_http_requests import FakeRequest
from fedbiomed.common.repository import Repository
from fedbiomed.common.exceptions import FedbiomedRepositoryError


class TestRepository(unittest.TestCase):
    """
    Runs unit tests for Repository class (from fedbiomed.common.repository)

    """
    class OpenMock:
        """
        Mimicks the builtin function `open`
        Fakes the method `write` when calling `open(somefile).write()`
        """
        file_name = None

        def write(self, content):
            self.content = content

    # before the tests
    def setUp(self):

        # creating arguments
        self.uploads_url = 'http://a.fake.url'
        self.builtin_open_fake = TestRepository.OpenMock()

        # instanciating Reporsitory objects 
        self.r1 = Repository(uploads_url=self.uploads_url,
                             tmp_dir='/my/temporary_folder/path',
                             cache_dir='/path/to/my/cache/dir')

        self.r2 = Repository(None, None, None)

        def open_side_effect(file_name, mode):
            self.builtin_open_fake.file_name = file_name
            return self.builtin_open_fake   
        self.open_side_effect = open_side_effect

    # after the tests
    def tearDown(self):
        pass

    @patch('fedbiomed.common.repository.Repository._raise_for_status_handler')
    @patch('fedbiomed.common.repository.Repository._request_handler')
    @patch('requests.post')
    @patch('builtins.open')
    def test_reporistory_01_upload_file_normal_case(self, 
                                                    builtins_open_patch, 
                                                    requests_post_patch,
                                                    request_handler_patch,
                                                    raise_for_status_handler_patch):
        """
        Tests `upload_file` method in the normal case scenario
        """

        # arguments
        fake_filename = 'my/file/to/upload'

        # side effect funtions    

        def request_handler_side_effect(callable_method: Callable,
                                        url: str,
                                        filename: str,
                                        *args,
                                        **kwargs) -> FakeRequest:
            """Mimicks `_request_handler` private method of `Repository` class.

            Args:
                callable_method (Callable): a callable method (unused in this test)
                url (str): a url to connect to (unused in this test)
                filename (str): the name of the file to upload
                req_method (str): the name of the HTTP request (should be "POST")

            Returns:
                FakeRequest: a FakeRequest object that mimicks the result of 
                a Request
            """
            fake_req = FakeRequest(files = kwargs.get('files'))
            setattr(fake_req, 'request', 'POST')
            return fake_req

        # patches & Mocking
        builtins_open_patch.side_effect = self.open_side_effect
        request_handler_patch.side_effect = request_handler_side_effect
        raise_for_status_handler_patch.return_value = None

        # action
        res = self.r1.upload_file(fake_filename)

        # checks
        # check correct calls
        builtins_open_patch.assert_called_once_with(fake_filename, 'rb')
        request_handler_patch.assert_called_once_with(requests_post_patch,
                                                      self.uploads_url,
                                                      fake_filename,
                                                      files={'file': self.builtin_open_fake})

        # check result of request
        self.assertEqual(res, {'file': self.builtin_open_fake})
        self.assertEqual(fake_filename, self.builtin_open_fake.file_name)

    def test_repository_02_upload_file_open_exceptions(self):
        """
        Tests `upload_file` method, under the case where builtins open method triggeres exception.
        It could raises 3 kind of exceptions:
        1. a FileNotFoundError due to a file not found on system
        2. a PermissionError due to unsatisfactory privileges
        3. a OSError due to the unability to read specified file

        When those errors are triggered, `upload_file` must catch those errors and trigger instead
        a FedBiomedRepositoryError with the corresponding message wrt the occured error.
        """
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.upload_file('/a/file/that/should/not/be/found/on/your/computer')

        with patch.object(builtins, 'open') as open_mock:
            open_mock.side_effect = PermissionError("mimicking a permission error when accessing to file")
            with self.assertRaises(FedbiomedRepositoryError):
                self.r1.upload_file('/a/file/to/upload')

            open_mock.side_effect = OSError('mimicking an OSError when trying to read a file that cannot be read using'
                                            ' builtin `open` function')
            with self.assertRaises(FedbiomedRepositoryError):
                self.r1.upload_file('another/file/to/upload')

    @patch('fedbiomed.common.repository.Repository._raise_for_status_handler')
    @patch('fedbiomed.common.repository.Repository._request_handler')
    @patch('builtins.open')       
    def test_repository_03_upload_file_json_deserialize_exception(self,
                                                                  builtins_open_patch, 
                                                                  request_handler_patch,
                                                                  raise_for_status_handler_patch):
        """
        Checks if in `upload_file` JSON DecodeError is handled correclty when it is triggered 
        during message deserialization.
        """                                                  
        # patches and mocks
        builtins_open_patch.return_value = None
        requests_post = MagicMock(return_value=None)
        requests_post.method = MagicMock(return_value=None)

        requests_post.json = MagicMock(side_effect=JSONDecodeError("mimicking a eception occuring when a JSON message"
                                                                   "is not deserializable", doc='a_doc', pos=22))
        request_handler_patch.return_value = requests_post
        raise_for_status_handler_patch.return_value = None

        # action & checks
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.upload_file('a/file/to/upload')

    @patch('builtins.open')
    @patch('fedbiomed.common.repository.Repository._raise_for_status_handler')
    @patch('fedbiomed.common.repository.Repository._request_handler')
    @patch('requests.get')
    def test_reporistory_04_download_file_normal_case(self,
                                                      requests_get_patch,
                                                      request_handler_patch,
                                                      raise_for_status_handler_patch,
                                                      open_patch):
        """
        Tests  `download_file` Repository method in the normal case scenario.
        """

        # arguments
        url = 'http://a.url.from.which.to?download#file'
        path_file = '/a/path/to/a/file/on/which/downloaded/content/will/be/saved'
        expected_path_file = os.path.join(self.r1.tmp_dir, path_file)

        # Patches & Mocks
        request_handler_patch.side_effect = FakeRequest
        raise_for_status_handler_patch.return_value = None
        open_patch.side_effect = self.open_side_effect

        # action
        status_code, filepath = self.r1.download_file(url,
                                                      path_file)

        # checks
        request_handler_patch.assert_called_once_with(requests_get_patch, 
                                                      url,
                                                      path_file)
        raise_for_status_handler_patch.assert_called_once()
        open_patch.assert_called_once_with(expected_path_file,
                                           'wb')

        self.assertEqual(filepath, expected_path_file)
        self.assertEqual(status_code, 200)  # HTTP request should be ok (status code = 200)

    @patch('builtins.open')
    @patch('fedbiomed.common.repository.Repository._raise_for_status_handler')
    @patch('fedbiomed.common.repository.Repository._request_handler')
    def test_repository_05_download_file_open_exceptions(self, 
                                                         request_handler_patch, 
                                                         raise_for_status_patch,
                                                         open_patch):
        """
        Tests exceptions regarding file opening are appropriately handled
        in `download_file` method. 

        In this test we will trigger :
        - FileNotFoundError
        - PermissionError
        - OSError
        - MemoryError
        """
        # arguments
        url = 'http://a.url.from.which.to?download#file'
        path_file = '/a/path/to/a/file/on/which/downloaded/content/will/be/saved'

        # patches and mocks
        request_handler_patch.return_value = None
        raise_for_status_patch.return_value = None

        # check FileNotFoundError
        open_patch.side_effect = FileNotFoundError("Mimicking case where directory is not exisiting")

        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.download_file(url, path_file)
        # check PermissionError
        open_patch.side_effect = PermissionError("Mimicking case where file cannot be write due to"
                                                 " some permission error")

        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.download_file(url, path_file)

        # check MemoryError

        open_mock = MagicMock(return_value = None)
        open_mock.write = MagicMock(side_effect=MemoryError("mimicking case where there is no available"
                                                            " space on system disk"))
        request_handler_patch.return_value = FakeRequest()
        open_patch.side_effect = None
        open_patch.return_value = open_mock
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.download_file(url, path_file)

        # check OSError
        open_mock.write = MagicMock(side_effect=OSError("mimicking case where file cannot be read"
                                                        " (eg used by another process)"))
        open_patch.return_value = open_mock

        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.download_file(url, path_file)

    @patch('fedbiomed.common.repository.Repository._get_method_request_msg')
    def test_repository_06_private_raise_for_status_handler_normal_case(self, 
                                                                        get_method_req_patch):
        """
        Tests private method `_raise_for_status_handler` in the normal case scenario 
        (when no errors are triggered)
        """

        req = FakeRequest()
        get_method_req_patch.return_value = "issuing some HTTP requests"

        # action & checks
        with patch.object(FakeRequest, 'raise_for_status') as raise_for_status_mock:
            self.r1._raise_for_status_handler(req, '/a/file/on/my/computer')
            # we are checking here if `raise_for_status` has been called
            raise_for_status_mock.assert_called_once()

    @patch('fedbiomed.common.repository.Repository._get_method_request_msg')       
    def test_repository_07_private_raise_for_status_exceptions(self,
                                                               get_method_req_patch):
        """
        Tests private method `_raise_for_status_handler` when HTTP request status code is 
        either 404 or 500, and checks if error is raised.
        """
        # run a first test that triggers HTTPError with status code error 404
        # patches and mocks definintion

        requests_mock = MagicMock(return_value=None)
        requests_mock.raise_for_status = MagicMock(side_effect=requests.HTTPError("mimicking an HTTP error"))
        requests_mock.status_code = 404  

        get_method_req_patch.return_value = "issing a HTTP request"

        # action & checks
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._raise_for_status_handler(requests_mock,
                                              'my/file/to/upload')

        # run a second test that triggers HTTPError with status code error 500
        # patches and mocks defintion

        requests_mock.status_code = 500

        # action & checks
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._raise_for_status_handler(requests_mock, 'my/other/file/to/upload')

    def test_repository_08_get_method_request(self):
        """
        Tests priavte method `_get_method_request` of Repository for different 
        HTTP requests ('GET', 'POST' or unknown)
        """
        # test 1 with a HTTP POST request

        msg_test_1 = Repository._get_method_request_msg("post")  
        self.assertEqual(msg_test_1, "uploading file")  

        # test 2 with a HTTP GET request

        msg_test_2 = Repository._get_method_request_msg("get")
        self.assertEqual(msg_test_2, 'downloading file')

        # test 3 with an unknown HTTP request

        msg_test_3 = self.r1._get_method_request_msg('unknown request')
        self.assertEqual('issuing unknown HTTP request', msg_test_3)

    @patch('fedbiomed.common.repository.Repository._get_method_request_msg')
    def test_repository_09_private_request_handler_normal_case(self, 
                                                               get_method_req_patch):
        """
        Tests normal case scenario when using `_request_handler` 
        private method
        """
        # side effect function definition
        def a_callable(url):
            return FakeRequest(url)

        # patches and mocks
        get_method_req_patch.return_value = "do some operation on file"
        request_callable = MagicMock(side_effect = a_callable, __name__='GET')

        # action
        res = self.r1._request_handler(request_callable, 
                                       'http://a.fake.url',
                                       'a/file/path')

        # checks
        request_callable.assert_called_once()
        self.assertIsInstance(res, FakeRequest)

    @patch('fedbiomed.common.repository.Repository._get_method_request_msg')           
    def test_repository_10_private_request_handler_exceptions(self,
                                                              get_method_req_msg_patch):
        """
        Checks that errors tirggered through `requests.post` (ie when issuing a HTTP POST Request)
        have been handled accordingly in `upload_file` method
        Errors triggered during unit test runtime executions are :
        1. requests.Timeout
        2. requests.TooManyRedirects
        3. requests.connectionError
        4. requests.InvalidSchema
        5. requests.RequestException
        """
        # arguments
        url = 'http://a.fake.url'
        filename = 'a/file/path'
        req_method = "issuing a unknown http method"

        # defining callables for test
        def callable_raising_timeout_exception(*args, **kwargs):
            """Function that raises a `requests.Timeout` exception"""
            raise requests.Timeout("Mimicking a Timeout error due to a connection that"
                                   " exceeded timeout")

        def callable_raising_too_many_redirection_errors(*args, **kwrags):
            """Function that raises a `requests.TooManyRedirects` exception"""
            raise requests.TooManyRedirects("Mimicking a TooManyRedirectsError due to reaching"
                                            " `max_redirection` threshold")

        def callable_raising_requests_error(*args, **kwargs):
            """Function that raises a `requests.RequestException`. This exception
            is the global exception of `requests package"""
            raise requests.RequestException("Mimicking a unknown RequestException")


        # patches
        get_method_req_msg_patch.return_value = 'issuing a unknwon HTTP request'

        # performing tests
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._request_handler(callable_raising_timeout_exception,
                                     url,
                                     filename,
                                     req_method)

        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._request_handler(callable_raising_too_many_redirection_errors, 
                                     url,
                                     filename, 
                                     req_method)

        with self.assertRaises(FedbiomedRepositoryError):
            # should return a requests.InvalidSchema error
            self.r2._request_handler(requests.post, 
                                     'a/file/to/upload',
                                     filename, 
                                     req_method)

        with self.assertRaises(FedbiomedRepositoryError):
            # should return a connectionError due to unknown server
            self.r1._request_handler(requests.post, 
                                     url,
                                     filename, 
                                     req_method)

        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._request_handler(callable_raising_requests_error, 
                                     url,
                                     filename, 
                                     req_method)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
