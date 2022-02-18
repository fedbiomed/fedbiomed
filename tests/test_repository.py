import os
import requests
import builtins
from json import JSONDecodeError
import unittest
from unittest import mock

from unittest.mock import MagicMock, patch

from fedbiomed.common.repository import Repository
from fedbiomed.common.exceptions import FedbiomedRepositoryError


class FakeRequest:
    file_name = None
    content = "some content"
    def __init__(self,   *args, **kwargs  ):

        self.file_name = kwargs.get('files')
        self.status_code = 200
        self.request = MagicMock(return_value="some http requests")
        
    def raise_for_status(self):
        return None

    def json(self):
        return self.file_name
        
        
class TestRepository(unittest.TestCase):
    class OpenMock:
        file_name = None
        
        def write(self, content):
            self.content = content
    # before the tests
    def setUp(self):
        
        self.uploads_url='http://a.fake.url'
        # instanciating objects
        
        self.r1 = Repository(uploads_url=self.uploads_url,
                             tmp_dir='/my/temporary_folder/path',
                             cache_dir='/path/to/my/cache/dir')

        self.r2 = Repository(None, None, None)
        
    # after the tests
    def tearDown(self):
        pass
    
    @patch('fedbiomed.common.repository.Repository._raise_for_status_handler')
    @patch('fedbiomed.common.repository.Repository._connection_handler')
    @patch('requests.post')
    @patch('builtins.open')
    def test_reporistory_01_upload_file_normal_case(self, 
                                                    builtins_open_patch, 
                                                    requests_post_patch,
                                                    connection_handler_patch,
                                                    raise_for_status_handler_patch):
        """
        Tests `upload_file` method in the normal case scenario
        """

        # arguments
        fake_filename = 'my/file/to/upload'
        open_fake = TestRepository.OpenMock()

        def open_side_effect(file_name, mode):
            open_fake.file_name = file_name
            return open_fake         
        
        def connection_handler_side_effect(callable_method, url,
                                           filename, req_method, *args, **kwargs):
            fake_req = FakeRequest(files = kwargs.get('files'))
            setattr(fake_req, 'request', req_method)
            return fake_req    
        # patches & Mocking
        
        builtins_open_patch.side_effect = open_side_effect
        connection_handler_patch.side_effect = connection_handler_side_effect
        raise_for_status_handler_patch.return_value = None
        # action
        
        res = self.r1.upload_file(fake_filename)
        print(res)
        # checks
        # check correct calls
        builtins_open_patch.assert_called_once_with(fake_filename, 'rb')
        connection_handler_patch.assert_called_once_with(requests_post_patch,
                                                         self.uploads_url,
                                                         fake_filename,
                                                         'POST',
                                                         files={'file': open_fake})
        
        # check result of request
        self.assertEqual(res, {'file': open_fake})
        self.assertEqual(fake_filename, open_fake.file_name)
    
    
    def test_repository_02_upload_file_open_exceptions(self):
        """
        Tests `upload_file` method, under the case where builtins open method triggeres exception.
        It could raises 3 kind of exceptions:
        1. a FileNotFoundError due to a file not found on system
        2. a PermissionError due to unsatisfactory privileges
        3. a OSError due to the unability to read specified file
        
        when those errors are triggered, `upload_file` must trigger a FedBimedRepositoryError
        with the corresponding message wrt the occured error
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
    @patch('fedbiomed.common.repository.Repository._connection_handler')
    @patch('builtins.open')       
    def test_repository_03_upload_file_json_deserialize_exception(self,
                                                                  builtins_open_patch, 
                                                                  connection_handler_patch,
                                                                  raise_for_status_handler_patch):
        """
        Checks if JSON DecodeError is handled correclty when trying to deserialize message.
        """                                                  
        # patches and mocks
        builtins_open_patch.return_value = None
        requests_post = MagicMock(return_value=None)
        requests_post.method = MagicMock(return_value=None)
        
        requests_post.json = MagicMock(side_effect=JSONDecodeError("mimicking a eception occuring when a JSON message"
                                                                   "is not deserializable", doc='a_doc', pos=22))
        connection_handler_patch.return_value = requests_post
        raise_for_status_handler_patch.return_value = None
        # action & checks
        
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.upload_file('a/file/to/upload')
    
    @patch('builtins.open')
    @patch('fedbiomed.common.repository.Repository._raise_for_status_handler')
    @patch('fedbiomed.common.repository.Repository._connection_handler')
    @patch('requests.get')
    def test_reporistory_04_download_file_normal_case(self,
                                                      requests_get_patch,
                                                      connection_handler_patch,
                                                      raise_for_status_handler_patch,
                                                      open_patch):
        
        open_fake = TestRepository.OpenMock()
        def open_side_effect(file_name, mode):
            open_fake.file_name = file_name
            return open_fake  
        
        # arguments
        url = 'http://a.url.from.which.to?download#file'
        path_file = '/a/path/to/a/file/on/which/downloaded/content/will/be/saved'
        # Patches & Mocks
        connection_handler_patch.side_effect = FakeRequest
        raise_for_status_handler_patch.return_value = None
        open_patch.side_effect = open_side_effect
        
        # action
        status_code, filepath = self.r1.download_file(url,
                                                      path_file)
        
        # checks
        connection_handler_patch.assert_called_once_with(requests_get_patch, 
                                                         url,
                                                         path_file,
                                                         'GET')
        raise_for_status_handler_patch.assert_called_once()
        expected_path_file = os.path.join(self.r1.tmp_dir, path_file)
        open_patch.assert_called_once_with(expected_path_file,
                                           'wb')
        
        self.assertEqual(filepath, expected_path_file)
        self.assertEqual(status_code, 200)
        
    @patch('builtins.open')
    @patch('fedbiomed.common.repository.Repository._raise_for_status_handler')
    @patch('fedbiomed.common.repository.Repository._connection_handler')
    def test_repository_05_download_file_open_exceptions(self, 
                                                         connection_handler_patch, 
                                                         raise_for_status_patch,
                                                         open_patch):
        
        # arguments
        url = 'http://a.url.from.which.to?download#file'
        path_file = '/a/path/to/a/file/on/which/downloaded/content/will/be/saved'
        # patches and mocks
        connection_handler_patch.return_value = None
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
        open_mock.write = MagicMock(side_effect=MemoryError("mimicking situation where there is no available"
                                                            " space on system disk"))
        #open_mock.write = 
        connection_handler_patch.return_value = FakeRequest()
        open_patch.side_effect = None
        open_patch.return_value = open_mock
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.download_file(url, path_file)
        
        # check OSError
        open_mock.write = MagicMock(side_effect=OSError("mimicking situation where file cannot be read"
                                                        " (eg used by another process)"))
        open_patch.return_value = open_mock
        
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.download_file(url, path_file)
    
    @patch('fedbiomed.common.repository.Repository._get_method_request')
    def test_repository_06_private_raise_for_status_handler_normal_case(self, 
                                                                        get_method_req_patch):
        
        req = FakeRequest()
        get_method_req_patch.return_value = "issuing some HTTP requests"
        
        # action & checks
        with patch.object(FakeRequest, 'raise_for_status') as raise_for_status_mock:
            self.r1._raise_for_status_handler(req, '/a/file/on/my/computer')
            # we are checking here if `raise_for_status` has been called
            raise_for_status_mock.assert_called_once()

    @patch('fedbiomed.common.repository.Repository._get_method_request')       
    def test_repository_07_private_raise_for_status_exceptions(self,
                                                               get_method_req_patch):
        # run a first test that triggers HTTPError with status code error 404
        # patches and mocks
        
        requests_mock = MagicMock(return_value=None)
        requests_mock.raise_for_status = MagicMock(side_effect=requests.HTTPError("mimicking an HTTP error"))
        requests_mock.status_code = 404  
        
        get_method_req_patch.return_value = "a HTTP request"
        
        # action & checks
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._raise_for_status_handler(requests_mock,
                                              'my/file/to/upload')
            
        # run a second test that triggers HTTPError with status code error 500
        # patches and mocks
        requests_mock.status_code = 500
        # action & checks
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._raise_for_status_handler(requests_mock, 'my/other/file/to/upload')
        
    def test_repository_08_get_method_request(self):
        # test 1 with a HTTP POST request
        
        msg_test_1 = self.r1._get_method_request("post")  
        self.assertEqual(msg_test_1, "uploading file")  
        
        # test 2 with a HTTP GET request
        
        msg_test_2 = self.r1._get_method_request("get")
        self.assertEqual(msg_test_2, 'downloading file')
        
        # test 3 with an unknown HTTP request
        
        msg_test_3 = self.r1._get_method_request('unknown request')
        self.assertEqual('issuing unknown HTTP request', msg_test_3)

    @patch('fedbiomed.common.repository.Repository._get_method_request')
    def test_repository_09_private_connection_handler_normal_case(self, 
                                                                  get_method_req_patch):
        def a_callable(url):
            return FakeRequest(url)

        get_method_req_patch.return_value = "do some operation on file"
        
        # action
        res = self.r1._connection_handler(a_callable, 
                                          'http://a.fake.url',
                                          'a/file/path',
                                          req_method="unknown_method")
        
        # checks
        self.assertIsInstance(res, FakeRequest)
        
                 
    def test_repository_10_private_connection_handler_exceptions(self):
        """
        Checks that errors tirggered through `requests.post` (ie when issuing a HTTP POST Request)
        have been handled accordingly in `upload_file` method
        Errors triggered during runtime executions are :
        1. requests.Timeout
        2. requests.TooManyRedirects
        3. requests.connectionError
        4. requests.InvalidSchema
        5. requests.RequestException
        """
        # arguments
        url = 'http://a.fake.url'
        filename = 'a/file/path'
        req_method = "unknown_method"

        # defining closures
        def callable_raising_timeout_exception(*args, **kwargs):
            raise requests.Timeout("Mimicking a Timeout error due to a connection that"
                                   " exceeded timeout")
            
        def callable_raising_too_many_redirection_errors(*args, **kwrags):
            raise requests.TooManyRedirects("Mimicking a TooManyRedirectsError due to reaching"
                                            " `max_redirection` threshold")
        
        def callable_raising_requests_error(*args, **kwargs):
            raise requests.RequestException("Mimicking a unknown RequestException")
        
        # tests
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._connection_handler(callable_raising_timeout_exception,
                                        url,
                                        filename,
                                        req_method)
        
            
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._connection_handler(callable_raising_too_many_redirection_errors, 
                                        url,
                                        filename, 
                                        req_method)
        
        with self.assertRaises(FedbiomedRepositoryError):
            # should return a requests.InvalidSchema error
            self.r2._connection_handler(requests.post, 
                                        'a/file/to/upload',
                                        filename, 
                                        req_method)
            
        with self.assertRaises(FedbiomedRepositoryError):
            # should return a connectionError due to unknown server
            self.r1._connection_handler(requests.post, 
                                        url,
                                        filename, 
                                        req_method)
            
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1._connection_handler(callable_raising_requests_error, 
                                        url,
                                        filename, 
                                        req_method)
