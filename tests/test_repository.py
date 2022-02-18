import builtins
from json import JSONDecodeError
import unittest
import requests
from unittest.mock import MagicMock, patch

from fedbiomed.common.repository import Repository
from fedbiomed.common.exceptions import FedbiomedRepositoryError


class FakeRequest:
        file_name = None
        def __init__(self, url, files):

            self.file_name = files
            self.status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return self.file_name
        
class TestRepository(unittest.TestCase):
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
    
    
    @patch('requests.post')
    @patch('builtins.open')
    def test_reporistory_01_upload_file_normal_case(self, 
                                                    builtins_open_patch, 
                                                    requests_post_patch):
        """
        Tests `upload_file` method in the normal case scenario
        """
        class OpenMock:
            file_name = None
        
        # arguments
        fake_filename = 'my/file/to/upload'
        
        open_fake = OpenMock()

        def side_effect_open(file_name, mode):
            open_fake.file_name = file_name
            return open_fake         
        
        # patches & Mocking
        
        builtins_open_patch.side_effect = side_effect_open
        requests_post_patch.side_effect = FakeRequest
        
        # action
        
        res = self.r1.upload_file(fake_filename)
        print(res)
        # checks
        # check correct calls
        builtins_open_patch.assert_called_once_with(fake_filename, 'rb')
        requests_post_patch.assert_called_once_with(self.uploads_url,
                                                    files={'file': open_fake})
        
        # check result of request
        self.assertEqual(res, {'file': open_fake})
    
    
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

    @patch('builtins.open')           
    def test_repository_03_upload_file_post_http_request_exception(self,
                                                                   builtins_open_patch):
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
        builtins_open_patch.return_value = None

        with patch.object(requests, 'post') as req_post_mock:
            
            req_post_mock.side_effect = requests.Timeout("Mimicking a Timeout error due to a connection that"
                                                         " exceeded timeout")
            with self.assertRaises(FedbiomedRepositoryError):
                self.r1.upload_file('a/file/to/upload')
            
            req_post_mock.side_effect = requests.TooManyRedirects("Mimicking a TooManyRedirectsError due to reaching"
                                                                  " `max_redirection` threshold")
            with self.assertRaises(FedbiomedRepositoryError):
                self.r1.upload_file('a/file/to/upload')
        
        with self.assertRaises(FedbiomedRepositoryError):
            # should return a connectionError due to unknown server
            self.r1.upload_file('a/file/to/upload')  
            
        with self.assertRaises(FedbiomedRepositoryError):
            # shoudlf return a requests.InvalidSchema error
            self.r2.upload_file('a/file/to/upload')
    
        with patch.object(requests, 'post') as req_post_mock:
            req_post_mock.side_effect = requests.RequestException("Mimicking a unknown RequestError")
            with self.assertRaises(FedbiomedRepositoryError):
                self.r1.upload_file('a/file/to/upload')

    @patch('requests.post')
    @patch('builtins.open')
    def test_repository_04_upload_file_http_exceptions(self,
                                                       builtins_open_patch, 
                                                       requests_post_patch):
        """
        Runs 2 tests that trigger HTTPError exception when calling `requests.post.raise_for_status`
        and check how it is handled
        - 1: raises an HTTPError with status code error = 404
        - 2: raises an HTTPError with status code error = 500
        """
        
        # run a first test that triggers HTTPError with status code error 404
        # patches and mocks
        builtins_open_patch.return_value = None
        requests_post = MagicMock(return_value=None)
        requests_post.raise_for_status = MagicMock(side_effect=requests.HTTPError("mimicking an HTTP error"))
        requests_post.status_code = 404  
        requests_post_patch.return_value = requests_post
        
        # action & checks
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.upload_file('my/file/to/upload')
            
        # run a second test that triggers HTTPError with status code error 500
        # patches and mocks
        requests_post.status_code = 505
        # action & checks
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.upload_file('my/file/to/upload')
     
    @patch('requests.post')
    @patch('builtins.open')       
    def test_repository_05_upload_file_json_deserialize_exception(self,
                                                                  builtins_open_patch, 
                                                                  requests_post_patch):
                                                                  
        # patches and mocks
        builtins_open_patch.return_value = None
        requests_post = MagicMock(return_value=None)
        requests_post.raise_for_status = MagicMock(return_value=None)
        
        requests_post.json = MagicMock(side_effect=JSONDecodeError("mimicking a eception occuring when a JSON message"
                                                                   "is not deserializable", doc='a_doc', pos=22))
        requests_post_patch.return_value = requests_post
        
        # action & checks
        
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.upload_file('a/file/to/upload')
