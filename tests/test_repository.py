# Managing NODE, RESEARCHER environ mock before running tests



import unittest
import requests
from unittest.mock import MagicMock, patch

from fedbiomed.common.repository import Repository
from fedbiomed.common.exceptions import FedbiomedRepositoryError
class FakeRequest:
        file_name = None
        status_code = 200
        def __init__(self, url, files):
            print('init')
            self.file_name = files
            self.status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            print("LOLO")
        
            return self.file_name
        
class TestRepository(unittest.TestCase):
    # before the tests
    def setUp(self):
        
        self.uploads_url='http://a.fake.url'
        # instanciating objects
        
        self.r1 = Repository(uploads_url=self.uploads_url,
                             tmp_dir='/my/temporary_folder/path',
                             cache_dir='/path/to/my/cache/dir')

    # after the tests
    def tearDown(self):
        pass
    
    
    @patch('requests.post')
    @patch('builtins.open')
    def test_reporistory_01_upload_file_normal_case(self, 
                                                    builtins_open_patch, 
                                                    requests_post_patch):
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
        #requests_post_patch.return_value = requests_mock
        #raise_for_status_patch.return_value = None
        
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
    
    @patch('requests.post')
    @patch('builtins.open')
    def test_repository_02_upoad_file_exceptions(self,
                                                builtins_open_patch, 
                                                requests_post_patch):
        
        builtins_open_patch.return_value = None
        requests_post = MagicMock(return_value=None)
        requests_post.raise_for_status = MagicMock(side_effect=requests.HTTPError("mimicking an HTTP error"))
        requests_post_patch.return_value = requests_post
        
        # action
        with self.assertRaises(FedbiomedRepositoryError):
            self.r1.upload_file('my/file/to/upload')