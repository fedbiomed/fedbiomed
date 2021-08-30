import unittest
import os
import sys
import importlib

class TestEnvironNode(unittest.TestCase):
    '''
    Test the node environment class
    '''
    # before the tests
    def setUp(self):
        self.config_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../etc'))
        
        # test config file names
        self.files = [
            'config_node.ini',      # default file name
            'config_bis.ini',       # alternate file names
            'config_ter',
            'node2_name',
            self.config_dir + 'config_node.ini',
            self.config_dir + 'config_4.ini',
            self.config_dir + 'config_5',
            self.config_dir + 'node3_name'
            ]          

        # test upload urls
        self.url_map = [
            [ '/arbitrary/path/', '/arbitrary/path/' ],
            [ '/uncomplete/path', '/uncomplete/path/' ]
        ]

        pass

    # after the tests
    def tearDown(self):
        pass

    #
    # TODO : add tests for other environment items
    #

    def test_file_creation(self):

        for config_file in self.files:

            if not os.path.isabs(config_file):
                config_path = os.path.join(self.config_dir, config_file)
            else:
                config_path = config_file
            # clean file if exists
            if os.path.isfile(config_path):
                os.remove(config_path)

            os.environ['CONFIG_FILE'] = str(config_file)
            # testing environment initialization done at module import
            if 'fedbiomed.node.environ' in sys.modules:
                # clean namespace before next test
                del sys.modules['fedbiomed.node.environ']
                # reloading is not enough
                #importlib.reload(fedbiomed.node.environ)
            import fedbiomed.node.environ

            # config file should now exist
            self.assertTrue(os.path.isfile(config_path))

            # clean file before ending test
            if(os.path.isfile(config_path)):
                os.remove(config_path)

    def test_uploads_url(self):

        config_path = os.path.join(self.config_dir, 'config_url')
        os.environ['CONFIG_FILE'] = config_path
        

        # test with unset upload url
        del os.environ['UPLOADS_URL']

        if os.path.isfile(config_path):
             os.remove(config_path)
        if 'fedbiomed.node.environ' in sys.modules:
            del sys.modules['fedbiomed.node.environ']
        import fedbiomed.node.environ
        self.assertEqual(fedbiomed.node.environ.UPLOADS_URL, 'http://localhost:8844/upload/')

        if 'fedbiomed.node.environ' in sys.modules:
            del sys.modules['fedbiomed.node.environ']
        import fedbiomed.node.environ
        self.assertEqual(fedbiomed.node.environ.UPLOADS_URL, 'http://localhost:8844/upload/')


        # test upload urls given directly
        for given_url, used_url in self.url_map:
            os.environ['UPLOADS_URL'] = given_url

            if os.path.isfile(config_path):
                os.remove(config_path)
            if 'fedbiomed.node.environ' in sys.modules:
                del sys.modules['fedbiomed.node.environ']
            import fedbiomed.node.environ
            self.assertEqual(used_url, fedbiomed.node.environ.UPLOADS_URL)

            # reload the same config from existing file (not from variable)
            del os.environ['UPLOADS_URL'] 

            if 'fedbiomed.node.environ' in sys.modules:
                del sys.modules['fedbiomed.node.environ']
            import fedbiomed.node.environ
            self.assertEqual(used_url, fedbiomed.node.environ.UPLOADS_URL)            

        # test upload url from IP
        uploads_ip = '1.2.3.4'
        os.environ['UPLOADS_IP'] = uploads_ip
        
        if os.path.isfile(config_path):
            os.remove(config_path)
        if 'fedbiomed.node.environ' in sys.modules:
            del sys.modules['fedbiomed.node.environ']
        import fedbiomed.node.environ
        self.assertEqual(fedbiomed.node.environ.UPLOADS_URL, 'http://' + uploads_ip + ':8844/upload/')

        # reload the config from existing file
        del os.environ['UPLOADS_IP'] 
        if 'fedbiomed.node.environ' in sys.modules:
            del sys.modules['fedbiomed.node.environ']
        import fedbiomed.node.environ
        self.assertEqual(fedbiomed.node.environ.UPLOADS_URL, 'http://' + uploads_ip + ':8844/upload/')

        os.remove(config_path)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
