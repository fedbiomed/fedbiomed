# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

import unittest
import os
import sys
import tempfile
import shutil

# dont import fedbiomed.*.environ here

class TestEnvironCommon(unittest.TestCase):
    '''
    Test the common configs for node and researcher environment class
    '''
    # before the tests
    def setUp(self):

        self.envs = [ 'fedbiomed.node.environ', 'fedbiomed.researcher.environ' ]

        # need a temp config dir inside ./etc to be able to test relative path CONFIG_FILE
        self.config_subdir = next(tempfile._get_candidate_names())
        self.config_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../etc')), self.config_subdir)
        # needed if etc does not exists (fresh install)
        os.makedirs(self.config_dir)

        # test config file names
        self.files = [
            # Cannot test with relative
            'config_node.ini',
            'config_bis.ini',
            'config_ter',
            'node2_name',
            os.path.join(self.config_dir, 'config_node.ini'),
            os.path.join(self.config_dir, 'config_4.ini'),
            os.path.join(self.config_dir, 'config_5'),
            os.path.join(self.config_dir, 'node3_name')
            ]

        # test upload urls
        self.url_map = [
            [
                os.path.join(os.path.sep,'arbitrary','path', ''), # /arbitrary/path/ on Linux
                os.path.join(os.path.sep,'arbitrary','path', '')  # /arbitrary/path/ on Linux
            ],
            [
                os.path.join(os.path.sep,'uncomplete','path'),    # /uncomplete/path on Linux
                os.path.join(os.path.sep,'uncomplete','path', '') # /uncomplete/path/ on Linux
            ]
        ]

        # test broker ip and names
        self.ip_map = [ 'foo', '1.2.3.4', 'my.full.name', '1.2.3.4.5.6' ]


    # after the tests
    def tearDown(self):
        shutil.rmtree(self.config_dir)

    def reload_environ(self, module: str):
        #
        # as environ is a singleton, we must reinforce
        # the loading of all modules
        #
        for m in ['fedbiomed.common.environ',
                  'fedbiomed.node.environ',
                  'fedbiomed.researcher.environ',
                  ]:
            if m in sys.modules:
                # clean namespace before next test
                del sys.modules[m]

        # we should also delete the global variable
        if 'environ' in globals():
            del globals()['environ']

        # and reimport the requested module
        exec('from ' + module + ' import environ', globals())


    #
    # TODO : add tests for other environment items
    #

    def test_file_creation(self):
        for env in self.envs:
            for config_file in self.files:

                if not os.path.isabs(config_file):
                    config_path = os.path.join(self.config_dir, config_file)
                    config_var = os.path.join(self.config_subdir, config_file)
                else:
                    config_path = config_file
                    config_var = config_file
                # clean file if exists
                if os.path.isfile(config_path):
                    os.remove(config_path)

                os.environ['CONFIG_FILE'] = str(config_var)
                # environment initialization done at module import
                self.reload_environ(env)

                # config file should now exist
                self.assertTrue(os.path.isfile(config_path))

                # clean file before ending test
                if(os.path.isfile(config_path)):
                    os.remove(config_path)

    def test_uploads_url(self):

        config_path = os.path.join(self.config_dir, 'config_url')
        os.environ['CONFIG_FILE'] = config_path

        for env in self.envs:
            # test with unset upload url
            if 'UPLOADS_URL' in os.environ:
                del os.environ['UPLOADS_URL']

            if os.path.isfile(config_path):
                 os.remove(config_path)
            self.reload_environ(env)
            self.assertEqual(environ['UPLOADS_URL'], 'http://localhost:8844/upload/')

            self.reload_environ(env)
            self.assertEqual(environ['UPLOADS_URL'], 'http://localhost:8844/upload/')


            # test upload urls given directly
            for given_url, used_url in self.url_map:
                os.environ['UPLOADS_URL'] = given_url

                if os.path.isfile(config_path):
                    os.remove(config_path)
                self.reload_environ(env)
                self.assertEqual(used_url, environ['UPLOADS_URL'])

                # reload the same config from existing file (not from variable)
                del os.environ['UPLOADS_URL']

                self.reload_environ(env)
                self.assertEqual(used_url, environ['UPLOADS_URL'])

            # test upload url from IP
            uploads_ip = '1.2.3.4'
            os.environ['UPLOADS_IP'] = uploads_ip

            if os.path.isfile(config_path):
                os.remove(config_path)
            self.reload_environ(env)
            self.assertEqual(environ['UPLOADS_URL'], 'http://' + uploads_ip + ':8844/upload/')

            # reload the config from existing file
            del os.environ['UPLOADS_IP']
            self.reload_environ(env)
            self.assertEqual(environ['UPLOADS_URL'], 'http://' + uploads_ip + ':8844/upload/')

            os.remove(config_path)

    def test_broker(self):

        #import pdb; pdb.set_trace()

        config_path = os.path.join(self.config_dir, 'config_broker')
        os.environ['CONFIG_FILE'] = config_path

        for env in self.envs:
            # test with unset broker ip
            if 'MQTT_BROKER' in os.environ:
                del os.environ['MQTT_BROKER']

            if os.path.isfile(config_path):
                 os.remove(config_path)
            self.reload_environ(env)
            self.assertEqual(environ['MQTT_BROKER'], 'localhost')

            self.reload_environ(env)
            self.assertEqual(environ['MQTT_BROKER'], 'localhost')


            # test upload urls given directly
            for ip in self.ip_map:
                os.environ['MQTT_BROKER'] = ip

                if os.path.isfile(config_path):
                    os.remove(config_path)
                self.reload_environ(env)

                self.assertEqual(ip, environ['MQTT_BROKER'])

                # reload the same config from existing file (not from variable)
                del os.environ['MQTT_BROKER']

                self.reload_environ(env)
                self.assertEqual(ip, environ['MQTT_BROKER'])

            if os.path.isfile(config_path):
                os.remove(config_path)



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
