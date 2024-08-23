import os
import importlib
import unittest
import inspect
import configparser
from unittest.mock import patch

from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedEnvironError
from testsupport.fake_common_environ import Environ


class TestResearcherEnviron(unittest.TestCase):

    def setUp(self) -> None:
        """Setup test for each test function"""

        self.patch_config = patch('fedbiomed.researcher.config.ResearcherConfig')
        self.config_mock = self.patch_config.start()

        self.patch_mkdir = patch('os.mkdir')
        self.patch_open = patch('builtins.open')

        self.patch_mkdir.start()
        self.patch_open.start()

        self.config_mock.return_value.get.side_effect = [
            'db.json', 'mpspdz-localhost', 'port-14000', 'True', 'c.pem', 'c.key', 'True',  # Common
            'researcher-id', 'localhost', '50051', 'pir-key', 'pub-key']  # Node

        environ_module_dir = os.path.join(os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ), "..", "fedbiomed", "researcher", "environ.py")

        self.env = importlib.machinery.SourceFileLoader(
            "environ_for_test_node", environ_module_dir)\
            .load_module()

        ResearcherEnviron = self.env.ResearcherEnviron

        ## Reset
        self.config_mock.return_value.get.side_effect = None
        self.config_mock.return_value.get.side_effect = [
            '../var/db.json', 'mpspdz-localhost', 'port-14000', 'True', 'c.pem', 'c.key', 'True',  # Common
            'researcher-id', 'localhost', '50051', 'pir-key', 'pub-key']  # Node

        if ResearcherEnviron in ResearcherEnviron._objects:
            del ResearcherEnviron._objects[ResearcherEnviron]

        self.environ = ResearcherEnviron(root_dir='test')

    def tearDown(self) -> None:
        self.patch_mkdir.stop()
        self.patch_open.stop()
        self.patch_config.stop()

    def test_01_researcher_environ_set_component_specific_variables(self):
        """Tests setting variables for researcher environ"""

        self.config_mock.return_value.get.side_effect = [
            '../var/db_researcher-1.json', 'mpspdz-localhost', 'port-14000', 'True', 'c.pem', 'c.key', 'True',
            'researcher-1', 'localhost', '50051', 'pir-key', 'pub-key']

        self.environ.set_environment()

        self.assertEqual(self.environ._values["ID"], "researcher-1")
        self.assertEqual(self.environ._values["EXPERIMENTS_DIR"],
                         os.path.join("test/var", "experiments"))
        self.assertEqual(self.environ._values["TENSORBOARD_RESULTS_DIR"],
                         os.path.join("test", "runs"))
        self.assertEqual(self.environ._values["MESSAGES_QUEUE_DIR"],
                         os.path.join("test/var", "queue_messages"))
        self.assertEqual(self.environ._values["DB_PATH"],
                         os.path.join("test/var", "db_researcher-1.json"))
        
        self.assertEqual(self.environ._values["SERVER_HOST"], "localhost")
        self.assertEqual(self.environ._values["SERVER_PORT"], "50051")


    @patch("fedbiomed.common.logger.logger.info")
    def test_05_researcher_environ_info(self, mock_logger_info):

        self.environ._values["COMPONENT_TYPE"] = ComponentType.RESEARCHER
        self.environ.info()
        self.assertEqual(mock_logger_info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
