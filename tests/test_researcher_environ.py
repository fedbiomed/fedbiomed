import os
import importlib
import unittest
import inspect
import tempfile
import configparser
from unittest.mock import patch, MagicMock

from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedEnvironError
from testsupport.fake_common_environ import Environ


class TestResearcherEnviron(unittest.TestCase):

    def setUp(self) -> None:
        """Setup test for each test function"""

        self.config_mock = MagicMock()

        self.patch_open = patch('builtins.open')
        self.patch_open.start()

        self.config_mock.return_value.get.side_effect = [
            'db.json', 'True',  # Common
            'researcher-id', 'localhost', '50051', 'pir-key', 'pub-key']  # Node

        environ_module_dir = os.path.join(os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ), "..", "fedbiomed", "researcher", "environ.py")

        self.tem = tempfile.TemporaryDirectory()
        os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = self.tem.name
        self.env = importlib.machinery.SourceFileLoader(
            "environ_for_test_node", environ_module_dir)\
            .load_module()
        del os.environ["FBM_RESEARCHER_COMPONENT_ROOT"]

        ResearcherEnviron = self.env.ResearcherEnviron

        ## Reset
        self.config_mock.return_value.get.side_effect = None
        self.config_mock.return_value.get.side_effect = [
            '../var/db.json', 'True',  # Common
            'researcher-id', 'localhost', '50051', 'pir-key', 'pub-key']  # Node

        if ResearcherEnviron in ResearcherEnviron._objects:
            del ResearcherEnviron._objects[ResearcherEnviron]

        self.tem = tempfile.TemporaryDirectory()
        self.environ = ResearcherEnviron(root_dir=self.tem.name)

    def tearDown(self) -> None:
        self.patch_open.stop()
        self.tem.cleanup()

    def test_01_researcher_environ_set_component_specific_variables(self):
        """Tests setting variables for researcher environ"""

        self.environ.set_environment()

        self.assertEqual(self.environ._values["EXPERIMENTS_DIR"],
                         os.path.join(self.tem.name, "var", "experiments"))
        self.assertEqual(self.environ._values["TENSORBOARD_RESULTS_DIR"],
                         os.path.join(self.tem.name, "runs"))
        self.assertEqual(self.environ._values["MESSAGES_QUEUE_DIR"],
                         os.path.join(self.tem.name, "var", "queue_messages"))
        self.assertEqual(self.environ._values["DB_PATH"],
                         os.path.join(self.tem.name, "var", f"db_{self.environ['ID']}.json"))

        self.assertEqual(self.environ._values["SERVER_HOST"], "localhost")
        self.assertEqual(self.environ._values["SERVER_PORT"], "50051")


    @patch("fedbiomed.common.logger.logger.info")
    def test_05_researcher_environ_info(self, mock_logger_info):

        self.environ._values["COMPONENT_TYPE"] = ComponentType.RESEARCHER
        self.environ.info()
        self.assertEqual(mock_logger_info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
