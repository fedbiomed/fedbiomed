import os
import importlib
import inspect
import unittest
import configparser
import tempfile

from unittest.mock import patch, MagicMock

from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedEnvironError
from testsupport.fake_common_environ import Environ


class TestNodeEnviron(unittest.TestCase):

    def setUp(self) -> None:
        """Setup test for each test function"""


        self.config_mock = MagicMock()
        self.patch_open = patch('builtins.open')
        self.patch_open.start()

        self.config_mock.get.side_effect = [
            'db.json', 'True', # Common
            'node-id', 'True', 'True', "SHA256", '', '', 'c.key', 'c.pem', "localhost", "50051"]  # Node

        environ_module_dir = os.path.join(os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        ), "..", "fedbiomed", "node", "environ.py")

        self.tem = tempfile.TemporaryDirectory()
        os.environ["FBM_NODE_COMPONENT_ROOT"] = self.tem.name
        self.env = importlib.machinery.SourceFileLoader(
            "environ_for_test", environ_module_dir)\
            .load_module()
        self.tem.cleanup()
        del os.environ["FBM_NODE_COMPONENT_ROOT"]
        NodeEnviron = self.env.NodeEnviron

        ## Reset
        self.config_mock.return_value.get.side_effect = None
        self.config_mock.return_value.get.side_effect = [
            'db.json', 'True',  # Common
            'node-id', 'True', 'True', "SHA256", '', '', 'c.key', 'c.pem' "localhost", "50051"]  # Node
        if NodeEnviron in NodeEnviron._objects:
            del NodeEnviron._objects[NodeEnviron]

        self.tem = tempfile.TemporaryDirectory()
        self.environ = NodeEnviron(root_dir=self.tem.name, autoset=False)


    def tearDown(self) -> None:
        self.patch_open.stop()
        self.tem.cleanup()


    def test_01_node_environ_set_environment(self):


        os.environ["ALLOW_DEFAULT_TRAINING_PLANS"] = "True"
        os.environ["ENABLE_TRAINING_PLAN_APPROVAL"] = "True"

        self.config_mock.return_value.get.side_effect = [
            'db.json', 'True',  # Common
            'node-1', None, None, "SHA256", '', '', 'c.pem', 'c.key', "localhost", "50051"]  # Node
        self.environ.set_environment()

        self.assertEqual(self.environ._values["MESSAGES_QUEUE_DIR"],
                         os.path.join(self.tem.name, "var", "queue_manager"))
        self.assertEqual(
            self.environ._values["DB_PATH"],
            os.path.join(self.tem.name, "var", f"db_{self.environ['ID']}.json"))
        self.assertEqual(
            self.environ._values["TRAINING_PLANS_DIR"],
            os.path.join(self.tem.name,"var", "training_plans")
        )

        with patch("os.mkdir") as m:

            self.environ._config = self.config_mock
            self.config_mock.get.side_effect = None
            self.config_mock.get.side_effect = [
                'db.json', 'True', # Common
                'node-1', None, None, "SHA256BLABLA", '', '', 'c.pem', 'c.key', "localhost", "50051"]

            with self.assertRaises(FedbiomedEnvironError):
                self.environ.set_environment()

            self.config_mock.get.side_effect = None
            self.config_mock.get.side_effect = [
                'db.json', 'True',  # Common
                'node-1', False, False, "SHA256", '', '', 'c.pem', 'c.key', "localhost", "50051"]
            os.environ["ALLOW_DEFAULT_TRAINING_PLANS"] = "True"
            os.environ["ENABLE_TRAINING_PLAN_APPROVAL"] = "True"
            self.environ.set_environment()

            self.assertTrue(self.environ._values['ALLOW_DEFAULT_TRAINING_PLANS'], "os.getenv did not overwrite the value")
            self.assertTrue(self.environ._values['TRAINING_PLAN_APPROVAL'], "os.getenv did not overwrite the value")

            os.environ.pop('RESEARCHER_SERVER_HOST', None)
            os.environ.pop('RESEARCHER_SERVER_PORT', None)
            self.config_mock.sections.return_value = ['researcher']
            self.config_mock.get.side_effect = None
            self.config_mock.get.side_effect = [
                'db.json', 'True',  # Common
                'node-1', False, False, "SHA256", 't', 't', "c.pem", 'c.key', "50051", "localhost"]
            self.environ.set_environment()
            self.assertEqual(self.environ._values["RESEARCHERS"][0]["ip"], "localhost")
            self.assertEqual(self.environ._values["RESEARCHERS"][0]["port"], "50051")

            self.config_mock.side_effect = None
            self.config_mock.get.side_effect = [
                'db.json', 'True',  # Common
                'node-1', False, False, "SHA256", 't', 't', "c.pem", 'c.key', None, None]
            os.environ["RESEARCHER_SERVER_HOST"] = "localhost"
            os.environ["RESEARCHER_SERVER_PORT"] = "50051"
            self.environ.set_environment()
            self.assertEqual(self.environ._values["RESEARCHERS"][0]["ip"], "localhost", "os.getenv did not overwrite the value")
            self.assertEqual(self.environ._values["RESEARCHERS"][0]["port"], "50051", "os.getenv did not overwrite the value")

    @patch("fedbiomed.common.logger.logger.info")
    @patch("os.mkdir")
    def test_05_node_environ_info(self, mock_mkdir, mock_logger_info):
        os.environ["NODE_ID"] = "node-1"
        os.environ["ALLOW_DEFAULT_TRAINING_PLANS"] = "True"
        os.environ["ENABLE_TRAINING_PLAN_APPROVAL"] = "True"

        # self.config_mock.return_value.get.side_effect= [None, None, None, "SHA256", "False", '', '', "50051", "localhost"]
        # self.environ.()
        self.environ.set_environment()
        self.environ.info()
        self.assertEqual(mock_logger_info.call_count, 3)


if __name__ == "__main__":
    unittest.main()
