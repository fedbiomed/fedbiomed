import shutil
import unittest
import os

import tempfile
import fedbiomed.common.mpc_controller

from unittest.mock import patch
from fedbiomed.common.mpc_controller import MPCController
from fedbiomed.common.exceptions import FedbiomedMPCControllerError
from fedbiomed.common.constants import ComponentType


class TestMPCController(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.mpc_controller = MPCController(
            tmp_dir=self.tmp_dir,
            component_type=ComponentType.NODE,
            component_id="node-1",
        )
        pass

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)
        pass

    def test_mpc_controller_01_init_getters(self):
        """Test instantiation and getters"""

        self.assertTrue(os.path.isdir(self.mpc_controller.mpc_data_dir))
        self.assertTrue(os.path.isdir(self.mpc_controller.tmp_dir))

    def test_mpc_controller_02_exec(self):
        """Tests exec method to execute commands"""

        # Failing wrong typed option in command
        with self.assertRaises(FedbiomedMPCControllerError):
            self.mpc_controller._exec(["shamir-server-key", 12, "wrong-value"])

        with patch.object(fedbiomed.common.mpc_controller, "subprocess") as mock_process:
            mock_process.Popen.return_value.returncode = 0
            mock_process.Popen.return_value.communicate.return_value = "Opps", False
            status = self.mpc_controller._exec(["shamir-server-key"])
            self.assertTrue(status)

    def test_mpc_controller_02_exec_shamir(self):
        """Tests shamir protocol"""

        # Test invalid num parties type
        with self.assertRaises(FedbiomedMPCControllerError):
            self.mpc_controller.exec_shamir(party_number=0,
                                            num_parties="0",
                                            ip_addresses="dummy/path/to/files")

        # Test invalid num parties type
        with self.assertRaises(FedbiomedMPCControllerError):
            self.mpc_controller.exec_shamir(party_number=0,
                                            num_parties=2,
                                            ip_addresses="dummy/path/to/files")

        # Test invalid num parties type
        with patch.object(fedbiomed.common.mpc_controller, "subprocess") as mock_process:
            mock_process.Popen.return_value.returncode = 0
            mock_process.Popen.return_value.communicate.return_value = "Opps", True
            result = self.mpc_controller.exec_shamir(party_number=0,
                                                     num_parties=3,
                                                     ip_addresses="dummy/path/to/files")
            self.assertTrue("Output-P0-0" in result)

            result = self.mpc_controller.exec_shamir(party_number=1,
                                                     num_parties=3,
                                                     ip_addresses="dummy/path/to/files")
            self.assertTrue("Input-P1-0" in result)

            mock_process.Popen.return_value.returncode = 1
            with self.assertRaises(FedbiomedMPCControllerError):
                self.mpc_controller.exec_shamir(party_number=1,
                                                num_parties=3,
                                                ip_addresses="dummy/path/to/files")

    def test_mpc_controller_03_init_errors(self):
        """Test instantiation errors"""

        with self.assertRaises(FedbiomedMPCControllerError):
            MPCController(
                tmp_dir='/not/a/path',
                component_type=ComponentType.NODE,
                component_id="node-1",
            )

        with patch('fedbiomed.common.mpc_controller.get_fedbiomed_root') as patch_gfr:
            patch_gfr.return_value = '/not/valid/path'
            with self.assertRaises(FedbiomedMPCControllerError):
                MPCController(
                    tmp_dir=self.tmp_dir,
                    component_type=ComponentType.NODE,
                    component_id="node-1",
                )

if __name__ == "__main__":
    unittest.main()
