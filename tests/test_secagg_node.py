import unittest
from unittest.mock import mock_open, patch
from copy import deepcopy

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from fedbiomed.common.exceptions import FedbiomedSecaggError, FedbiomedError
from fedbiomed.node.environ import environ
from fedbiomed.node.secagg import SecaggServkeySetup, SecaggBiprimeSetup, SecaggBaseSetup, SecaggSetup
import fedbiomed.node.secagg


class TestSecaggBaseSetup(NodeTestCase):

    def setUp(self) -> None:
        self.abstract_methods_patcher = patch.multiple(SecaggBaseSetup, __abstractmethods__=set())
        self.abstract_methods_patcher.start()

        self.args = {
                'researcher_id': 'my researcher',
                'secagg_id': "my secagg",
                'experiment_id': '123345',
                'parties': ['my researcher', 'my node1', 'my node2', 'my node3'],
        }

        self.base_secagg_setup = SecaggBaseSetup(**self.args)

    def tearDown(self) -> None:
        self.abstract_methods_patcher.stop()

    def test_base_secagg_setup_01_init_bad_args(self):
        """Tests bad init arguments """

        # Faulty typed researcher id
        args = deepcopy(self.args)
        args["researcher_id"] = None
        with self.assertRaises(FedbiomedSecaggError):
            SecaggBaseSetup(**args)

        # Invalid number of parties (must be at least 3 parties)
        args = deepcopy(self.args)
        args["parties"] = ["my researcher", "p2"]
        with self.assertRaises(FedbiomedSecaggError):
            SecaggBaseSetup(**args)

        # Unmatch self id and parties
        args = deepcopy(self.args)
        args["researcher_id"] = "opss different researcher"
        with self.assertRaises(FedbiomedSecaggError):
            SecaggBaseSetup(**args)

    def test_base_secagg_setup_02_getters(self):
        """Tests getters properties"""

        self.assertEqual(self.base_secagg_setup.researcher_id, self.args["researcher_id"])
        self.assertEqual(self.base_secagg_setup.secagg_id, self.args["secagg_id"])
        self.assertEqual(self.base_secagg_setup.experiment_id, self.args["experiment_id"])
        self.assertEqual(self.base_secagg_setup.element, None)

    def test_base_secagg_setup_03_create_secagg_reply(self):
        """Tests reply creation """

        reply = self.base_secagg_setup._create_secagg_reply(
            message="Test message",
            success=False
        )

        self.assertDictEqual(reply, {
                            'researcher_id': self.args["researcher_id"],
                            'secagg_id': self.args["secagg_id"],
                            'success': False,
                            'msg': "Test message",
                            'command': 'secagg'
        })


class SecaggTestCase(NodeTestCase):

    def setUp(self) -> None:
        self.patch_skm = patch.object(fedbiomed.node.secagg, "SKManager")
        self.patch_cm = patch.object(fedbiomed.node.secagg, "_CManager")
        self.patch_mpc = patch.object(fedbiomed.node.secagg, 'MPCController')
        self.patch_bpm = patch.object(fedbiomed.node.secagg, "BPrimeManager")

        self.mock_skm = self.patch_skm.start()
        self.mock_cm = self.patch_cm.start()
        self.mock_mpc = self.patch_mpc.start()
        self.mock_bpm = self.patch_bpm.start()

        # Set MOCK variables
        self.mock_cm.write_mpc_certificates_for_experiment.return_value = ('dummy/ip', [])
        self.mock_mpc.exec_shamir.return_value = 'dummy/path/to/output'
        unittest.mock.MagicMock.mpc_data_dir = unittest.mock.PropertyMock(
            return_value='dummy/path/to/output'
        )
        unittest.mock.MagicMock.tmp_dir = unittest.mock.PropertyMock(
            return_value=environ["TMP_DIR"]
        )

    def tearDown(self) -> None:
        self.patch_skm.stop()
        self.patch_cm.stop()
        self.patch_mpc.stop()
        self.patch_bpm.stop()


class TestSecaggServkey(SecaggTestCase):

    def setUp(self) -> None:

        super().setUp()
        self.args = {
            'researcher_id': "my researcher",
            'secagg_id': "my secagg",
            'experiment_id': 'my_experiment_id',
            'parties': ['my researcher', environ["ID"], 'my node2', 'my node3'],
        }
        self.secagg_servkey = SecaggServkeySetup(**self.args)

    def tearDown(self) -> None:
        super().tearDown()

    def test_secagg_servkey_setup_01_init(self):
        """Tests failing due to experiment id"""

        args = deepcopy(self.args)
        args["experiment_id"] = None
        with self.assertRaises(FedbiomedSecaggError):
            SecaggServkeySetup(**args)

        args["experiment_id"] = ''
        with self.assertRaises(FedbiomedSecaggError):
            SecaggServkeySetup(**args)

    def test_secagg_servkey_setup_02_setup_specific(self):
        """Test setup operation for servkey"""

        with patch("builtins.open") as mock_open:
            self.secagg_servkey._setup_specific()

            self.mock_cm.write_mpc_certificates_for_experiment.assert_called_once_with(
                path_certificates='dummy/path/to/output',
                path_ips=environ["TMP_DIR"],
                self_id=environ["ID"],
                self_ip=environ["MPSPDZ_IP"],
                self_port=environ["MPSPDZ_PORT"],
                self_private_key=environ["MPSPDZ_CERTIFICATE_KEY"],
                self_public_key=environ["MPSPDZ_CERTIFICATE_PEM"],
                parties=['my researcher', environ["ID"], 'my node2',
                         'my node3']
            )

            self.mock_mpc.exec_shamir.called_once_with(
                party_number=self.args["parties"].index(environ["ID"]),
                num_parties=len(self.args["parties"]),
                ip_addresses='dummy/ip'
            )

            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedSecaggError):
                self.secagg_servkey._setup_specific()

    def test_secagg_servkey_setup_03_setup(self):
        #import pdb; pdb.set_trace()
        shamir_key_share = '123245'
        with (patch('fedbiomed.node.secagg._CManager.write_mpc_certificates_for_experiment') as cm_patch,
              patch('fedbiomed.node.secagg.open', mock_open(read_data=shamir_key_share)) as builtin_open_mock):
            for e, m in zip((FedbiomedError, Exception,), (builtin_open_mock, self.mock_mpc.exec_shamir,)):
                builtin_open_mock.reset_mock()
                cm_patch.return_value = '/a/path/to/my/ips/certificate/files', None
                self.mock_skm.add.return_value = None
                builtin_open_mock.return_value = None
                self.mock_mpc.exec_shamir.return_value = '/a/path/to/my/key/share'
                m.side_effect = e  # setting different exception to mock
                
                reply = self.secagg_servkey.setup()
                self.assertFalse(reply['success'])
        
        # for get_value, return_value in (
        #     # Not tested by _matching_parties* 
        #     #
        #     # (3, False),
        #     # ({}, False),
        #     # ({'parties': None}, False),
        #     ({'parties': ['not', 'matching', 'current', 'parties']}, False),
        #     ({'parties': ['my researcher', environ["ID"], 'my node2', 'my node3']}, True),
        #     ({'parties': ['my researcher', environ["ID"], 'my node3', 'my node2']}, True),
        #     ({'parties': ['my researcher', environ["ID"], 'my node2', 'my node3', 'another']}, False),
        #     ({'parties': ['my node2', environ["ID"], 'my researcher', 'my node3']}, False),
        # ):
            
        with (patch('fedbiomed.node.secagg._CManager.write_mpc_certificates_for_experiment') as cm_patch,
              patch('fedbiomed.node.secagg.open', mock_open(read_data=shamir_key_share)) as builtin_open_mock):    
            # prefering to recreate mock than using mock.reset_mock() method
            cm_patch.return_value = '/a/path/to/my/ips/certificate/files', None
            self.mock_mpc.exec_shamir.return_value = '/a/path/to/my/key/share'

            self.mock_skm.add.return_value = None
            reply = self.secagg_servkey.setup()
            print(reply["msg"])
            self.assertEqual(reply["success"], True)
            self.assertIsInstance(reply["success"], bool)
            self.mock_skm.add.assert_called_once_with(self.args['secagg_id'],
                                                      self.args['parties'],
                                                      {'server_key': int(shamir_key_share)},
                                                      self.args['experiment_id'])

        with patch("fedbiomed.node.secagg.SecaggServkeySetup._setup_specific") as mock_:
            # FIXME: these are already tested...
            mock_.side_effect = Exception
            self.mock_skm.get.return_value = None
            reply = self.secagg_servkey.setup()
            self.assertEqual(reply["success"], False)

            mock_.side_effect = FedbiomedError
            self.mock_skm.get.return_value = None
            reply = self.secagg_servkey.setup()
            self.assertEqual(reply["success"], False)


class TestSecaggBiprime(SecaggTestCase):

    def setUp(self) -> None:

        super().setUp()
        self.args = {
            'researcher_id': "my researcher",
            'secagg_id': "my secagg",
            'experiment_id': None,
            'parties': ['my researcher', environ["ID"], 'my node2', 'my node3'],
        }
        self.secagg_bprime = SecaggBiprimeSetup(**self.args)

    def tearDown(self) -> None:
        super().tearDown()

    def test_secagg_biprime_setup_01_init(self):
        """Tests init with bad experiment_id"""
        args = deepcopy(self.args)
        args["experiment_id"] = "non-empty-string"

        with self.assertRaises(FedbiomedSecaggError):
            SecaggBiprimeSetup(**args)


    def test_secagg_biprime_setup_02_setup(self):
        """Tests init """

        # for get_value, return_value in (
        #     # Not tested by _matching_parties* 
        #     #
        #     # (3, False),
        #     # ({}, False),
        #     ({'parties': None}, True),
        #     ({'parties': ['not', 'matching', 'current', 'parties']}, False),
        #     ({'parties': ['my researcher', environ["ID"], 'my node2', 'my node3']}, True),
        #     ({'parties': ['my researcher', environ["ID"], 'my node3', 'my node2']}, True),
        #     ({'parties': ['my researcher', environ["ID"], 'my node2', 'my node3', 'another']}, True),
        #     ({'parties': ['my node2', environ["ID"], 'my researcher', 'my node3']}, True),
        # ):

        self.mock_bpm.is_default_biprime.return_value = True
        with patch('time.sleep'):
            reply = self.secagg_bprime.setup()
            self.assertEqual(reply["success"], True)

        self.mock_bpm.is_default_biprime.side_effect = FedbiomedError("error generaated for testing purposes")
        reply = self.secagg_bprime.setup()
        self.assertEqual(reply["success"], False)


class TestSecaggSetup(NodeTestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_secagg_setup_01_initialization(self):

        args = {
            "researcher_id": "r-1",
            "experiment_id": "experiment-id",
            "element": 0,
            "secagg_id": "secagg-id",
            "parties": ["r-1", "node-1", "node-2"]

        }

        # Test server key setup
        secagg_setup = SecaggSetup(**args)()
        self.assertIsInstance(secagg_setup, SecaggServkeySetup)

        # Test biprime setup
        args["element"] = 1
        del args["experiment_id"]
        secagg_setup = SecaggSetup(**args)()
        self.assertIsInstance(secagg_setup, SecaggBiprimeSetup)

        # Test forcing checking experiment_id None if Secagg setup is Biprime
        args["element"] = 1
        args["experiment_id"] = 12
        with self.assertRaises(FedbiomedSecaggError):
            secagg_setup = SecaggSetup(**args)()

        # Raise element type
        args["element"] = 2
        args["experiment_id"] = ""
        with self.assertRaises(FedbiomedSecaggError):
            SecaggSetup(**args)()

        # Raise element type
        args["element"] = 0
        args["experiment_id"] = 1234
        with self.assertRaises(FedbiomedSecaggError):
            SecaggSetup(**args)()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
