import unittest
from unittest.mock import MagicMock, mock_open, patch
from copy import deepcopy

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from fedbiomed.common.message import (
    AdditiveSSharingRequest,
    NodeToNodeMessages,
    AdditiveSSharingReply,
    ErrorMessage,
)
from fedbiomed.common.secagg import AdditiveShare
from fedbiomed.common.synchro import EventWaitExchange

from fedbiomed.transport.controller import GrpcController
from testsupport.base_case import NodeTestCase

#############################################################

from fedbiomed.common.exceptions import FedbiomedSecaggError, FedbiomedError
from fedbiomed.node.environ import environ
from fedbiomed.node.secagg import (
    SecaggDHSetup,
    SecaggKeySetup,
    SecaggServkeySetup,
    SecaggBaseSetup,
    SecaggSetup,
    SecaggKeySetup
)
import fedbiomed.node.secagg


class TestSecaggBaseSetup(NodeTestCase):

    def setUp(self) -> None:
        self.abstract_methods_patcher = patch.multiple(
            SecaggBaseSetup, __abstractmethods__=set()
        )
        self.abstract_methods_patcher.start()

        self.args = {
            "researcher_id": "my researcher",
            "secagg_id": "my secagg",
            "experiment_id": "123345",
            "parties": ["my researcher", "my node1", "my node2", "my node3"],
        }

        self.base_secagg_setup = SecaggBaseSetup(**self.args)

    def tearDown(self) -> None:
        self.abstract_methods_patcher.stop()

    def test_base_secagg_setup_01_init_bad_args(self):
        """Tests bad init arguments"""

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

    def test_base_secagg_setup_02_getters(self):
        """Tests getters properties"""

        self.assertEqual(
            self.base_secagg_setup.researcher_id, self.args["researcher_id"]
        )
        self.assertEqual(self.base_secagg_setup.secagg_id, self.args["secagg_id"])
        self.assertEqual(
            self.base_secagg_setup.experiment_id, self.args["experiment_id"]
        )
        self.assertEqual(self.base_secagg_setup.element, None)

    def test_base_secagg_setup_03_create_secagg_reply(self):
        """Tests reply creation"""

        reply = self.base_secagg_setup._create_secagg_reply(
            message="Test message", success=False
        )

        self.assertIsInstance(reply, ErrorMessage)


class SecaggTestCase(NodeTestCase):

    def setUp(self) -> None:
        self.patch_skm = patch.object(fedbiomed.node.secagg._secagg_setups, "SKManager")
        self.patch_cm = patch.object(fedbiomed.node.secagg._secagg_setups, "_CManager")
        self.patch_mpc = patch.object(
            fedbiomed.node.secagg._secagg_setups, "MPCController"
        )

        self.mock_skm = self.patch_skm.start()
        self.mock_cm = self.patch_cm.start()
        self.mock_mpc = self.patch_mpc.start()

        # Set MOCK variables
        self.mock_cm.write_mpc_certificates_for_experiment.return_value = (
            "dummy/ip",
            [],
        )
        self.mock_mpc.exec_shamir.return_value = "dummy/path/to/output"
        unittest.mock.MagicMock.mpc_data_dir = unittest.mock.PropertyMock(
            return_value="dummy/path/to/output"
        )
        unittest.mock.MagicMock.tmp_dir = unittest.mock.PropertyMock(
            return_value=environ["TMP_DIR"]
        )

    def tearDown(self) -> None:
        self.patch_skm.stop()
        self.patch_cm.stop()
        self.patch_mpc.stop()


class TestSecaggServkey(SecaggTestCase):

    def setUp(self) -> None:

        super().setUp()
        self.args = {
            "researcher_id": "my researcher",
            "secagg_id": "my secagg",
            "experiment_id": "my_experiment_id",
            "parties": ["my researcher", environ["ID"], "my node2", "my node3"],
        }
        self.secagg_servkey = SecaggServkeySetup(**self.args)

    def tearDown(self) -> None:
        super().tearDown()

    def test_secagg_servkey_setup_01_init(self):
        """Tests failing due to experiment id"""

        # Unmatch self id and parties
        args = deepcopy(self.args)
        args["researcher_id"] = "opss different researcher"
        with self.assertRaises(FedbiomedSecaggError):
            SecaggServkeySetup(**args)

    def test_secagg_servkey_setup_02_setup_specific(self):
        """Test setup operation for servkey"""

        with patch("builtins.open") as mock_open:

            mock_open.return_value.__enter__.return_value.read.side_effect = [
                "123123",
                '{"biprime": 12345}',
            ]
            self.secagg_servkey._setup_specific()
            self.mock_cm.write_mpc_certificates_for_experiment.assert_called_once_with(
                path_certificates="dummy/path/to/output",
                path_ips=environ["TMP_DIR"],
                self_id=environ["ID"],
                self_ip=environ["MPSPDZ_IP"],
                self_port=environ["MPSPDZ_PORT"],
                self_private_key=environ["MPSPDZ_CERTIFICATE_KEY"],
                self_public_key=environ["MPSPDZ_CERTIFICATE_PEM"],
                parties=["my researcher", environ["ID"], "my node2", "my node3"],
            )

            self.mock_mpc.exec_shamir.called_once_with(
                party_number=self.args["parties"].index(environ["ID"]),
                num_parties=len(self.args["parties"]),
                ip_addresses="dummy/ip",
            )

            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedSecaggError):
                self.secagg_servkey._setup_specific()

    def test_secagg_servkey_setup_03_setup(self):

        shamir_key_share = "123245"
        with (
            patch(
                "fedbiomed.node.secagg._secagg_setups._CManager.write_mpc_certificates_for_experiment"
            ) as cm_patch,
            patch(
                "fedbiomed.node.secagg._secagg_setups.open",
                mock_open(read_data=shamir_key_share),
            ) as builtin_open_mock,
        ):
            for e, m in zip(
                (
                    FedbiomedError,
                    Exception,
                ),
                (
                    builtin_open_mock,
                    self.mock_mpc.exec_shamir,
                ),
            ):
                builtin_open_mock.reset_mock()
                cm_patch.return_value = "/a/path/to/my/ips/certificate/files", None
                self.mock_skm.add.return_value = None
                builtin_open_mock.return_value = None
                self.mock_mpc.exec_shamir.return_value = "/a/path/to/my/key/share"
                m.side_effect = e  # setting different exception to mock

                reply = self.secagg_servkey.setup()
                self.assertIsInstance(reply, ErrorMessage)

        with (
            patch(
                "fedbiomed.node.secagg._secagg_setups._CManager.write_mpc_certificates_for_experiment"
            ) as cm_patch,
            patch(
                "fedbiomed.node.secagg._secagg_setups.open",
                mock_open(read_data=shamir_key_share),
            ) as builtin_open_mock,
            patch(
                "fedbiomed.node.secagg._secagg_setups.get_default_biprime"
            ) as gd_biprime,
        ):

            gd_biprime.return_value = "12345"
            cm_patch.return_value = "/a/path/to/my/ips/certificate/files", None
            self.mock_mpc.exec_shamir.return_value = "/a/path/to/my/key/share"

            self.mock_skm.add.return_value = None
            reply = self.secagg_servkey.setup()
            self.assertEqual(reply.success, True)
            self.assertIsInstance(reply.success, bool)
            self.mock_skm.add.assert_called_once_with(
                self.args["secagg_id"],
                self.args["parties"],
                {"server_key": int(shamir_key_share), "biprime": 12345},
                self.args["experiment_id"],
            )

        with patch(
            "fedbiomed.node.secagg._secagg_setups.SecaggServkeySetup._setup_specific"
        ) as mock_:
            # FIXME: these are already tested...
            mock_.side_effect = FedbiomedError
            self.mock_skm.get.return_value = None
            reply = self.secagg_servkey.setup()
            self.assertIsInstance(reply, ErrorMessage)


class TestSecaggKeySetup(SecaggTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.mock_controller_data = MagicMock(spec=EventWaitExchange)
        self.mock_grpc_controller = MagicMock(spec=GrpcController)
        self.mock_pending_requests = MagicMock(spec=EventWaitExchange)
        self.args = {
            "researcher_id": "my researcher",
            "secagg_id": "my secagg",
            "experiment_id": "my_experiment_id",
            #"share": AdditiveShare(1234),
            "parties": [
                environ["ID"],
                "node2",
                "node3",
                "node4",
            ],
        }

        self.messages = [
            AdditiveSSharingReply(
                node_id="node3",
                dest_node_id="node1",
                secagg_id="test",
                share=1234 
            ),
            AdditiveSSharingReply(
                node_id="node2",
                dest_node_id="node1",
                secagg_id="test",
                share=4321

            ),
            AdditiveSSharingReply(
                node_id="node4",
                dest_node_id="node1",
                secagg_id="test",
                share=4321

            ),
        ]
        self.mock_pending_requests.wait.return_value = True, self.messages

        self.args["grpc_client"] = self.mock_grpc_controller
        self.args["pending_requests"] = self.mock_pending_requests
        self.args["controller_data"] = self.mock_controller_data

    def tearDown(self) -> None:
        pass

    def test_secagg_key_01_init(self):
        secagg = SecaggKeySetup(**self.args)
        self.assertTrue(hasattr(secagg, '_secagg_manager'))

    @patch("fedbiomed.node.secagg_manager.SecaggServkeyManager.add")
    def test_secagg_key_02_setup(self, skmanager_add):
        """Tests key setup for additive key"""
        secagg_addss = SecaggKeySetup(**self.args)

        reply = secagg_addss.setup()
        self.assertEqual(reply.success, True)
        self.assertIsInstance(reply.share, int)

        self.mock_pending_requests.wait.side_effect = FedbiomedError
        reply = secagg_addss.setup()
        self.assertIsInstance(reply, ErrorMessage)

    @patch("fedbiomed.node.secagg_manager.SecaggServkeyManager.add")
    def test_secagg_key_03_setup_2(self, skmanager_add):
        get_rand_values = {'test-1': (5432,  # user_key
                                      (1234, 4321, 1122),  # random splits
                                      (1234, 4321, 4321,)),  #other nodes share
                           'test-2': (1111,
                                      (1111, 2222),
                                      (3333, 5555,)),
                           'test-3': (1012,
                                      (1234, 5678, 3214, 1111, 1023),
                                      (9021, 1022, 4521, 7690, 3213,))
                            }

        for get_rand_bits_val, rand_int_vals, node_shares_val in get_rand_values.values():
            messages = []
            self.args["parties"] = [environ["ID"],]
            for i, val in enumerate(node_shares_val):
                messages.append(
                    AdditiveSSharingReply(
                        node_id=f"node{i}",
                        dest_node_id="node1",
                        secagg_id="test",
                        share=val
                    ))

                self.args["parties"].append(f"node{i}")

            self.mock_pending_requests.wait.return_value = True, messages

            with (patch('fedbiomed.common.secagg._additive_ss.random.randint') as randomint_mock,
                  patch('fedbiomed.node.secagg._secagg_setups.random.SystemRandom.getrandbits') as getrandbits_mock):
                randomint_mock.side_effect = rand_int_vals
                getrandbits_mock.return_value = get_rand_bits_val
                secagg_addss = SecaggKeySetup(**self.args)

                reply = secagg_addss.setup()
            self.assertEqual(reply.success, True)
            check_sum_share = lambda randbits, rand_ints, other_shares: randbits - sum(rand_ints) + sum(other_shares)
            #self.assertEqual(reply.share, 5432 - 1234 - 4321 - 1122 + 1234 + 4321 + 4321)
            self.assertEqual(reply.share, check_sum_share(get_rand_bits_val,
                                                          rand_int_vals,
                                                          node_shares_val))


class TestSecaggDHSetup(SecaggTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.mock_controller_data = MagicMock(spec=EventWaitExchange)
        self.mock_grpc_controller = MagicMock(spec=GrpcController)
        self.mock_pending_requests = MagicMock(spec=EventWaitExchange)
        self.args = {
            "researcher_id": "my researcher",
            "secagg_id": "my secagg",
            "experiment_id": "my_experiment_id",
            "parties": [
                "my researcher",
                environ["ID"],
                "my node2",
                "my node3",
                "my_node4",
            ],
        }
        self.args["grpc_client"] = self.mock_grpc_controller
        self.args["pending_requests"] = self.mock_pending_requests
        self.args["controller_data"] = self.mock_controller_data

    def tearDown(self) -> None:
        pass

    @patch("fedbiomed.node.secagg_manager.SecaggDhManager.add")
    @patch("fedbiomed.node.secagg._secagg_setups.DHKey.export_public_key")
    @patch("fedbiomed.node.secagg._secagg_setups.DHKeyAgreement.agree")
    @patch("fedbiomed.node.secagg._secagg_setups.send_nodes")
    def test_secagg_dh_02_setup(
        self,
        send_node_mock,
        dh_key_agreement_agree,
        dh_key_export_public_key,
        dhmanager_add_mock,
    ):

        received_msg_with_all_nodes = []

        for n in self.args["parties"][1:]:
            # remove first and last parties (simulates a drop out from 'node 4')
            received_msg_with_all_nodes.append(
                NodeToNodeMessages.format_outgoing_message(
                    {
                        "request_id": "1234",
                        "node_id": n,
                        "dest_node_id": n,
                        "secagg_id": self.args["secagg_id"],
                        "command": "key-reply",
                        "public_key": b"some-public-key",
                    }
                )
            )
        # received_msg_with_dropout = received_msg_with_all_nodes.copy()
        # received_msg_with_dropout.pop(-1)
        # def fake_send_node(grpc_client, pending_req, researcher_id, other_nodes, other_nodes_msg):

        key = b"public-key"
        dskey = b"derived-shared-key"
        dh_key_agreement_agree.return_value = dskey
        send_node_mock.return_value = True, received_msg_with_all_nodes
        dh_key_export_public_key.return_value = key
        secagg_dh = SecaggDHSetup(**self.args)
        reply = secagg_dh.setup()

        # checks
        self.mock_controller_data.event.assert_called_once_with(
            self.args["secagg_id"], {"public_key": key}
        )

        context = {n: dskey for n in self.args["parties"][1:]}
        dhmanager_add_mock.assert_called_once_with(
            self.args["secagg_id"],
            self.args["parties"],
            context,
            self.args["experiment_id"],
        )
        self.assertTrue(reply.success)

    @patch("fedbiomed.node.secagg_manager.SecaggDhManager.add")
    @patch("fedbiomed.node.secagg._secagg_setups.DHKey.export_public_key")
    @patch("fedbiomed.node.secagg._secagg_setups.DHKeyAgreement.agree")
    @patch("fedbiomed.node.secagg._secagg_setups.send_nodes")
    def test_secagg_dh_03_setup_error(
        self,
        send_node_mock,
        dh_key_agreement_agree,
        dh_key_export_public_key,
        dhmanager_add_mock,
    ):

        received_msg_with_node_dropout = []

        for n in self.args["parties"][1:-2]:
            # remove first and last parties (simulates a drop out from 'node 4')
            received_msg_with_node_dropout.append(
                NodeToNodeMessages.format_outgoing_message(
                    {
                        "request_id": "1234",
                        "node_id": n,
                        "dest_node_id": n,
                        "secagg_id": self.args["secagg_id"],
                        "command": "key-reply",
                        "public_key": b"some-public-key",
                    }
                )
            )

        key = b"public-key"
        dskey = b"derived-shared-key"
        dh_key_agreement_agree.return_value = dskey
        send_node_mock.return_value = False, received_msg_with_node_dropout
        dh_key_export_public_key.return_value = key
        secagg_dh = SecaggDHSetup(**self.args)
        reply = secagg_dh.setup()
        self.assertIsInstance(reply, ErrorMessage)


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
            "parties": ["r-1", "node-1", "node-2"],
        }

        # Test server key setup
        secagg_setup = SecaggSetup(**args)()
        self.assertIsInstance(secagg_setup, SecaggServkeySetup)

        # Raise element type
        args["element"] = 1
        args["experiment_id"] = ""
        with self.assertRaises(FedbiomedSecaggError):
            SecaggSetup(**args)()

        args = {
            "experiment_id": "experiment-id",
            "element": 2,
            "secagg_id": "secagg-id",
            "parties": ["r-1", "node-1", "node-2"],
            "grpc_client": MagicMock(spec=GrpcController),
            "pending_requests": MagicMock(spec=EventWaitExchange),
            "controller_data": MagicMock(spec=EventWaitExchange),
            "researcher_id": "r-1",}

        secagg_setup = SecaggSetup( **args, )()
        self.assertIsInstance(secagg_setup, SecaggKeySetup)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
