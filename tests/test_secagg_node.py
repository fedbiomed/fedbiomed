import unittest
from copy import deepcopy
from unittest.mock import MagicMock, mock_open, patch

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase

#############################################################
from fedbiomed.common.exceptions import FedbiomedSecaggError, FedbiomedError
from fedbiomed.common.message import (
    AdditiveSSharingReply,
    ErrorMessage,
    KeyReply,
)
from fedbiomed.common.synchro import EventWaitExchange

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ
from fedbiomed.node.secagg import (
    SecaggDHSetup,
    SecaggServkeySetup,
    SecaggSetup,
)
import fedbiomed.node.secagg
from fedbiomed.node.requests import NodeToNodeRouter


class SecaggTestCase(NodeTestCase):

    def setUp(self) -> None:
        self.patch_skm = patch.object(fedbiomed.node.secagg._secagg_setups, "SKManager")
        self.patch_cm = patch.object(fedbiomed.node.secagg._secagg_setups, "_CManager")

        self.mock_skm = self.patch_skm.start()
        self.mock_cm = self.patch_cm.start()

        unittest.mock.MagicMock.mpc_data_dir = unittest.mock.PropertyMock(
            return_value="dummy/path/to/output"
        )
        unittest.mock.MagicMock.tmp_dir = unittest.mock.PropertyMock(
            return_value=environ["TMP_DIR"]
        )

    def tearDown(self) -> None:
        self.patch_skm.stop()
        self.patch_cm.stop()


class TestSecaggServkeySetup(SecaggTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.mock_controller_data = MagicMock(spec=EventWaitExchange)
        self.mock_grpc_controller = MagicMock(spec=GrpcController)
        self.mock_pending_requests = MagicMock(spec=EventWaitExchange)
        self.mock_n2n_router = MagicMock(spec=NodeToNodeRouter)
        self.args = {
            "researcher_id": "my researcher",
            "secagg_id": "my secagg",
            "experiment_id": "my_experiment_id",
            "parties": [
                environ["ID"],
                "node2",
                "node3",
                "node4",
            ],
        }

        self.messages = [
            AdditiveSSharingReply(
                node_id="node3", dest_node_id="node1", secagg_id="test", share=1234
            ),
            AdditiveSSharingReply(
                node_id="node2", dest_node_id="node1", secagg_id="test", share=4321
            ),
            AdditiveSSharingReply(
                node_id="node4", dest_node_id="node1", secagg_id="test", share=4321
            ),
        ]
        self.mock_pending_requests.wait.return_value = True, self.messages

        self.args["grpc_client"] = self.mock_grpc_controller
        self.args["pending_requests"] = self.mock_pending_requests
        self.args["controller_data"] = self.mock_controller_data
        self.args['n2n_router'] = self.mock_n2n_router

    def tearDown(self) -> None:
        pass

    def test_secagg_key_01_init(self):
        secagg = SecaggServkeySetup(**self.args)
        self.assertTrue(hasattr(secagg, "_secagg_manager"))

    @patch("fedbiomed.node.secagg_manager.SecaggServkeyManager.add")
    @patch("fedbiomed.node.secagg._secagg_setups.send_nodes")
    def test_secagg_key_02_setup(self, send_node_mock, skmanager_add):
        """Tests key setup for additive key"""
        secagg_addss = SecaggServkeySetup(**self.args)

        received_msg_with_all_nodes = []

        for n in self.args["parties"][1:]:
            # remove first and last parties (simulates a drop out from 'node 4')
            received_msg_with_all_nodes.append(
                AdditiveSSharingReply(
                    **{
                        "request_id": "1234",
                        "node_id": n,
                        "dest_node_id": n,
                        "secagg_id": self.args["secagg_id"],
                        "share": 12345,
                    }
                )
            )
        send_node_mock.return_value = True, received_msg_with_all_nodes

        reply = secagg_addss.setup()
        self.assertEqual(reply.success, True)
        self.assertIsInstance(reply.share, int)

    @patch("fedbiomed.node.secagg_manager.SecaggServkeyManager.add")
    def test_secagg_key_02_setup(self, skmanager_add):
        """Tests key setup for additive key"""
        secagg_addss = SecaggServkeySetup(**self.args)
        self.mock_n2n_router.format_outgoing_overlay.return_value = [b'overlay'], b'salt'

        self.mock_pending_requests.wait.side_effect = FedbiomedError
        reply = secagg_addss.setup()
        self.assertIsInstance(reply, ErrorMessage)

    @patch("fedbiomed.node.secagg_manager.SecaggServkeyManager.add")
    @patch("fedbiomed.node.secagg._secagg_setups.send_nodes")
    def test_secagg_key_03_setup_2(self, send_nodes_mock, skmanager_add):
        get_rand_values = {
            "test-1": (
                5432,  # user_key
                (1234, 4321, 1122),  # random splits
                (
                    1234,
                    4321,
                    4321,
                ),
            ),  # other nodes share
            "test-2": (
                1111,
                (1111, 2222),
                (
                    3333,
                    5555,
                ),
            ),
            "test-3": (
                1012,
                (1234, 5678, 3214, 1111, 1023),
                (
                    9021,
                    1022,
                    4521,
                    7690,
                    3213,
                ),
            ),
        }

        for (
            get_rand_bits_val,
            rand_int_vals,
            node_shares_val,
        ) in get_rand_values.values():
            messages = []
            self.args["parties"] = [
                environ["ID"],
            ]
            for i, val in enumerate(node_shares_val):
                messages.append(
                    AdditiveSSharingReply(
                        node_id=f"node{i}",
                        dest_node_id="node1",
                        secagg_id="test",
                        share=val,
                    )
                )

                self.args["parties"].append(f"node{i}")

            send_nodes_mock.return_value = True, messages
            self.mock_n2n_router.format_outgoing_overlay.return_value = [b'overlay'], b'salt'

            with (
                patch(
                    "fedbiomed.common.secagg._additive_ss.random.randint"
                ) as randomint_mock,
                patch(
                    "fedbiomed.node.secagg._secagg_setups.random.SystemRandom.getrandbits"
                ) as getrandbits_mock,
            ):
                randomint_mock.side_effect = rand_int_vals
                getrandbits_mock.return_value = get_rand_bits_val
                secagg_addss = SecaggServkeySetup(**self.args)

                reply = secagg_addss.setup()
            self.assertEqual(reply.success, True)
            check_sum_share = (
                lambda randbits, rand_ints, other_shares: randbits
                - sum(rand_ints)
                + sum(other_shares)
            )
            # self.assertEqual(reply.share, 5432 - 1234 - 4321 - 1122 + 1234 + 4321 + 4321)
            self.assertEqual(
                reply.share,
                check_sum_share(get_rand_bits_val, rand_int_vals, node_shares_val),
            )


class TestSecaggDHSetup(SecaggTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.mock_controller_data = MagicMock(spec=EventWaitExchange)
        self.mock_grpc_controller = MagicMock(spec=GrpcController)
        self.mock_pending_requests = MagicMock(spec=EventWaitExchange)
        self.mock_n2n_router = MagicMock(spec=NodeToNodeRouter)
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
        self.args['n2n_router'] = self.mock_n2n_router

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
                KeyReply(
                    **{
                        "request_id": "1234",
                        "node_id": n,
                        "dest_node_id": n,
                        "secagg_id": self.args["secagg_id"],
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


class TestSecaggSetup(NodeTestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_secagg_setup_01_initialization(self):
        # Raise element type
        args = {
            "experiment_id": "experiment-id",
            "element": 0,
            "secagg_id": "secagg-id",
            "parties": ["node-1", "node-2"],
            "grpc_client": MagicMock(spec=GrpcController),
            "pending_requests": MagicMock(spec=EventWaitExchange),
            "controller_data": MagicMock(spec=EventWaitExchange),
            "n2n_router": MagicMock(spec=NodeToNodeRouter),
            "researcher_id": "r-1",
        }

        args["element"] = 12
        with self.assertRaises(FedbiomedSecaggError):
            SecaggSetup(**args)()

        args['element'] = 0
        args["parties"] = []
        with self.assertRaises(FedbiomedSecaggError):
            SecaggSetup(**args)()


        args["parties"] = ["node-1", "node-2"]
        secagg_setup = SecaggSetup(
            **args,
        )()
        self.assertIsInstance(secagg_setup, SecaggServkeySetup)

        args2 = {**args}
        args2["element"] = 1
        secagg_setup = SecaggSetup(
            **args2,
        )()
        self.assertIsInstance(secagg_setup, SecaggDHSetup)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
