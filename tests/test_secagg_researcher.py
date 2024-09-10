import copy
from typing import Tuple
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from fedbiomed.common.secagg_manager import BaseSecaggManager
from testsupport.base_case import ResearcherTestCase
from testsupport.base_mocks import MockRequestModule

#############################################################
import fedbiomed.researcher.secagg._secagg_context

from fedbiomed.researcher.environ import environ
from fedbiomed.common.exceptions import FedbiomedSecaggError, FedbiomedError
from fedbiomed.common.constants import SecaggElementTypes, __secagg_element_version__
from fedbiomed.common.message import AdditiveSSSetupReply, SecaggReply, SecaggDeleteReply
from fedbiomed.researcher.secagg import (
    SecaggServkeyContext,
    SecaggContext,
    SecaggDHContext,
    SecaggKeyContext
)
from fedbiomed.researcher.requests import FederatedRequest


class BaseTestCaseSecaggContext(ResearcherTestCase, MockRequestModule):

    def setUp(self) -> None:

        MockRequestModule.setUp(
            self, "fedbiomed.researcher.secagg._secagg_context.Requests"
        )

        self.patch_cm = patch.object(
            fedbiomed.researcher.secagg._secagg_context, "_CManager"
        )
        self.patch_mpc = patch.object(
            fedbiomed.researcher.secagg._secagg_context, "MPCController"
        )
        # self.patch_requests = patch("fedbiomed.researcher.secagg._secagg_context.Requests")
        self.patch_skmanager = patch.object(
            fedbiomed.researcher.secagg._secagg_context, "_SKManager"
        )

        self.mock_cm = self.patch_cm.start()
        self.mock_mpc = self.patch_mpc.start()
        # self.m_requests = self.patch_requests.start()
        self.mock_skmanager = self.patch_skmanager.start()

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
        self.patch_cm.stop()
        self.patch_mpc.stop()
        self.patch_skmanager.stop()
        super().tearDown()


class TestBaseSecaggContext(BaseTestCaseSecaggContext):
    create_round_specific_output: Tuple = (
        None,
        None,
    )

    def setUp(self):
        super().setUp()
        self.abstract_methods_patcher = patch.multiple(
            SecaggContext, __abstractmethods__=set()
        )
        self.abstract_methods_patcher.start()
        self.parties = [environ["ID"], "party2", "party3"]
        self.secagg_context = SecaggContext(
            parties=self.parties, experiment_id="experiment-id"
        )

    def tearDown(self) -> None:
        super().tearDown()
        self.abstract_methods_patcher.stop()
        TestBaseSecaggContext.create_round_specific_output = (
            None,
            None,
        )

    @staticmethod
    def create_round_specific(msg, payload) -> Tuple:
        payload()
        return TestBaseSecaggContext.create_round_specific_output

    def test_base_secagg_context_01_init(self):
        """Test successful and failed object instantiations"""
        # Succeeded with various secagg_id
        for secagg_id in (None, "one secagg id string", "x"):
            context = SecaggContext(
                parties=[environ["ID"], "party2", "party3"],
                experiment_id="experiment-id",
                secagg_id=secagg_id,
            )
        self.assertEqual(context.secagg_id, secagg_id)

        # Invalid type parties
        with self.assertRaises(FedbiomedSecaggError):
            SecaggContext(
                parties=[environ["ID"], 12, 12], experiment_id="experiment-id"
            )

        # Failed with bad secagg_id
        for secagg_id in ("", 3, ["not a string"]):
            with self.assertRaises(FedbiomedSecaggError):
                SecaggContext(
                    parties=[environ["ID"], "party2", "party3"],
                    experiment_id="a experiment id",
                    secagg_id=secagg_id,
                )

    def test_secagg_context_02_getters_setters(self):
        """Tests setters and getters"""

        self.assertEqual(self.secagg_context.experiment_id, "experiment-id")
        self.assertIsInstance(self.secagg_context.secagg_id, str)
        self.assertFalse(self.secagg_context.status)
        self.assertIsNone(self.secagg_context.context)

    def test_secagg_context_03_secagg_round(self):
        """Setup then delete a secagg class"""

        replies = {
            "node-1": SecaggReply(
                **{
                    "researcher_id": environ["ID"],
                    "secagg_id": self.secagg_context.secagg_id,
                    "success": True,
                    "node_id": "party2",
                    "msg": "Fake request",
                    "command": "secagg",
                }
            )
        }

        replies.update({"node-2": copy.deepcopy(replies["node-1"])})
        replies["node-2"].node_id = "party3"

        # Patch response

        self.mock_federated_request.replies.return_value = replies
        self.mock_federated_request.errors.return_value = []
        self.secagg_context._element = (
            SecaggElementTypes.SERVER_KEY
        )  # pylint: disable=W0212
        setattr(
            self.secagg_context,
            "_secagg_manager",
            MagicMock(
                spec=BaseSecaggManager,
                get=MagicMock(return_value={x: "1235" for x in self.parties}),
            ),
        )
        with (
            patch(
                "fedbiomed.researcher.secagg.SecaggContext._create_payload_specific"
            ) as mock_payload,
            patch(
                "fedbiomed.researcher.secagg.SecaggContext._secagg_round_specific"
            ) as mock_secagg_round_specific,
        ):

            # Test 1: with `create_round_specific` returning a context
            TestBaseSecaggContext.create_round_specific_output = {
                x: "1235" for x in self.parties
            }, {x: True for x in self.parties}
            mock_secagg_round_specific.side_effect = (
                TestBaseSecaggContext.create_round_specific
            )
            mock_payload.return_value = ("KEY", True)
            result = self.secagg_context.setup()
            self.assertTrue(result)
            self.assertIsInstance(result, bool)

            # Test 2: with `create_round_specific` returning `None`
            secagg_manager_get_mock = MagicMock()
            secagg_manager_get_mock.side_effect = [
                None,
                {x: "1235" for x in self.parties},
            ]
            setattr(
                self.secagg_context,
                "_secagg_manager",
                MagicMock(spec=BaseSecaggManager, get=secagg_manager_get_mock),
            )
            result = self.secagg_context.setup()
            self.assertTrue(result)
            self.assertIsInstance(result, bool)

            # with self.assertRaises(FedbiomedSecaggError):
            #     self.mock_policy.has_stopped_any.return_value = True
            #     self.secagg_context.setup()

    def test_secagg_06_breakpoint(self):
        """Save and load breakpoint status for secagg class"""

        expected_state = {
            "class": type(self.secagg_context).__name__,
            "module": self.secagg_context.__module__,
            "arguments": {
                "secagg_id": self.secagg_context.secagg_id,
                "experiment_id": self.secagg_context.experiment_id,
                "parties": self.secagg_context.parties,
            },
            "attributes": {
                "_researcher_id": self.env["RESEARCHER_ID"],
                "_status": self.secagg_context.status,
                "_context": self.secagg_context.context,
            },
        }

        state = self.secagg_context.save_state_breakpoint()
        self.assertEqual(state, expected_state)

        # 2. Load complete breakpoint
        # nota: Cannot test content of complete state (not verified by function)

        # prepare
        state = {
            "class": "SecaggContext",
            "module": "fedbiomed.researcher.secagg",
            "arguments": {
                "secagg_id": "my_secagg_id",
                "parties": [environ["ID"], "TWO_PARTIES", "THREE_PARTIES"],
                "experiment_id": "my_experiment_id",
            },
            "attributes": {
                "_researcher_id": environ["ID"],
                "_status": False,
                "_context": "MY CONTEXT",
            },
        }


class TestSecaggServkeyContext(BaseTestCaseSecaggContext):

    def setUp(self) -> None:
        super().setUp()
        self.parties = [environ["ID"], "party2", "party3"]

        self.mock_skmanager.get.return_value = None
        self.srvkey_context = SecaggServkeyContext(
            parties=self.parties, experiment_id="experiment-id"
        )

        self.database_entry = {
            "secagg_version": str(__secagg_element_version__),
            "secagg_id": "secagg_id",
            "parties": self.parties,
            "secagg_elem": SecaggElementTypes.SERVER_KEY,
            "experiment_id": "experiment_id",
            "context": {"server_key": 123445, "biprime": 123445},
        }

    def tearDown(self) -> None:
        super().tearDown()

    def test_servkey_context_01_init(self):
        """Tests failed init scenarios with bad_experiment_id"""

        # Does not respect min number of parties
        with self.assertRaises(FedbiomedError):
            SecaggServkeyContext(parties=[environ["ID"], "party2"], experiment_id="x1")
        # First party is not the researcher
        with self.assertRaises(FedbiomedError):
            SecaggServkeyContext(
                parties=["pp1", "party2", "party3"], experiment_id="x2"
            )

    @patch("fedbiomed.researcher.secagg.SecaggServkeyContext._create_payload_specific")
    def test_secagg_02_payload(self, patch_create_payload):
        """Test cases payload for secagg servkey setup"""
        dummy_context = "context"
        dummy_status = "status"
        patch_create_payload.return_value = (dummy_context, dummy_status)

        for return_value, context, value in (
            (None, dummy_context, dummy_status),
            # Not tested by _matching_parties*
            #
            # (3, 3, False),
            # ({}, {}, False),
            # ({'toto': 1}, {'toto': 1}, False),
            # ({'parties': 3}, {'parties': 3}, False),
            # ({'parties': ['only_one_party']}, {'parties': ['only_one_party']}, False),
            (
                {
                    "parties": [environ["ID"], "party2", "party3"],
                    "context": {"servkey": "123456"},
                },
                {
                    "parties": [environ["ID"], "party2", "party3"],
                    "context": {"servkey": "123456"},
                },
                True,
            ),
            (
                {
                    "parties": [environ["ID"], "party3", "party2"],
                    "context": {"servkey": "123456"},
                },
                {
                    "parties": [environ["ID"], "party3", "party2"],
                    "context": {"servkey": "123456"},
                },
                True,
            ),
            (
                {
                    "parties": [environ["ID"], "party2", "party3", "party4"],
                    "context": {"servkey": "123456"},
                },
                {
                    "parties": [environ["ID"], "party2", "party3", "party4"],
                    "context": {"servkey": "123456"},
                },
                False,
            ),
            (
                {
                    "parties": ["party2", environ["ID"], "party3"],
                    "context": {"servkey": "123456"},
                },
                {
                    "parties": ["party2", environ["ID"], "party3"],
                    "context": {"servkey": "123456"},
                },
                False,
            ),
        ):
            self.mock_skmanager.get.side_effect = [return_value, context]
            srvkey_context = SecaggServkeyContext(
                parties=[environ["ID"], "party2", "party3"],
                experiment_id="experiment-id",
            )

            payload_context, payload_value = srvkey_context._create_payload()
            self.assertEqual(payload_context, context)
            self.assertEqual(payload_value, value)

    def test_servkey_context_03_payload_create(self):
        key_value = 123456789

        def mock_skmanager_add(secagg_id, parties, context, experiment_id):
            """Mimicks saving and after loading from a database"""
            self.database_entry["context"] = context

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.side_effect = [
                str(key_value),
                '{"biprime": 12345}',
            ]
            self.mock_skmanager.get.side_effect = [None, self.database_entry]
            self.mock_skmanager.add.side_effect = mock_skmanager_add
            context, status = self.srvkey_context._create_payload()

            self.mock_cm.write_mpc_certificates_for_experiment.assert_called_once_with(
                path_certificates="dummy/path/to/output",
                path_ips=environ["TMP_DIR"],
                self_id=environ["ID"],
                self_ip=environ["MPSPDZ_IP"],
                self_port=environ["MPSPDZ_PORT"],
                self_private_key=environ["MPSPDZ_CERTIFICATE_KEY"],
                self_public_key=environ["MPSPDZ_CERTIFICATE_PEM"],
                parties=[environ["ID"], "party2", "party3"],
            )

            self.mock_mpc.exec_shamir.called_once_with(
                party_number=0, num_parties=3, ip_addresses="dummy/ip"
            )

            self.assertEqual(context["context"]["server_key"], key_value)
            self.assertEqual(status, True)
            mock_open.reset_mock()
            self.mock_skmanager.get.side_effect = [None, self.database_entry]

            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedSecaggError):
                self.srvkey_context._create_payload()

            self.mock_mpc.exec_shamir.side_effect = Exception
            self.mock_skmanager.get.side_effect = [None, self.database_entry]
            with self.assertRaises(FedbiomedSecaggError):
                self.srvkey_context._create_payload()

    @patch("time.sleep")
    def test_servkey_context_04_secagg_delete(self, patch_time_sleep):
        """Setup then delete a secagg class"""

        replies = {
            "node-1": SecaggDeleteReply(
                **{
                    "researcher_id": environ["ID"],
                    "secagg_id": self.srvkey_context.secagg_id,
                    "success": True,
                    "node_id": "party2",
                    "msg": "Fake request",
                    "command": "secagg-delete",
                }
            )
        }

        replies.update({"node-2": copy.deepcopy(replies["node-1"])})
        replies["node-2"].node_id = "party3"

        # Patch response
        self.mock_federated_request.replies.return_value = replies
        self.srvkey_context._element = SecaggElementTypes.SERVER_KEY

        result = self.srvkey_context.delete()
        self.assertTrue(result)

        with patch(
            "fedbiomed.researcher.secagg.SecaggContext._delete_payload"
        ) as mock_payload:
            mock_payload.return_value = ("KEY", True)
            result = self.srvkey_context.delete()
            self.assertTrue(result)

            mock_payload.return_value = ("KEY", False)
            result = self.srvkey_context.delete()
            self.assertFalse(result)

    def test_servkey_context_05_delete_payload_fail(self):
        """Test when removing from the database fails"""
        for s in (None, True, False, 3, [], {"one": 1}):
            self.mock_skmanager.remove.return_value = s

            context, status = self.srvkey_context._delete_payload()

            self.assertEqual(context, None)
            self.assertEqual(status, s)


class TestServerKeyContext(BaseTestCaseSecaggContext):
    
    def setUp(self) -> None:
        
        super().setUp()
        self.parties = [environ["ID"], "party2", "party3"]

        self.mock_skmanager.get.return_value = None

        self._secagg_key_context = SecaggKeyContext(parties=self.parties[1:],
                                                    experiment_id="experiment-id",
                                                    secagg_id='secagg_id')
        self.database_entry = {
            "secagg_version": str(__secagg_element_version__),
            "secagg_id": "secagg_id",
            "parties": self.parties,
            "secagg_elem": SecaggElementTypes.SERVER_KEY,
            "experiment_id": "experiment_id",
            "context": {"share": 1234},
        }

    def test_01_init(self):
        secagg_key_context_ok = SecaggKeyContext(self.parties,
                                                 experiment_id="experiment-id",
                                                 )

        with self.assertRaises(FedbiomedSecaggError):
            SecaggKeyContext(parties=[], experiment_id='experiment_id')


    def test_02_secagg_round_specific(self):
        shares = {'test-1': ({'party1': 12, 'party2': 12}, 24),
                  'tets-2': ({'party1': [12, 13], 'party2': [12, 13]},
                             [12*2, 13*2])}

        for (test_share_values, expected_share_value) in shares.values():
            _, res = self._run_secagg_config(self._secagg_key_context, test_share_values, 
                                             expected_share_value)
            self.assertTrue(res)
            self.assertIsInstance(res, bool)
            # check save from SKManager
            self.mock_skmanager.add.assert_called_with('secagg_id',
                                                       self.parties[1:],
                                                       {'share': expected_share_value},
                                                       'experiment-id')

    def test_03_saving_loading_breakpoint(self):
        same_obj_attr = ('_secagg_id', '_parties', '_researcher_id', '_status', '_context', '_experiment_id', '_element')
        same_inst_attr = ('_v', '_secagg_manager', '_requests',)
        # before running round_specific
        state = self._secagg_key_context.save_state_breakpoint()

        self.assertDictContainsSubset({'_status': False,
                                       '_context': None,
                                       '_researcher_id': self.env["RESEARCHER_ID"],},
                                      state['attributes'])
        state = copy.deepcopy(state)
        loaded_secagg = SecaggKeyContext.load_state_breakpoint(state)

        self.check_similarities_in_obj(self._secagg_key_context, loaded_secagg, same_obj_attr, same_inst_attr 
                                       )
        
        shares = {'test-1': ({'party1': 12, 'party2': 12}, 24),
                  'tets-2': ({'party1': [12, 13], 'party2': [12, 13]},
                             [12*2, 13*2])}
        
        for (test_share_values, expected_share_value) in shares.values():
            del loaded_secagg
            self._run_secagg_config(self._secagg_key_context,
                                    test_share_values, expected_share_value)
            
            state = self._secagg_key_context.save_state_breakpoint()
            state = copy.deepcopy(state)

            loaded_secagg = SecaggKeyContext.load_state_breakpoint(state)

            self.check_similarities_in_obj(self._secagg_key_context, loaded_secagg,
                                           same_obj_attr, same_inst_attr)

    def _run_secagg_config(self, secagg, test_share_values, expected_share_value):
        self.mock_federated_request.replies.return_value = {
                "party1": AdditiveSSSetupReply(
                    **{
                        "researcher_id": "xx",
                        "success": True,
                        "node_id": "party2",
                        "command": "secagg-additive-ss-setup-reply",
                        "msg": "x",
                        "secagg_id": "s1",
                        "share": test_share_values['party1']
                    }
                ),
                "party2": AdditiveSSSetupReply(
                    **{
                        "researcher_id": "xx",
                        "success": True,
                        "node_id": "party3",
                        "command": "secagg-additive-ss-setup-reply",
                        "msg": "x",
                        "secagg_id": "s1",
                        "share": test_share_values['party2']
                    }
                ),
            }

        self.mock_federated_request.errors.return_value = None
        type(
            self.mock_federated_request
        ).policy.return_value.has_stopped_any.return_value = False

        res = secagg.setup()
        return secagg, res
        
    def check_similarities_in_obj(self, obj1, obj2, same_obj_attr,
                                  same_instance_attr):

        for attr in obj1.__dict__:
            attr1, attr2 = getattr(obj1, attr), getattr(obj2, attr)
            if attr in same_obj_attr and attr1 != attr2:
                self.assertFalse(True, f"{attr1} and {attr2} are not equal")
            elif attr in same_instance_attr:
                self.assertIsInstance(attr1, type(attr2))


class TestSecaggDHContext(BaseTestCaseSecaggContext):

    def setUp(self) -> None:
        super().setUp()

        self.dhmanager_p = patch.object(
            fedbiomed.researcher.secagg._secagg_context, "_DHManager"
        )
        self.dhmanager = self.dhmanager_p.start()

        self.secagg_dhcontext = SecaggDHContext(
            parties=["party2", "party3"], experiment_id="", secagg_id="secagg_id"
        )

    def tearDown(self) -> None:
        super().tearDown()
        self.dhmanager_p.stop()

    def test_01_dhcontext_init_error_cases(self):

        with self.assertRaises(FedbiomedError):
            SecaggDHContext(
                parties=["party2"], experiment_id="exp_id", secagg_id="secagg_id"
            )

    def test_02_dhcontext_create_payload_specific(self):
        """Test creating payload"""
        test, _ = self.secagg_dhcontext._create_payload_specific({"my-context": 1})
        self.assertDictEqual(test, {"my-context": 1})

    def test_03_dh_context_secagg_round_specific(self):

        self.mock_federated_request.replies.return_value = {
            "party1": SecaggReply(
                **{
                    "researcher_id": "xx",
                    "success": True,
                    "node_id": "party2",
                    "command": "secagg",
                    "msg": "x",
                    "secagg_id": "s1",
                }
            ),
            "party2": SecaggReply(
                **{
                    "researcher_id": "xx",
                    "success": True,
                    "node_id": "party3",
                    "command": "secagg",
                    "msg": "x",
                    "secagg_id": "s1",
                }
            ),
        }

        self.dhmanager.get.return_value = {"parties": ["party2", "party3"]}

        self.mock_federated_request.errors.return_value = None
        type(
            self.mock_federated_request
        ).policy.return_value.has_stopped_any.return_value = False

        result = self.secagg_dhcontext.setup()
        self.assertTrue(result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
