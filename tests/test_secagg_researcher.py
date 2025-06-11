import copy
import unittest
import tempfile
from typing import Tuple
from unittest.mock import ANY, MagicMock, patch

from testsupport.base_mocks import MockRequestModule

#############################################################
from fedbiomed.common.constants import SecaggElementTypes, __secagg_element_version__
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedSecaggError
from fedbiomed.common.message import (
    AdditiveSSSetupReply,
    SecaggReply,
)
from fedbiomed.researcher.secagg import (
    SecaggContext,
    SecaggDHContext,
    SecaggServkeyContext,
)

test_id = "researcher-test-id"


class BaseTestCaseSecaggContext(unittest.TestCase, MockRequestModule):  # pylint: disable=missing-docstring
    def setUp(self) -> None:
        MockRequestModule.setUp(
            self, "fedbiomed.researcher.secagg._secagg_context.Requests"
        )

        # self.patch_requests = patch("fedbiomed.researcher.secagg._secagg_context.Requests")
        self.patch_skmanager = patch(
            "fedbiomed.researcher.secagg._secagg_context.SecaggServkeyManager"
        )
        self.patch_skmanager_s = self.patch_skmanager.start()
        self.mock_skmanager = MagicMock()
        self.patch_skmanager_s.return_value = self.mock_skmanager

        temp_dir = tempfile.TemporaryDirectory()
        unittest.mock.MagicMock.tmp_dir = unittest.mock.PropertyMock(
            return_value=temp_dir
        )

    def tearDown(self) -> None:
        self.patch_skmanager.stop()
        super().tearDown()


class TestBaseSecaggContext(BaseTestCaseSecaggContext):  # pylint: disable=missing-class-docstring
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
        self.parties = [test_id, "party2", "party3"]
        self.secagg_context = SecaggContext(
            researcher_id=test_id, parties=self.parties, experiment_id="experiment-id"
        )

    def tearDown(self) -> None:
        super().tearDown()
        self.abstract_methods_patcher.stop()
        TestBaseSecaggContext.create_round_specific_output = (
            None,
            None,
        )

    @staticmethod
    def create_round_specific(msg, payload) -> Tuple:  # pylint: disable=unused-argument, missing-docstring
        payload()
        return TestBaseSecaggContext.create_round_specific_output

    def test_base_secagg_context_01_init(self):
        """Test successful and failed object instantiations"""
        # Succeeded with various secagg_id
        for secagg_id in (None, "one secagg id string", "x"):
            context = SecaggContext(
                researcher_id=test_id,
                parties=[test_id, "party2", "party3"],
                experiment_id="experiment-id",
                secagg_id=secagg_id,
            )  # type: ignore
        self.assertEqual(context.secagg_id, secagg_id)

        # Invalid type parties
        with self.assertRaises(FedbiomedSecaggError):
            SecaggContext(
                researcher_id=test_id,
                parties=[test_id, 12, 12],
                experiment_id="experiment-id",
            )  # type: ignore

        # Failed with bad secagg_id
        for secagg_id in ("", 3, ["not a string"]):
            with self.assertRaises(FedbiomedSecaggError):
                SecaggContext(
                    researcher_id=test_id,
                    parties=[test_id, "party2", "party3"],
                    experiment_id="a experiment id",
                    secagg_id=secagg_id,
                )  # type: ignore

    def test_secagg_context_02_getters_setters(self):
        """Tests setters and getters"""

        self.assertEqual(self.secagg_context.experiment_id, "experiment-id")
        self.assertIsInstance(self.secagg_context.secagg_id, str)
        self.assertFalse(self.secagg_context.status)
        self.assertIsNone(self.secagg_context.context)

    def test_secagg_06_breakpoint(self):
        """Save and load breakpoint status for secagg class"""

        expected_state = {
            "class": type(self.secagg_context).__name__,
            "module": self.secagg_context.__module__,
            "arguments": {
                "secagg_id": self.secagg_context.secagg_id,
                "experiment_id": self.secagg_context.experiment_id,
                "parties": self.secagg_context.parties,
                "researcher_id": test_id,
            },
            "attributes": {
                "_status": self.secagg_context.status,
                "_context": self.secagg_context.context,
            },
        }

        state = self.secagg_context.save_state_breakpoint()
        self.assertEqual(state, expected_state)

        # prepare
        state = {
            "class": "SecaggContext",
            "module": "fedbiomed.researcher.secagg",
            "arguments": {
                "secagg_id": "my_secagg_id",
                "parties": [test_id, "TWO_PARTIES", "THREE_PARTIES"],
                "experiment_id": "my_experiment_id",
                "researcher_id": test_id,
            },
            "attributes": {
                "_status": False,
                "_context": "MY CONTEXT",
            },
        }


class TestSecaggServkeyContext(BaseTestCaseSecaggContext):  # pylint: disable=missing-docstring
    reply = AdditiveSSSetupReply(
        **{
            "researcher_id": "xx",
            "success": True,
            "node_id": "party1",
            "msg": "x",
            "secagg_id": "s1",
            "share": 12,
        }
    )

    def setUp(self) -> None:
        super().setUp()
        self.parties = [test_id, "party2", "party3"]

        self.mock_skmanager.get.return_value = None

        self._secagg_key_context = SecaggServkeyContext(
            researcher_id=test_id,
            parties=self.parties[1:],
            experiment_id="experiment-id",
            secagg_id="secagg_id",
        )
        self.database_entry = {
            "secagg_version": str(__secagg_element_version__),
            "secagg_id": "secagg_id",
            "parties": self.parties,
            "secagg_elem": SecaggElementTypes.SERVER_KEY,
            "experiment_id": "experiment_id",
            "context": {"share": 1234},
        }

    def test_01_init(self):
        _ = SecaggServkeyContext(
            test_id,
            self.parties,
            experiment_id="experiment-id",
        )

        with self.assertRaises(FedbiomedSecaggError):
            SecaggServkeyContext(test_id, parties=[], experiment_id="experiment_id")

    def test_02_secagg_round(self):
        """Test secagg round"""

        r1 = copy.deepcopy(self.reply)
        r2 = copy.deepcopy(self.reply)
        r2.node_id = "party2"

        self.mock_federated_request.replies.return_value = {"party1": r1, "party2": r2}
        self.mock_federated_request.errors.return_value = None
        type(
            self.mock_federated_request
        ).policy.return_value.has_stopped_any.return_value = False

        res = self._secagg_key_context.setup()

        self.assertTrue(res)
        self.assertIsInstance(res, bool)
        # check save from SKManager
        self.mock_skmanager.add.assert_called_with(
            "secagg_id",
            self.parties[1:],
            {"server_key": -24, "biprime": ANY},
            "experiment-id",
        )

        # Test request error
        self.mock_federated_request.errors.return_value = {"Error": "error"}
        with self.assertRaises(FedbiomedError):
            self._secagg_key_context.setup()

        # test if context is already in db
        self.mock_skmanager.get.return_value = {
            "context": {"server_key": 123, "birpime": 1234},
            "parties": ["party2", "party3"],
        }
        self._secagg_key_context.setup()
        self.assertEqual(self._secagg_key_context.context["server_key"], 123)

    def test_03_saving_loading_breakpoint(self):  # pylint: disable=missing-docstring
        same_obj_attr = (
            "_secagg_id",
            "_parties",
            "_researcher_id",
            "_status",
            "_context",
            "_experiment_id",
            "_element",
        )
        same_inst_attr = (
            "_v",
            "_secagg_manager",
            "_requests",
        )
        # before running round_specific
        state = self._secagg_key_context.save_state_breakpoint()

        self.assertDictContainsSubset(
            {
                "_status": False,
                "_context": None,
            },
            state["attributes"],
        )
        state = copy.deepcopy(state)
        loaded_secagg = SecaggServkeyContext.load_state_breakpoint(state)

        self.check_similarities_in_obj(
            self._secagg_key_context, loaded_secagg, same_obj_attr, same_inst_attr
        )

        # Set request replies
        r1 = copy.deepcopy(self.reply)
        r2 = copy.deepcopy(self.reply)
        r2.node_id = "party2"
        self.mock_federated_request.replies.return_value = {"party1": r1, "party2": r2}
        self.mock_federated_request.errors.return_value = None
        type(
            self.mock_federated_request
        ).policy.return_value.has_stopped_any.return_value = False

        self._secagg_key_context.setup()

        state = self._secagg_key_context.save_state_breakpoint()
        state = copy.deepcopy(state)

        loaded_secagg = SecaggServkeyContext.load_state_breakpoint(state)

        self.check_similarities_in_obj(
            self._secagg_key_context, loaded_secagg, same_obj_attr, same_inst_attr
        )

    def check_similarities_in_obj(self, obj1, obj2, same_obj_attr, same_instance_attr):  # pylint: disable=missing-docstring
        for attr in obj1.__dict__:
            attr1, attr2 = getattr(obj1, attr), getattr(obj2, attr)
            if attr in same_obj_attr and attr1 != attr2:
                self.assertFalse(True, f"{attr1} and {attr2} are not equal")
            elif attr in same_instance_attr:
                self.assertIsInstance(attr1, type(attr2))


class TestSecaggDHContext(BaseTestCaseSecaggContext):  # pylint: disable=missing-docstring
    def setUp(self) -> None:
        super().setUp()

        self.dhmanager_p = patch(
            "fedbiomed.researcher.secagg._secagg_context.SecaggDhManager"
        )
        self.dhmanager_p_s = self.dhmanager_p.start()
        self.dhmanager = MagicMock()
        self.dhmanager_p_s.return_value = self.dhmanager

        self.secagg_dhcontext = SecaggDHContext(
            researcher_id=test_id,
            parties=["party2", "party3"],
            experiment_id="",
            secagg_id="secagg_id",
        )

    def tearDown(self) -> None:
        super().tearDown()
        self.dhmanager_p.stop()

    def test_01_dhcontext_init_error_cases(self):
        with self.assertRaises(FedbiomedError):
            SecaggDHContext(
                researcher_id=test_id,
                parties=["party2"],
                experiment_id="exp_id",
                secagg_id="secagg_id",
            )

    def test_03_dh_context_secagg_setup(self):
        self.mock_federated_request.replies.return_value = {
            "party1": SecaggReply(
                **{
                    "researcher_id": "xx",
                    "node_id": "party2",
                    "msg": "x",
                    "success": True,
                    "secagg_id": "s1",
                }
            ),
            "party2": SecaggReply(
                **{
                    "researcher_id": "xx",
                    "node_id": "party3",
                    "msg": "x",
                    "success": True,
                    "secagg_id": "s1",
                }
            ),
        }

        self.dhmanager.get.return_value = {
            "context": {},
            "parties": ["party2", "party3"],
        }

        self.mock_federated_request.errors.return_value = None
        type(
            self.mock_federated_request
        ).policy.return_value.has_stopped_any.return_value = False

        result = self.secagg_dhcontext.setup()
        self.assertTrue(result)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
