import builtins
import copy
import inspect
import os
import tempfile
import unittest
from unittest.mock import patch

from tinydb.table import Document

from fedbiomed.common.constants import (
    TrainingPlanApprovalStatus,
    TrainingPlanStatus,
)
from fedbiomed.common.exceptions import FedbiomedTrainingPlanSecurityManagerError
from fedbiomed.common.message import ApprovalRequest, TrainingPlanStatusRequest
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager


class TestTrainingPlanSecurityManager(unittest.TestCase):
    """
    Unit tests for class `TrainingPlanSecurityManager` (from fedbiomed.node.training_plan_security_manager)
    """

    # Raise an arbitrary error (eg: SystemError), whatever the conditions and parameters.
    # Used for mocking, when you want to have an error on some function.
    raise_some_error = SystemError("my error message")

    # dummy class for testing typing of parameters
    class Dummy:
        pass

    # before the tests
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp_dir.name, "test-db.json")

        # DB patch start
        self.db_patch = patch(
            "fedbiomed.node.training_plan_security_manager.DBTable", autospec=True
        )

        self.db_mock = self.db_patch.start()

        # Build TrainingPlanSecurityManager
        self.tp_security_manager = TrainingPlanSecurityManager(
            db=self.db,
            node_id="test-id",
            node_name="test-name",
            hashing="SHA256",
            tp_approval=True,
        )

        # get test directory to access test-training plan files
        self.testdir = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
            "test-training-plan",
        )

    # after the tests
    def tearDown(self):
        # DB should be removed after each test to
        # have clear database for tests

        self.db_patch.stop()
        self.db_mock.reset_mock()
        self.tp_security_manager._tinydb.drop_table("TrainingPlans")
        self.tp_security_manager._tinydb.close()
        self.temp_dir.cleanup()

    def test_training_plan_manager_01_create_training_plan_hash(self):
        """Files and strings use the same fixed SHA256 digest."""
        path = os.path.join(self.testdir, "test-training-plan-1.txt")
        digest, algorithm, source = self.tp_security_manager._create_hash(path)
        self.assertEqual(algorithm, "SHA256")
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            self.tp_security_manager._create_hash(source, from_string=True),
            (digest, "SHA256", source),
        )

    def test_training_plan_manager_02_create_hash_hashing_exception(self):
        """Tests `create_hash` method is raising exception if hashing
        algorithm does not exist"""
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            TrainingPlanSecurityManager(
                db=self.db,
                node_id="test-id",
                node_name="test-name",
                hashing="UNKNOWN_HASHING_ALGO",
            )

    def test_training_plan_manager_03_create_hash_open_exceptions(self):
        """Tests `create_hash` method is raising appropriate exception if
        cannot open and read training_plan file (test try/catch blocks when opening
        a file)
        """
        training_plan_path = os.path.join(self.testdir, "test-training-plan-1.txt")
        # test 1 : test case where training plan file has not been found

        # action
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager._create_hash(
                "a/path/that/should/not/exist/on/your/computer"
            )

        # test 2: test case where training plan file cannot be read (due to a lack of privilege)
        with patch.object(builtins, "open") as builtin_open_mock:
            builtin_open_mock.side_effect = PermissionError(
                "mimicking a PermissionError"
            )
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._create_hash(training_plan_path)

        # test 3: test where training plan file cannot be open and read
        with patch.object(builtins, "open") as builtin_open_mock:
            builtin_open_mock.side_effect = OSError("mimicking a OSError")

            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._create_hash(training_plan_path)

    @patch("fedbiomed.node.training_plan_security_manager.minify")
    def test_training_plan_manager_04_create_hash_minify_exception(self, minify_patch):
        """Tests that `_create_hash` method is catching exception coming
        from `minify` package"""
        training_plan_path = os.path.join(self.testdir, "test-training-plan-1.txt")

        minify_patch.side_effect = Exception(
            "Mimicking an Exception triggered by `minify` package"
        )

        # action
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager._create_hash(training_plan_path)

    def test_training_plan_manager_05_create_hash_param_exception(self):
        """ "Tests `_create_patch` with incorrect parameters"""

        for mpath in [
            None,
            18,
            -2.4,
            True,
            {},
            {"clef": "valeur"},
            [],
            ["un"],
            self.Dummy,
            self.Dummy(),
        ]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._create_hash(mpath)

    @patch(
        "fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager._check_training_plan_not_existing",
        autospec=True,
    )
    def test_training_plan_manager_10_register_training_plan(
        self, check_training_plan_not_existing
    ):
        """Testing registering method for new training plans"""

        training_plan_file_1 = os.path.join(self.testdir, "test-training-plan-1.txt")
        training_plan_file_2 = os.path.join(self.testdir, "test-training-plan-2.txt")

        self.tp_security_manager.register_training_plan(
            name="test-training-plan",
            path=training_plan_file_1,
            training_plan_type="registered",
            description="desc",
        )

        # Wrong training plan type
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_training_plan(
                name="tesasdsad",
                path=training_plan_file_2,
                training_plan_type="123123",
                description="desc",
            )

        self.db_mock.return_value.insert.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_training_plan(
                name="tesasdsad",
                path=training_plan_file_2,
                training_plan_type="registered",
                description="desc",
            )
        self.db_mock.return_value.insert.side_effect = Exception

    @patch(
        "fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager._check_training_plan_not_existing",
        autospec=True,
    )
    def test_training_plan_manager_15_update_training_plan(
        self, check_training_plan_not_existing
    ):
        """Tests method `update_training_plan_hash` in the normal case scenario"""

        self.db_mock.return_value.get.return_value = {
            "training_plan": "test",
            "training_plan_id": "test-training-plan-id",
            "hash": "dummy-hash",
            "training_plan_type": "registered",
        }
        # action
        training_plan_file_2 = os.path.join(self.testdir, "test-training-plan-2.txt")
        self.tp_security_manager.update_training_plan_hash(
            "test-training-plan-id", training_plan_file_2
        )

        # Back to normal
        self.db_mock.return_value.get.return_value = {
            "training_plan": "test",
            "training_plan_id": "test-training-plan-id",
            "hash": "dummy-hash",
            "training_plan_type": "registered",
        }

        # Db get raises
        self.db_mock.return_value.get.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.update_training_plan_hash(
                "test-training-plan-id", training_plan_file_2
            )
        self.db_mock.return_value.get.side_effect = None

        # Db update raises
        self.db_mock.return_value.update.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.update_training_plan_hash(
                "test-training-plan-id", training_plan_file_2
            )
        self.db_mock.return_value.update.side_effect = None

    def test_training_plan_manager_18_delete_training_plan(self):
        """Testing delete operation for training plan manager"""

        tp = {"training_plan_type": "registered"}
        self.db_mock.return_value.get.return_value = (tp, Document(tp, doc_id=1))
        self.tp_security_manager.delete_training_plan("dummy_id")

        # Remove raises error
        self.db_mock.return_value.remove.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.delete_training_plan("dummy_id")
        self.db_mock.return_value.remove.side_effect = None

        # If there is no tp
        self.db_mock.return_value.get.return_value = None
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.delete_training_plan("dummy_id")

    def test_training_plan_manager_20_list_training_plans(self):
        """Testing list method of training plan manager for correct request cases"""

        self.db_mock.return_value.search.return_value = [
            {"training_plan_name": "dummy-plan", "training_plan": "source_code"},
            {"training_plan_name": "dummy_plan-2", "training_plan": "source_code"},
        ]

        self.tp_security_manager.list_training_plans(
            select_status=[TrainingPlanApprovalStatus.PENDING], verbose=False
        )

        self.tp_security_manager.list_training_plans(
            select_status=[TrainingPlanApprovalStatus.PENDING],
            search={"by": "training_plan_name", "text": "dummy"},
            verbose=False,
        )

        self.tp_security_manager.list_training_plans(
            select_status=[TrainingPlanApprovalStatus.PENDING],
            search={"by": "training_plan_name", "text": "dummy"},
            sort_by="training_plan_name",
            verbose=False,
        )

        self.tp_security_manager.list_training_plans(
            select_status=[TrainingPlanApprovalStatus.PENDING],
            search={"by": "training_plan_name", "text": "dummy"},
            sort_by="training_plan_name",
            verbose=True,
        )

    def test_training_plan_manager_22_reply_training_plan_status_request(self):
        """Tests training plan manager `reply_training_plan_status_request` method (normal case scenarii)"""

        request = TrainingPlanStatusRequest(
            researcher_id="ssss",
            request_id="request_id",
            experiment_id="xxx",
            training_plan="class TestTrainingPlan:\n\tpass",
        )

        self.db_mock.return_value.get.return_value = {
            "training_plan_status": "approved",
            "training_plan": "source_code",
            "training_plan_id": "id",
        }
        result = self.tp_security_manager.reply_training_plan_status_request(request)
        self.assertEqual(result.get_param("status"), "approved")

        self.db_mock.return_value.get.return_value = {
            "training_plan_status": "rejected",
            "training_plan": "source_code",
            "training_plan_id": "id",
        }
        result = self.tp_security_manager.reply_training_plan_status_request(request)
        self.assertEqual(result.get_param("status"), "rejected")

        self.db_mock.return_value.get.return_value = {
            "training_plan_status": "pending",
            "training_plan": "source_code",
            "training_plan_id": "id",
        }
        result = self.tp_security_manager.reply_training_plan_status_request(request)
        self.assertEqual(result.get_param("status"), "pending")

        self.tp_security_manager._tp_approval = False
        result = self.tp_security_manager.reply_training_plan_status_request(request)
        self.assertEqual(result.get_param("approval_obligation"), False)

        self.db_mock.return_value.get.return_value = None
        result = self.tp_security_manager.reply_training_plan_status_request(request)
        self.assertEqual(result.get_param("status"), "Not Registered")

        self.db_mock.return_value.get.side_effect = Exception
        result = self.tp_security_manager.reply_training_plan_status_request(request)
        self.assertEqual(result.get_param("status"), "Error")
        self.db_mock.return_value.get.side_effect = None

    def test_training_plan_manager_17_check_training_plan_not_existing(self):
        """Test `_check_training_plan_not_existing` function"""
        # note: if _check_training_plan_not_existing succeeds, it returns/changes nothing,
        # so we don't have to do a `assert` in this case

        self.db_mock.return_value.get.return_value = None
        result = self.tp_security_manager._check_training_plan_not_existing(
            name="t", hash_="t", algorithm="SHA256"
        )
        self.assertIsNone(result)

        self.db_mock.return_value.get.return_value = None
        result = self.tp_security_manager._check_training_plan_not_existing(
            name=None, hash_="t", algorithm="SHA256"
        )
        self.assertIsNone(result)

        self.db_mock.return_value.get.return_value = None
        result = self.tp_security_manager._check_training_plan_not_existing(
            name=None, hash_=None, algorithm="SHA256"
        )
        self.assertIsNone(result)

        self.db_mock.return_value.get.return_value = {"name": "test"}
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            result = self.tp_security_manager._check_training_plan_not_existing(
                name=None, hash_=None, algorithm="SHA256"
            )

        self.db_mock.return_value.get.return_value = {"name": "test"}
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            result = self.tp_security_manager._check_training_plan_not_existing(
                name="test", hash_=None, algorithm=None
            )

    def test_training_plan_manager_24_check_training_plan_status(self):
        """Test `check_training_plan_status` function"""

        tp = "class TrainingPlan:\n\tpass"
        self.db_mock.return_value.get.return_value = {"name": "t"}
        s, t = self.tp_security_manager.check_training_plan_status(
            tp, TrainingPlanApprovalStatus.REJECTED
        )
        self.assertTrue(s)

        s, t = self.tp_security_manager.check_training_plan_status(tp, None)
        self.assertTrue(s)

        s, t = self.tp_security_manager.check_training_plan_status(
            tp, TrainingPlanStatus.REGISTERED
        )
        self.assertTrue(s)

        self.db_mock.return_value.get.return_value = None
        s, t = self.tp_security_manager.check_training_plan_status(
            tp, TrainingPlanStatus.REGISTERED
        )
        self.assertFalse(s)

    def test_training_plan_manager_25_get_training_plan_by_name(self):
        """Test `get_training_plan_by_name` function"""

        self.db_mock.return_value.get.return_value = None
        s = self.tp_security_manager.get_training_plan_by_name("name")
        self.assertIsNone(s)

    def test_training_plan_manager_26_get_training_plan_from_database(self):
        """Test `get_training_plan_from_database` function"""

        tp = "class TrainingPlan:\n\tpass"
        self.db_mock.return_value.get.return_value = {"name": "tp"}
        s = self.tp_security_manager.get_training_plan_from_database(tp)
        self.assertDictEqual(s, {"name": "tp"})

    def test_training_plan_manager_27_get_training_plan_by_id(self):
        """Test `get_training_plan_by_id` function"""

        tp = "class TrainingPlan:\n\tpass"
        self.db_mock.return_value.get.return_value = {
            "name": "tp",
            "training_plan": "test",
            "hash": "hash",
            "date_modified": 1,
            "date_created": 2,
        }

        s = self.tp_security_manager.get_training_plan_by_id(tp)
        self.assertEqual(s["name"], "tp")

        s = self.tp_security_manager.get_training_plan_by_id(tp, True)
        self.assertEqual(s["name"], "tp")

    def test_training_plan_manager_28_reply_training_plan_approval_request(self):
        """Test training plan manager `reply_training_plan_approval_request` function."""

        request = ApprovalRequest(
            researcher_id="r-1",
            request_id="req-id",
            training_plan="class TrainingPlan:\n\tpass",
            description="test",
        )

        self.db_mock.return_value.get.return_value = {"name": "tp"}
        result = self.tp_security_manager.reply_training_plan_approval_request(request)

        self.db_mock.return_value.get.return_value = None
        result = self.tp_security_manager.reply_training_plan_approval_request(request)

        self.db_mock.return_value.upsert.side_effect = Exception
        result = self.tp_security_manager.reply_training_plan_approval_request(request)
        self.assertFalse(result.get_param("success"))
        self.db_mock.return_value.upsert.side_effect = None

        # Invalid TP code
        request.training_plan = "class TrainingPlan"
        result = self.tp_security_manager.reply_training_plan_approval_request(request)
        self.assertFalse(result.get_param("success"))

        request.training_plan = "class TrainingPlan:\n\tpass"
        self.db_mock.return_value.get.side_effect = [None, {"name": "tp"}]
        result = self.tp_security_manager.reply_training_plan_approval_request(request)
        self.assertTrue(result.get_param("success"))

        self.db_mock.return_value.get.side_effect = [None, None, {"name": "tp"}]
        result = self.tp_security_manager.reply_training_plan_approval_request(request)
        self.assertTrue(result.get_param("success"))

        self.db_mock.return_value.get.side_effect = [None, None, None]
        result = self.tp_security_manager.reply_training_plan_approval_request(request)
        self.assertTrue(result.get_param("success"))
        self.db_mock.return_value.get.side_effect = None

    def test_training_plan_manager_29_update_training_plan_status(self):
        """Test training plan manager `_update_training_plan_status` function."""

        self.db_mock.return_value.get.return_value = None
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager._update_training_plan_status(
                "id", TrainingPlanApprovalStatus.REJECTED, "new_notes"
            )

        self.db_mock.return_value.get.return_value = {
            "training_plan_status": "rejected"
        }
        result = self.tp_security_manager._update_training_plan_status(
            "id", TrainingPlanApprovalStatus.REJECTED, "new_notes"
        )
        self.assertTrue(result)

        self.db_mock.return_value.get.return_value = {"training_plan_status": "approve"}
        result = self.tp_security_manager._update_training_plan_status(
            "id", TrainingPlanApprovalStatus.REJECTED, "new_notes"
        )
        self.assertTrue(result)

    def test_training_plan_manager_30_remove_sensible_keys_from_request(self):
        """Test training plan manager `_remove_sensible_keys_from_request` function."""

        # prepare
        key_sensible = "hash"
        key_notsensible = "training_plan_id"

        doc = {key_sensible: "valeur clef", key_notsensible: "autre valeur"}
        doc2 = copy.deepcopy(doc)

        # test
        self.tp_security_manager._remove_sensible_keys_from_request(doc2)

        # check
        self.assertEqual(doc2[key_notsensible], doc[key_notsensible])
        self.assertFalse(key_sensible in doc2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
