import builtins
import copy
import inspect
import os
import tempfile
import unittest
from unittest.mock import patch

import pytest
from tinydb.table import Document

from fedbiomed.common.constants import (
    HashingAlgorithms,
    TrainingPlanApprovalStatus,
    TrainingPlanStatus,
)
from fedbiomed.common.exceptions import FedbiomedTrainingPlanSecurityManagerError
from fedbiomed.common.message import ApprovalRequest, TrainingPlanStatusRequest
from fedbiomed.common.utils import SHARE_DIR
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

    def test_training_plan_manager_01_create_default_training_plan_hashes(self):
        """Testing whether created hash for training plan files are okay
        or not. It also tests every default with each provided hashing algorithm
        """
        dtp = os.path.join(SHARE_DIR, "envs", "common", "default_training_plans")
        default_training_plans = os.listdir(dtp)
        for training_plan in default_training_plans:
            # set default hashing algorithm

            full_path = os.path.join(dtp, training_plan)

            # Control return values with default hashing algorithm
            hash, algorithm, _ = self.tp_security_manager._create_hash(full_path)
            self.assertIsInstance(hash, str, "Hash creation is not successful")
            self.assertEqual(algorithm, "SHA256", "Wrong hashing algorithm")

            algorithms = HashingAlgorithms.list()
            for algo in algorithms:
                tpsm = TrainingPlanSecurityManager(
                    db=self.db,
                    node_id="test-id",
                    node_name="test-name",
                    hashing=algo,
                )
                hash, algorithm, _ = tpsm._create_hash(full_path)
                self.assertIsInstance(hash, str, "Hash creation is not successful")
                self.assertEqual(algorithm, algo, "Wrong hashing algorithm")

    def test_training_plan_manager_02_create_hash_hashing_exception(self):
        """Tests `create_hash` method is raising exception if hashing
        algorithm does not exist"""
        training_plan_path = os.path.join(self.testdir, "test-training-plan-1.txt")
        tpsm = TrainingPlanSecurityManager(
            db=self.db,
            node_id="test-id",
            node_name="test-name",
            hashing="UNKNOWN_HASHING_ALGO",
        )

        # action:
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            tpsm._create_hash(training_plan_path)

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
        "fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.register_training_plan",
        autospec=True,
    )
    @patch(
        "fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager._check_training_plan_not_existing",
        autospec=True,
    )
    def test_training_plan_manager_06_update_default_hashes_when_algo_is_changed(
        self,
        check_training_not_existing,
        register_training_plan,
    ):
        """Testing method for update/register default training plans when hashing
        algorithm has changed
        """

        # Single test with default hash algorithm
        self.db_mock.return_value.search.return_value = []
        self.tp_security_manager.register_update_default_training_plans()

        # If there training plans registered already
        self.db_mock.return_value.search.return_value = [
            {
                "name": "pytorch-mnist.txt",
                "algorithm": "SHA51211",
            }
        ]
        self.db_mock.return_value.get.return_value = {
            "name": "pytorch-mnist.txt",
            "algorithm": "SHA512",
            "training_plan_id": "test-id",
        }
        self.tp_security_manager.register_update_default_training_plans()

        # If search raises an exception
        self.db_mock.return_value.search.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_update_default_training_plans()
        self.db_mock.return_value.search.side_effect = None

        # Test deleted default training plans
        self.db_mock.return_value.search.return_value = [
            {
                "name": "pytorch-deleted-mnist.txt",
                "algorithm": "SHA51211",
            }
        ]
        self.db_mock.return_value.get.return_value = (
            "",
            Document(
                {
                    "name": "pytorch-deleted-mnist.txt",
                    "algorithm": "SHA512",
                    "training_plan_id": "test-id",
                },
                doc_id=1,
            ),
        )
        self.tp_security_manager.register_update_default_training_plans()

        # If get raises exception
        self.db_mock.return_value.remove.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_update_default_training_plans()
        self.db_mock.return_value.remove.side_effect = None

        # Test if hashing algorithm is different
        self.db_mock.return_value.search.return_value = [
            {
                "name": "pytorch-mnist.txt",
                "algorithm": "SHA51211",
            }
        ]
        self.db_mock.return_value.get.return_value = {
            "name": "pytorch-mnist.txt",
            "algorithm": "SHA512",
            "training_plan_id": "test-id",
        }
        self.tp_security_manager.register_update_default_training_plans()

        # If get raises an exception
        self.db_mock.return_value.get.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_update_default_training_plans()
        self.db_mock.return_value.get.side_effect = None

        # Test if creation date is different
        self.db_mock.return_value.search.return_value = [
            {
                "name": "pytorch-mnist.txt",
                "algorithm": "SHA256",
            }
        ]
        self.db_mock.return_value.get.return_value = {
            "name": "pytorch-mnist.txt",
            "algorithm": "SHA256",
            "training_plan_id": "test-id",
            "hash": "dummy",
            "date_modified": "12-12-1999 12:12:12.123",
        }
        self.tp_security_manager.register_update_default_training_plans()

        # If update raises an error
        self.db_mock.return_value.update.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_update_default_training_plans()
        self.db_mock.return_value.update.side_effect = None

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
        "fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager._create_hash"
    )
    def test_training_plan_manager_12_check_hashes_for_registerd_training_plans(
        self, create_hash_patch
    ):
        """Tests `hashes_for_registered_training_plans` method, with 3 settings"""

        # patch
        def create_hash_side_effect(path, from_string=False):
            return f"a correct unique hash-{path}", "a correct hashing algo", None

        create_hash_patch.side_effect = create_hash_side_effect

        # DB search return value
        self.db_mock.return_value.search.return_value = ([], [])

        # test 1: case where there is no training plan registered
        self.tp_security_manager.check_hashes_for_registered_training_plans()

        # test 2: case where default training plans have been registered
        # we will check that training plans have correct hashes and hashing algorithm
        # in the database
        self.db_mock.return_value.search.return_value = []
        self.db_mock.return_value.get.return_value = None
        self.tp_security_manager.register_update_default_training_plans()

        self.db_mock.return_value.search.return_value = (
            [
                {
                    "name": "test-1",
                    "training_plan": "test-source-1",
                    "training_plan_id": "tp-id-1",
                    "algorithm": "oops",
                },
                {
                    "name": "test-2",
                    "training_plan": "test-source-2",
                    "training_plan_id": "tp-id-2",
                    "algorithm": "oops",
                },
            ],
            ["Unused", "Unused"],
        )

        self.tp_security_manager.check_hashes_for_registered_training_plans()

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.db_mock.return_value.update.side_effect = Exception
            self.tp_security_manager.check_hashes_for_registered_training_plans()
            self.db_mock.return_value.update.side_effect = None

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.db_mock.return_value.search.side_effect = Exception
            self.tp_security_manager.check_hashes_for_registered_training_plans()
            self.db_mock.return_value.search.side_effect = None

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
        default_training_plan_file_2 = os.path.join(
            self.testdir, "test-training-plan-2.txt"
        )
        self.tp_security_manager.update_training_plan_hash(
            "test-training-plan-id", default_training_plan_file_2
        )

        # default training plan
        self.db_mock.return_value.get.return_value = {
            "training_plan": "test",
            "training_plan_id": "test-training-plan-id",
            "hash": "dummy-hash",
            "training_plan_type": "default",
        }
        # action
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            default_training_plan_file_2 = os.path.join(
                self.testdir, "test-training-plan-2.txt"
            )
            self.tp_security_manager.update_training_plan_hash(
                "test-training-plan-id", default_training_plan_file_2
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
                "test-training-plan-id", default_training_plan_file_2
            )
        self.db_mock.return_value.get.side_effect = None

        # Db update raises
        self.db_mock.return_value.update.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.update_training_plan_hash(
                "test-training-plan-id", default_training_plan_file_2
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

        tp = {"training_plan_type": "default"}
        self.db_mock.return_value.get.return_value = (tp, Document(tp, doc_id=1))
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


# ============ PYTEST STYLE TESTS FOR SECURITY LOGGING ============


@pytest.fixture
def temp_db_pytest():
    """Pytest fixture for temporary database."""
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, "test-db.json")
    yield db_path, temp_dir
    temp_dir.cleanup()


@pytest.fixture
def tp_security_manager_pytest(temp_db_pytest):
    """Pytest fixture for TrainingPlanSecurityManager with I/O-free hashing + mocked logging.

    Notes:
    - Do NOT patch `builtins.open` globally: TinyDB uses it for the DB JSON file.
    - Instead, stub `_create_hash` to avoid reading training plan files.
    - Mock `logger` in the module under test to avoid side effects (eg. system info collection).
    """

    import hashlib

    db_path, _ = temp_db_pytest

    with (
        patch("fedbiomed.node.training_plan_security_manager.logger") as mock_logger,
        patch(
            "fedbiomed.node.training_plan_security_manager.os.path.getctime",
            return_value=1000000,
        ),
        patch(
            "fedbiomed.node.training_plan_security_manager.os.path.getmtime",
            return_value=1000000,
        ),
        patch.object(
            TrainingPlanSecurityManager,
            "_create_hash",
            autospec=True,
        ) as mock_create_hash,
    ):

        def _create_hash_side_effect(_self, source: str, from_string: bool = False):
            # When hashing a file path, avoid touching the filesystem.
            # When hashing source code (from_string=True), keep determinism based on content.
            if from_string:
                content = source
                source_out = source
            else:
                content = f"mock_tp_source:{source}"
                source_out = content

            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            return digest, "SHA256", source_out

        mock_create_hash.side_effect = _create_hash_side_effect

        manager = TrainingPlanSecurityManager(
            db=db_path,
            node_id="pytest-node-id",
            node_name="pytest-node-name",
            hashing="SHA256",
            tp_approval=True,
        )

        # Attach mocks to manager for easy access in tests
        manager.mock_security_event = mock_logger.security_event
        manager.mock_logger = mock_logger
        manager.mock_create_hash = mock_create_hash

        yield manager

        manager._tinydb.drop_table("TrainingPlans")
        manager._tinydb.close()


@pytest.fixture
def test_training_plan_source():
    """Pytest fixture for test training plan source code."""
    return """
from fedbiomed.common.training_plans import TorchTrainingPlan

class TestTrainingPlan(TorchTrainingPlan):
    def init_model(self, model_args):
        return None

    def training_step(self, batch):
        pass
"""


def test_security_log_register_training_plan_success(tp_security_manager_pytest):
    """Pytest: Verify training plan registration logs security event."""
    manager = tp_security_manager_pytest

    tp_id = manager.register_training_plan(
        name="pytest_tp",
        description="Pytest training plan",
        path="/mock/path/tp.py",
        researcher_id="pytest_researcher",
    )

    # Verify security_event call
    manager.mock_security_event.assert_called()
    call_kwargs = manager.mock_security_event.call_args.kwargs

    assert call_kwargs["operation"] == "training_plan_register"
    assert call_kwargs["status"] == "success"
    assert call_kwargs["node_id"] == "pytest-node-id"
    assert call_kwargs["node_name"] == "pytest-node-name"
    assert call_kwargs["researcher_id"] == "pytest_researcher"
    assert call_kwargs["training_plan_id"] == tp_id
    assert call_kwargs["training_plan_name"] == "pytest_tp"


def test_security_log_delete_training_plan_success(tp_security_manager_pytest):
    """Pytest: Verify training plan deletion logs security event."""
    manager = tp_security_manager_pytest

    # Register first
    tp_id = manager.register_training_plan(
        name="pytest_tp_del",
        description="Pytest training plan for deletion",
        path="/mock/path/tp_del.py",
        researcher_id="pytest_researcher",
    )

    manager.mock_security_event.reset_mock()

    # Delete
    manager.delete_training_plan(tp_id)

    # Verify security_event call
    manager.mock_security_event.assert_called()
    call_kwargs = manager.mock_security_event.call_args.kwargs

    assert call_kwargs["operation"] == "training_plan_delete"
    assert call_kwargs["status"] == "success"
    assert call_kwargs["node_id"] == "pytest-node-id"
    assert call_kwargs["node_name"] == "pytest-node-name"
    assert call_kwargs["training_plan_id"] == tp_id


def test_security_log_approve_training_plan(
    tp_security_manager_pytest, test_training_plan_source
):
    """Pytest: Verify training plan approval logs security event."""
    manager = tp_security_manager_pytest

    # Create a PENDING training plan first (registration via CLI creates APPROVED by default)
    request = ApprovalRequest(
        researcher_id="pytest_researcher",
        request_id="pytest_req_approve_001",
        training_plan=test_training_plan_source,
        description="Pytest approval",
    )
    manager.reply_training_plan_approval_request(request)

    tp_info = manager.get_training_plan_from_database(test_training_plan_source)
    assert tp_info is not None
    tp_id = tp_info["training_plan_id"]

    manager.mock_security_event.reset_mock()

    # Approve
    manager.approve_training_plan(tp_id, extra_notes="Approved in pytest")

    # Verify security_event call
    manager.mock_security_event.assert_called()
    call_kwargs = manager.mock_security_event.call_args.kwargs

    assert call_kwargs["operation"] == "training_plan_status_update"
    assert call_kwargs["status"] == "success"
    assert call_kwargs["node_id"] == "pytest-node-id"
    assert call_kwargs["node_name"] == "pytest-node-name"
    assert call_kwargs["new_status"] == TrainingPlanApprovalStatus.APPROVED.value


def test_security_log_reject_training_plan(tp_security_manager_pytest):
    """Pytest: Verify training plan rejection logs security event."""
    manager = tp_security_manager_pytest

    # Register first
    tp_id = manager.register_training_plan(
        name="pytest_tp_reject",
        description="Pytest training plan for rejection",
        path="/mock/path/tp_reject.py",
        researcher_id="pytest_researcher",
    )

    manager.mock_security_event.reset_mock()

    # Reject
    manager.reject_training_plan(tp_id, extra_notes="Rejected in pytest")

    # Verify security_event call
    manager.mock_security_event.assert_called()
    call_kwargs = manager.mock_security_event.call_args.kwargs

    assert call_kwargs["operation"] == "training_plan_status_update"
    assert call_kwargs["status"] == "success"
    assert call_kwargs["new_status"] == TrainingPlanApprovalStatus.REJECTED.value


def test_security_log_approval_request_received(
    tp_security_manager_pytest, test_training_plan_source
):
    """Pytest: Verify approval request logs security event with request_id."""
    manager = tp_security_manager_pytest
    request = ApprovalRequest(
        researcher_id="pytest_researcher",
        request_id="pytest_req_001",
        training_plan=test_training_plan_source,
        description="Pytest approval request",
    )

    manager.reply_training_plan_approval_request(request)

    # Verify security_event call
    manager.mock_security_event.assert_called()
    call_kwargs = manager.mock_security_event.call_args.kwargs

    assert call_kwargs["operation"] == "training_plan_approval_request_received"
    assert call_kwargs["status"] == "success"
    assert call_kwargs["node_id"] == "pytest-node-id"
    assert call_kwargs["node_name"] == "pytest-node-name"
    assert call_kwargs["researcher_id"] == "pytest_researcher"
    assert call_kwargs["request_id"] == "pytest_req_001"
    assert (
        call_kwargs["training_plan_status"] == TrainingPlanApprovalStatus.PENDING.value
    )


def test_security_log_status_request(
    tp_security_manager_pytest, test_training_plan_source
):
    """Pytest: Verify status request logs security event with experiment_id."""
    manager = tp_security_manager_pytest
    # Register and approve first
    request = ApprovalRequest(
        researcher_id="pytest_researcher",
        request_id="pytest_req_001",
        training_plan=test_training_plan_source,
        description="Pytest approval",
    )

    manager.reply_training_plan_approval_request(request)
    tp_info = manager.get_training_plan_from_database(test_training_plan_source)
    manager.approve_training_plan(tp_info["training_plan_id"])

    manager.mock_security_event.reset_mock()

    # Send status request
    status_request = TrainingPlanStatusRequest(
        researcher_id="pytest_researcher",
        request_id="pytest_status_001",
        training_plan=test_training_plan_source,
        experiment_id="pytest_exp_001",
    )

    manager.reply_training_plan_status_request(status_request)

    # Verify security_event call
    manager.mock_security_event.assert_called()
    call_kwargs = manager.mock_security_event.call_args.kwargs

    assert call_kwargs["operation"] == "training_plan_status_request"
    assert call_kwargs["status"] == "checked"
    assert call_kwargs["node_id"] == "pytest-node-id"
    assert call_kwargs["node_name"] == "pytest-node-name"
    assert call_kwargs["researcher_id"] == "pytest_researcher"
    assert call_kwargs["request_id"] == "pytest_status_001"
    assert call_kwargs["experiment_id"] == "pytest_exp_001"
    assert (
        call_kwargs["training_plan_status"] == TrainingPlanApprovalStatus.APPROVED.value
    )


@pytest.mark.parametrize(
    "status,expected_message",
    [
        ("already_pending", "Training plan already pending approval"),
        ("already_approved", "Training plan already approved"),
    ],
)
def test_security_log_duplicate_approval_requests(
    tp_security_manager_pytest,
    test_training_plan_source,
    status,
    expected_message,
):
    """Pytest: Verify duplicate approval requests log correct status."""
    manager = tp_security_manager_pytest
    # Register and set appropriate status
    request = ApprovalRequest(
        researcher_id="pytest_researcher",
        request_id="pytest_req_001",
        training_plan=test_training_plan_source,
        description="Pytest approval",
    )

    manager.reply_training_plan_approval_request(request)
    tp_info = manager.get_training_plan_from_database(test_training_plan_source)

    if status == "already_approved":
        manager.approve_training_plan(tp_info["training_plan_id"])

    manager.mock_security_event.reset_mock()

    # Send another request
    request2 = ApprovalRequest(
        researcher_id="pytest_researcher_2",
        request_id="pytest_req_002",
        training_plan=test_training_plan_source,
        description="Pytest approval 2",
    )

    manager.reply_training_plan_approval_request(request2)

    # Verify security_event call
    manager.mock_security_event.assert_called()
    call_kwargs = manager.mock_security_event.call_args.kwargs

    assert call_kwargs["operation"] == "training_plan_approval_request_received"
    assert call_kwargs["status"] == status
    assert call_kwargs["node_id"] == "pytest-node-id"
    assert call_kwargs["node_name"] == "pytest-node-name"
    assert call_kwargs["request_id"] == "pytest_req_002"


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
