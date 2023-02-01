import builtins
import copy
from datetime import datetime
import os
import shutil
import tempfile
import unittest
import inspect
from unittest.mock import patch, MagicMock

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################
from fedbiomed.node.environ import environ

from fedbiomed.common.constants import ErrorNumbers, HashingAlgorithms, TrainingPlanApprovalStatus, TrainingPlanStatus
from fedbiomed.common.exceptions import FedbiomedMessageError, \
    FedbiomedTrainingPlanSecurityManagerError, \
    FedbiomedRepositoryError
from fedbiomed.common.logger import logger
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager


class TestTrainingPlanSecurityManager(NodeTestCase):
    """
    Unit tests for class `TrainingPlanSecurityManager` (from fedbiomed.node.training_plan_security_manager)
    """

    # Raise an arbitrary error (eg: SystemError), whatever the conditions and parameters.
    # Used for mocking, when you want to have an error on some function.
    raise_some_error = SystemError('my error message')


    # dummy class for testing typing of parameters
    class Dummy():
        pass

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.env_initial = copy.deepcopy(cls.env)

    # before the tests
    def setUp(self):

        TestTrainingPlanSecurityManager.env = TestTrainingPlanSecurityManager.env_initial

        def side_effect(arg):
            return TestTrainingPlanSecurityManager.env[arg]

        def side_effect_set_item(key, value):
            TestTrainingPlanSecurityManager.env[key] = value

        self.environ_training_plan_manager_patch = patch('fedbiomed.node.training_plan_security_manager.environ')
        self.environ_training_plan = self.environ_training_plan_manager_patch.start()

        self.environ_training_plan.__getitem__.side_effect = side_effect
        self.environ_training_plan.__setitem__.side_effect = side_effect_set_item

        self.patcher_db_get = patch('tinydb.table.Table.get', MagicMock(side_effect=self.raise_some_error))
        self.patcher_db_search = patch('tinydb.table.Table.search', MagicMock(side_effect=self.raise_some_error))
        self.patcher_db_update = patch('tinydb.table.Table.update', MagicMock(side_effect=self.raise_some_error))
        self.patcher_db_remove = patch('tinydb.table.Table.remove', MagicMock(side_effect=self.raise_some_error))
        self.patcher_db_upsert = patch('tinydb.table.Table.upsert', MagicMock(side_effect=self.raise_some_error))
        self.patcher_db_all = patch('tinydb.table.Table.all', MagicMock(side_effect=self.raise_some_error))
        # ---------------------------------------------------------------------------

        # handle case where previous test did not properly clean
        if os.path.exists(environ['DB_PATH']):
            os.remove(environ['DB_PATH'])
        #
        # import nose.tools
        # nose.tools.set_trace()


        # Build TrainingPlanSecurityManager
        self.tp_security_manager = TrainingPlanSecurityManager()

        # get test directory to access test-training plan files
        self.testdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ),
            "test-training-plan"
        )

    # after the tests
    def tearDown(self):
        # DB should be removed after each test to
        # have clear database for tests

        self.environ_training_plan_manager_patch.stop()
        self.tp_security_manager._tinydb.drop_table('TrainingPlans')
        self.tp_security_manager._tinydb.close()
        os.remove(environ['DB_PATH'])

    def test_training_plan_manager_01_create_default_training_plan_hashes(self):

        """ Testing whether created hash for training plan files are okay
        or not. It also tests every default with each provided hashing algorithm
        """

        default_training_plans = os.listdir(environ['DEFAULT_TRAINING_PLANS_DIR'])
        logger.info('Controlling Training plans Dir')
        logger.info(environ['DEFAULT_TRAINING_PLANS_DIR'])
        for training_plan in default_training_plans:

            # set default hashing algorithm
            TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM'] = 'SHA256'
            full_path = os.path.join(environ['DEFAULT_TRAINING_PLANS_DIR'], training_plan)

            # Control return vlaues with default hashing algorithm
            hash, algortihm = self.tp_security_manager._create_hash(full_path)
            self.assertIsInstance(hash, str, 'Hash creation is not successful')
            self.assertEqual(algortihm, 'SHA256', 'Wrong hashing algorithm')

            algorithms = HashingAlgorithms.list()
            for algo in algorithms:
                TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM'] = algo
                hash, algortihm = self.tp_security_manager._create_hash(full_path)
                self.assertIsInstance(hash, str, 'Hash creation is not successful')
                self.assertEqual(algortihm, algo, 'Wrong hashing algorithm')

    def test_training_plan_manager_02_create_hash_hashing_exception(self):
        """Tests `create_hash` method is raising exception if hashing
        algorithm does not exist"""
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
        TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM'] = "AN_UNKNOWN_HASH_ALGORITHM"

        # action:
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager._create_hash(training_plan_path)

        # Back to normal
        TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM'] = "SHA256"

    def test_training_plan_manager_03_create_hash_open_exceptions(self):
        """Tests `create_hash` method is raising appropriate exception if
        cannot open and read training_plan file (test try/catch blocks when opening
        a file)
        """
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
        # test 1 : test case where training plan file has not been found

        # action 
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager._create_hash("a/path/that/should/not/exist/on/your/computer")

        # test 2: test case where training plan file cannot be read (due to a lack of privilege)
        with patch.object(builtins, 'open') as builtin_open_mock:
            builtin_open_mock.side_effect = PermissionError("mimicking a PermissionError")
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._create_hash(training_plan_path)

        # test 3: test where training plan file cannot be open and read
        with patch.object(builtins, 'open') as builtin_open_mock:
            builtin_open_mock.side_effect = OSError("mimicking a OSError")

            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._create_hash(training_plan_path)

    @patch('fedbiomed.node.training_plan_security_manager.minify')
    def test_training_plan_manager_04_create_hash_minify_exception(self, minify_patch):
        """Tests that `_create_hash` method is catching exception coming
        from `minify` package"""
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')

        minify_patch.side_effect = Exception('Mimicking an Exception triggered by `minify` package')

        # action
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager._create_hash(training_plan_path)

    def test_training_plan_manager_05_create_hash_param_exception(self):
        """"Tests `_create_patch` with incorrect parameters"""

        for mpath in [None, 18, -2.4, True, {}, { 'clef': 'valeur' }, [], ['un'], self.Dummy, self.Dummy()]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._create_hash(mpath)

    def test_training_plan_manager_06_update_default_hashes_when_algo_is_changed(self):
        """  Testing method for update/register default training plans when hashing
             algorithm has changed
        """

        # Single test with default hash algorithm
        self.tp_security_manager.register_update_default_training_plans()

        # # Multiple test with different hashing algorithms
        algorithms = HashingAlgorithms.list()
        for algo in algorithms:
            TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM'] = algo
            self.tp_security_manager.register_update_default_training_plans()
            doc = self.tp_security_manager._db.get(self.tp_security_manager._database.training_plan_type == "default")
            logger.info(doc)
            self.assertEqual(doc["algorithm"], algo,
                             'Hashes are not properly updated after hashing algorithm is changed')  # noqa

    def test_training_plan_manager_07_update_default_training_plan_deleted(self):
        """Tests `update_default_training_plan` when a training plans file that had been registered
        has been deleted
        """
        file_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
        new_default_training_plan_path = os.path.join(environ['TMP_DIR'], 'test-training-plan-1-2.txt')
        shutil.copy(file_path, new_default_training_plan_path)

        # update database
        self.tp_security_manager.register_training_plan(
            name='test-training-plan',
            path=new_default_training_plan_path,
            training_plan_type='default',
            description='desc'
        )
        # now, remove copied training plan from system
        os.remove(new_default_training_plan_path)

        # check that training plan is in database
        training_plan = self.tp_security_manager._db.get(self.tp_security_manager._database.training_plan_path == new_default_training_plan_path)
        self.assertIsNotNone(training_plan)

        # action
        self.tp_security_manager.register_update_default_training_plans()

        # check that copied training plan entry has been removed
        removed_training_plan = self.tp_security_manager._db.get(self.tp_security_manager._database.training_plan_path == new_default_training_plan_path)
        self.assertIsNone(removed_training_plan)

    def test_training_plan_manager_08_update_errors(self):
        """Testing error cases in function `register_update_default_training_plans`.
        """

        # prepare: first select a hashing algorithm that is not the one set in ENVIRON
        # (first one that is different in the list)

        algo_initial = TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM']
        for a in HashingAlgorithms.list():
            if a != algo_initial:
                algo_tempo = a
                break

        new_default_training_plan_path = os.path.join(environ['DEFAULT_TRAINING_PLANS_DIR'], 'my_default_training_plan.txt')

        for error in ['db_search', 'db_get-delete', 'db_remove-delete', 'db_get-exists', 'db_update-exists']:

            # prepare
            #
            # some test may manage update database or files, or leave them in an unclean state
            # so we need to prepare at each test

            shutil.copy(os.path.join(self.testdir, 'test-training-plan-1.txt'), new_default_training_plan_path)
            self.tp_security_manager.register_update_default_training_plans()
            training_plans_before = self.tp_security_manager._db.all()

            # test-specific prepare
            if error == 'db_search':
                self.patcher_db_search.start()
            elif error == 'db_get-delete':
                self.patcher_db_get.start()
                os.remove(new_default_training_plan_path)
            elif error == 'db_remove-delete':
                self.patcher_db_remove.start()
                os.remove(new_default_training_plan_path)
            elif error == 'db_get-exists':
                self.patcher_db_get.start()
                TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM'] = algo_tempo
            elif error == 'db_update-exists':
                self.patcher_db_update.start()
                TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM'] = algo_tempo

            # test and check
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.register_update_default_training_plans()

            # test-specific clean
            if error == 'db_search':
                self.patcher_db_search.stop()
            elif error == 'db_get-delete':
                self.patcher_db_get.stop()
            elif error == 'db_remove-delete':
                self.patcher_db_remove.stop()
            elif error == 'db_get-exists':
                self.patcher_db_get.stop()
                TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM'] = algo_initial
            elif error == 'db_update-exists':
                self.patcher_db_update.stop()
                TestTrainingPlanSecurityManager.env['HASHING_ALGORITHM'] = algo_initial

            # check training plans where not modified by failed operation
            training_plans_after = self.tp_security_manager._db.all()
            self.assertEqual(training_plans_before, training_plans_after)

        # clean
        if os.path.exists(new_default_training_plan_path):
            os.remove(new_default_training_plan_path)
        self.tp_security_manager.register_update_default_training_plans()

    def test_training_plan_manager_09_update_modified_training_plan_files(self):
        """ Testing update of modified default training plans """

        default_training_plans = os.listdir(environ['DEFAULT_TRAINING_PLANS_DIR'])

        # Test with only first file

        for training_plan in default_training_plans:
            file_path = os.path.join(environ['DEFAULT_TRAINING_PLANS_DIR'], training_plan)
            self.tp_security_manager.register_update_default_training_plans()
            doc = self.tp_security_manager._db.get(self.tp_security_manager._database.training_plan_path == file_path)

            # Open the file in append & read mode ('a+')
            with open(file_path, "a+") as file:
                lines = file.readlines()  # lines is list of line, each element '...\n'
                lines.insert(0, "\nprint('Hello world') \t \n")  # you can use any index if you know the line index
                file.seek(0)  # file pointer locates at the beginning to write the whole file again
                file.writelines(lines)

            self.tp_security_manager.register_update_default_training_plans()
            docAfter = self.tp_security_manager._db.get(self.tp_security_manager._database.training_plan_path == file_path)

            self.assertNotEqual(doc['hash'], docAfter['hash'], "Hash couldn't updated after file has modified")

    def test_training_plan_manager_10_register_training_plan(self):
        """ Testing registering method for new training plans """

        # We should import environ to get fake values

        self.tp_security_manager.register_update_default_training_plans()

        training_plan_file_1 = os.path.join(self.testdir, 'test-training-plan-1.txt')
        training_plan_file_2 = os.path.join(self.testdir, 'test-training-plan-2.txt')

        self.tp_security_manager.register_training_plan(
            name='test-training-plan',
            path=training_plan_file_1,
            training_plan_type='registered',
            description='desc'
        )

        # When same training plan file wants to be added it should raise and exception
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_training_plan(
                name='test-training-plan-2',
                path=training_plan_file_1,
                training_plan_type='registered',
                description='desc')

        # When same training plan wants to be added with same name  and different file
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_training_plan(
                name='test-training-plan',
                path=training_plan_file_2,
                training_plan_type='registered',
                description='desc')

        # Wrong types
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_training_plan(
                name='test-training-plan-2',
                path=training_plan_file_2,
                training_plan_type=False,
                description='desc')

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_training_plan(
                name='tesasdsad',
                path=training_plan_file_2,
                training_plan_type=False,
                description=False)

        # Wrong training plan type
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_training_plan(
                name='tesasdsad',
                path=training_plan_file_2,
                training_plan_type='123123',
                description='desc')

    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager._check_training_plan_not_existing')
    def test_training_plan_manager_11_register_training_plan_DB_error(self, check_training_plan_not_existing_patch):
        """ Testing registering method for new training plans continued, case where training plan is corrupted"""

        # patch
        check_training_plan_not_existing_patch.value = None

        training_plan_file_2 = os.path.join(self.testdir, 'test-training-plan-2.txt')

        # Cannot access corrupted database
        with open(environ['DB_PATH'], 'w') as f:
            f.write('CORRUPTED DATABASE CONTENT')

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.register_training_plan(
                name='test-training-plan-3',
                path=training_plan_file_2,
                training_plan_type='registered',
                description='desc')        

        with open(environ['DB_PATH'], 'w') as f:
            f.write('') 

    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager._create_hash')
    def test_training_plan_manager_12_check_hashes_for_registerd_training_plans(self,
                                                                create_hash_patch):
        """
        Tests `hashes_for_registered_training_plans` method, with 3 settings
        - Test 1: no training plans are registered
        - Test 2: training plans are registered and are stored on computer
        - Test 3: training plan is no longer stored on computer.
        """
        # patch
        def create_hash_side_effect(path):
            return f'a correct unique hash {path}', 'a correct hashing algo'
        create_hash_patch.side_effect = create_hash_side_effect

        # test 1: case where there is no training plan registered
        self.tp_security_manager.check_hashes_for_registered_training_plans()

        # check that no training plans are not registered
        self.assertListEqual(self.tp_security_manager._db.search(self.tp_security_manager._database.training_plan_type.all('registered')),
                             [])

        # test 2: case where default training plans have been registered
        # we will check that training plans have correct hashes and hashing algorithm
        # in the database
        self.tp_security_manager.register_update_default_training_plans()

        training_plan_file_1_path = os.path.join(self.testdir, 'test-training-plan-1.txt')

        # copying training plan (we will delete it afterward)
        training_plan_file_copied_path = os.path.join(environ['TMP_DIR'], 'copied-test-training-plan-1.txt')
        shutil.copy(training_plan_file_1_path, training_plan_file_copied_path)
        self.tp_security_manager.register_training_plan(
            name='test-training-plan',
            path=training_plan_file_copied_path,
            training_plan_type='registered',
            description='desc',
            training_plan_id='test-training-plan-id'
        )

        # update database with non-existing hash and hashing algorithms
        training_plans = self.tp_security_manager._db.search(self.tp_security_manager._database.training_plan_type.all('registered'))
        for training_plan in training_plans:
            self.tp_security_manager._db.update({'hash': "an incorrect hash",
                                           'algorithm': "incorrect_hashing_algorithm"},
                                          self.tp_security_manager._database.training_plan_id.all(training_plan["training_plan_id"]))

        self.tp_security_manager.check_hashes_for_registered_training_plans()

        # checks
        training_plans = self.tp_security_manager._db.search(self.tp_security_manager._database.training_plan_type.all('registered'))

        for training_plan in training_plans:
            correct_hash, correct_hashing_algo = create_hash_side_effect(training_plan['training_plan_path'])
            self.assertEqual(training_plan['hash'], correct_hash)
            self.assertEqual(training_plan['algorithm'], correct_hashing_algo)

        # Test 3: here we are testing that a file that has been removed on
        # the system is also removed from the database

        # remove the training plan file stored on the system
        # FIXME: should we skip the remaining tests if a PermissionError is triggered
        os.remove(training_plan_file_copied_path)

        # action
        self.tp_security_manager.check_hashes_for_registered_training_plans()
        removed_training_plan = self.tp_security_manager._db.get(self.tp_security_manager._database.name == 'test-training-plan')
        # check that the training plan has been removed
        self.assertIsNone(removed_training_plan)

    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager._create_hash')
    def test_training_plan_manager_13_check_hashes_for_registerd_training_plans_DB_error(self, create_hash_patch):
        """
        Tests `hashes_for_registered_training_plans` method, with 3 settings, causing database access errors
        - Test 1: no training plan registered, cannot read database
        - Test 2: one training plan registered, but cannot update database
        - Test 3: training plan is no longer stored on computer.
        """
        # patch
        def create_hash_side_effect(path):
            return f'a correct unique hash {path}', 'a correct hashing algo'
        create_hash_patch.side_effect = create_hash_side_effect

        # Test 1: no training plan registered, cannot read database
        self.patcher_db_search.start()

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.check_hashes_for_registered_training_plans()

        self.patcher_db_search.stop()

        # Test 2: one training plan registered, but cannot update database

        # register training plan
        training_plan_file_1_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
        training_plan_file_copied_path = os.path.join(environ['TMP_DIR'], 'copied-test-training-plan-1.txt')
        shutil.copy(training_plan_file_1_path, training_plan_file_copied_path)
        self.tp_security_manager.register_training_plan(
            name='test-training-plan',
            path=training_plan_file_copied_path,
            training_plan_type='registered',
            description='desc',
            training_plan_id='test-training-plan-id'
        )
        # update database with other hash and hashing algorithms
        training_plans = self.tp_security_manager._db.search(self.tp_security_manager._database.training_plan_type.all('registered'))
        for training_plan in training_plans:
            self.tp_security_manager._db.update({'hash': "different hash",
                                           'algorithm': "different_hashing_algorithm"},
                                          self.tp_security_manager._database.training_plan_id.all(training_plan["training_plan_id"]))

        self.patcher_db_update.start()

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.check_hashes_for_registered_training_plans()

        self.patcher_db_update.stop()

        # Test 3 : one registered training_plan, but file has been removed

        # remove the training_plan file stored on the system
        os.remove(training_plan_file_copied_path)

        self.patcher_db_remove.start()

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.check_hashes_for_registered_training_plans()

        self.patcher_db_remove.stop()

    def test_training_plan_manager_14_create_txt_training_plan_from_py(self):
        """Test training plan manager: tests if txt file can be created from py file"""
        # initialisation: creating a *.py file
        randomfolder = tempfile.mkdtemp()
        if not os.access(randomfolder, os.W_OK):
            self.skipTest("Test skipped cause temporary directory not writtable")
        else:
            file = os.path.join(environ['TMP_DIR'], 'training_plan.py')
            code_source = \
                "class TestClass:\n" + \
                "   def __init__(self, **kwargs):\n" + \
                "       self._kwargs = kwargs\n" + \
                "   def load_state(self, state :str):\n" + \
                "       self._state = state\n"
            with open(file, 'w') as f:
                f.write(code_source)

            # action
            txt_training_plan_path = self.tp_security_manager.create_txt_training_plan_from_py(file)

            # checks
            # tests if `txt_training_plan` has a *.txt extension
            _, ext = os.path.splitext(txt_training_plan_path)
            self.assertEqual(ext, '.txt')

            # check if content is the same in *.txt file and in *.py file
            with open(txt_training_plan_path, 'r') as f:
                code = f.read()

            self.assertEqual(code, code_source)

    def test_training_plan_manager_15_update_training_plan_normal_case(self, ):
        """Tests method `update_training_plan_hash` in the normal case scenario"""

        # database initialisation
        default_training_plan_file_1 = os.path.join(self.testdir, 'test-training-plan-1.txt')
        self.tp_security_manager.register_training_plan(
            name='test-training-plan',
            path=default_training_plan_file_1,
            training_plan_type='registered',
            description='desc',
            training_plan_id='test-training-plan-id'
        )
        # value to update the database
        file_modification_date_timestamp = 987654321.1234567
        file_modification_date_literal = datetime.fromtimestamp(file_modification_date_timestamp). \
            strftime("%d-%m-%Y %H:%M:%S.%f")
        file_creation_date_timestamp = 1234567890.1234567
        file_creation_date_literal = datetime.fromtimestamp(file_creation_date_timestamp). \
            strftime("%d-%m-%Y %H:%M:%S.%f")
        training_plan_hash = 'a hash'
        training_plan_hashing_algorithm = 'a_hashing_algorithm'

        # action
        default_training_plan_file_2 = os.path.join(self.testdir, 'test-training-plan-2.txt')
        with (patch.object(TrainingPlanSecurityManager, '_create_hash',
                           return_value=(training_plan_hash, training_plan_hashing_algorithm)),
              patch.object(os.path, 'getmtime', return_value=file_modification_date_timestamp),
              patch.object(os.path, 'getctime', return_value=file_creation_date_timestamp)):
            self.tp_security_manager.update_training_plan_hash('test-training-plan-id', default_training_plan_file_2)

        # checks
        # first, we are accessing to the updated training plan
        updated_training_plan = self.tp_security_manager._db.get(self.tp_security_manager._database.name == 'test-training-plan')

        # we are then checking that each entry in the database is correct
        self.assertEqual(updated_training_plan['hash'], training_plan_hash)
        self.assertEqual(updated_training_plan['algorithm'], training_plan_hashing_algorithm)
        self.assertEqual(updated_training_plan['date_modified'], file_modification_date_literal)
        self.assertEqual(updated_training_plan['date_created'], file_creation_date_literal)
        self.assertEqual(updated_training_plan['training_plan_path'], default_training_plan_file_2)

    def test_training_plan_manager_16_update_training_plan_exception1(self):
        """Tests method `update_training_plan_hash` in error cases """

        # Test 1 : update of a default training plan

        # database preparation
        default_training_plan_file_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
        self.tp_security_manager.register_training_plan(
            name='test-training-plan',
            path=default_training_plan_file_path,
            training_plan_type='default',
            description='desc',
            training_plan_id='test-training-plan-id'
        )
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.update_training_plan_hash(training_plan_id='test-training-plan-id',
                                                 path=default_training_plan_file_path)

    def test_training_plan_manager_17_update_training_plan_exception2(self):
        """Tests method `update_training_plan_hash` in error cases continued """

        # Test 2 : database access error

        # prepare
        for patch_start, patch_stop in [
                (self.patcher_db_get.start, self.patcher_db_get.stop),
                (self.patcher_db_update.start, self.patcher_db_update.stop)]: 
            for training_plan_type in [TrainingPlanStatus.REGISTERED.value, TrainingPlanStatus.REQUESTED.value]:
                training_plan_file_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
                self.tp_security_manager.register_training_plan(
                    name='test-training-plan',
                    path=training_plan_file_path,
                    training_plan_type=training_plan_type,
                    description='desc',
                    training_plan_id='test-training-plan-id'
                )
                training_plan_file_path_new = os.path.join(self.testdir, 'test-training-plan-2.txt')

                patch_start()

                # test + check
                with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                    self.tp_security_manager.update_training_plan_hash(training_plan_id='test-training-plan-id',
                                                         path=training_plan_file_path_new)

                # clean
                patch_stop()
                self.tp_security_manager.delete_training_plan('test-training-plan-id')


    def test_training_plan_manager_18_delete_registered_training_plans(self):
        """ Testing delete operation for training plan manager """

        training_plan_file_path = os.path.join(self.testdir, 'test-training-plan-1.txt')

        self.tp_security_manager.register_training_plan(
            name='test-training-plan-1',
            path=training_plan_file_path,
            training_plan_type='registered',
            description='desc'
        )

        # Get registered training plan
        training_plan_1 = self.tp_security_manager._db.get(self.tp_security_manager._database.name == 'test-training-plan-1')

        # Delete training plan
        self.tp_security_manager.delete_training_plan(training_plan_1['training_plan_id'])

        # Check training plan is removed
        training_plan_1_r = self.tp_security_manager._db.get(self.tp_security_manager._database.name == 'test-training-plan-1')
        self.assertIsNone(training_plan_1_r, "Registered training plan is not removed")

        # Load default training plans
        self.tp_security_manager.register_update_default_training_plans()

        default_training_plans = os.listdir(environ['DEFAULT_TRAINING_PLANS_DIR'])
        for training_plan in default_training_plans:
            training_plan_path = os.path.join(environ['DEFAULT_TRAINING_PLANS_DIR'], training_plan)
            training_plan = self.tp_security_manager._db.get(self.tp_security_manager._database.training_plan_path == training_plan_path)

            # Check delete method removed default training plans (it shouldnt)
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.delete_training_plan(training_plan['training_plan_id'])

    def test_training_plan_manager_19_delete_training_plan_more_cases(self):
        """Test training plan manager `delete_training_plan` function: more cases.
        """

        training_plan_name = 'my_training_plan_name'
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
        training_plan_id = 'my_training_plan_id_for_test'


        # Test 1 : correct training plan removal from database

        # add one registered training plan in database
        self.tp_security_manager.register_training_plan(training_plan_name, 'my_training_plan_description', training_plan_path, training_plan_id = training_plan_id)
        training_plan1 = self.tp_security_manager.get_training_plan_by_name(training_plan_name)

        # test
        self.tp_security_manager.delete_training_plan(training_plan_id)
        training_plan2 = self.tp_security_manager.get_training_plan_by_name(training_plan_name)

        # check
        self.assertNotEqual(training_plan1, None)
        self.assertEqual(training_plan2, None)


        # Test 2 : try remove non existing training plan
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.delete_training_plan('non_existing_training_plan')


        # Test 3 : bad parameter type
        # test + check
        for bad_id in [None, 3, ['my_training_plan'], [], {}, {'training_plan_id': 'my_training_plan'}]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.delete_training_plan(bad_id)


        # Test 4 : database access error
        self.tp_security_manager.register_training_plan(training_plan_name, 'my_training_plan_description', training_plan_path, training_plan_id = training_plan_id)
        training_plan1 = self.tp_security_manager.get_training_plan_by_name(training_plan_name)

        # test + check        
        self.assertNotEqual(training_plan1, None)

        for patch_start, patch_stop in [
                (self.patcher_db_get.start, self.patcher_db_get.stop),
                (self.patcher_db_remove.start, self.patcher_db_remove.stop)]:
            patch_start()

            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.delete_training_plan(training_plan_id)

            patch_stop()

    def test_training_plan_manager_20_list_training_plans_correct(self):
        """ Testing list method of training plan manager for correct request cases """

        self.tp_security_manager.register_update_default_training_plans()
        training_plans = self.tp_security_manager.list_training_plans(verbose=False)
        self.assertIsInstance(training_plans, list, 'Could not get list of training plans properly')

        # Check with verbose
        training_plans = self.tp_security_manager.list_training_plans(verbose=True)
        self.assertIsInstance(training_plans, list, 'Could not get list of training plans properly in verbose mode')

        # do some tests on the first training plan of training plans contained in database
        self.assertNotIn('training_plan_path', training_plans[0])
        self.assertNotIn('hash', training_plans[0])
        self.assertNotIn('date_modified', training_plans[0])
        self.assertNotIn('date_created', training_plans[0])

        # check by sorting results ()
        # list of fields we are going to sort in alphabetical order
        sort_by_fields = ['date_last_action',
                          'training_plan_type',
                          'training_plan_status',
                          'algorithm', 
                          'researcher_id']

        for sort_by_field in sort_by_fields:
            training_plans_sorted_by_modified_date = self.tp_security_manager.list_training_plans(sort_by=sort_by_field,
                                                                            )
            for i in range(len(training_plans_sorted_by_modified_date) - 1):

                # do not compare if values extracted are set to None
                if training_plans_sorted_by_modified_date[i][sort_by_field] is not None \
                   and training_plans_sorted_by_modified_date[i + 1][sort_by_field] is not None:
                    self.assertLessEqual(training_plans_sorted_by_modified_date[i][sort_by_field],
                                         training_plans_sorted_by_modified_date[i + 1][sort_by_field])

        # check with results filtered on `training_plan_status` field
        # first, register a training plan
        training_plan_file_path = os.path.join(self.testdir, 'test-training-plan-1.txt')

        self.tp_security_manager.register_training_plan(
            name='test-training-plan-1',
            path=training_plan_file_path,
            training_plan_type='registered',
            description='desc'
        )
        # second, reject it 
        _, training_plan_to_reject = self.tp_security_manager.check_training_plan_status(training_plan_file_path, TrainingPlanStatus.REGISTERED)
        self.tp_security_manager.reject_training_plan(training_plan_to_reject['training_plan_id'])

        # action: gather only rejected training plans
        rejected_training_plans = self.tp_security_manager.list_training_plans(select_status=TrainingPlanApprovalStatus.REJECTED)

        self.assertIn('test-training-plan-1', [x['name'] for x in rejected_training_plans])
        self.assertNotIn(TrainingPlanApprovalStatus.APPROVED.value,
                         [x['training_plan_status'] for x in rejected_training_plans])
        # gather only pending training plans (there should be no training plan with pending status,
        # so request returns empty list)
        pending_training_plans = self.tp_security_manager.list_training_plans(select_status=TrainingPlanApprovalStatus.PENDING,
                                                        verbose=False)
        self.assertEqual(pending_training_plans, [])

        # filtering with more than one status (get only REJECTED and APPROVAL training plan)
        # plus sorting
        for sort_by in [None, 'training_plan_id', 'NON_EXISTING_KEY']:
            rejected_and_approved_training_plans = self.tp_security_manager.list_training_plans(sort_by=sort_by,
                                                                          select_status=[TrainingPlanApprovalStatus.REJECTED,
                                                                                         TrainingPlanApprovalStatus.APPROVED],
                                                                          verbose=False)
            for training_plan in rejected_and_approved_training_plans:
                # Training plan status should be either Rejected or Approved...
                self.assertIn(training_plan['training_plan_status'],
                              [TrainingPlanApprovalStatus.REJECTED.value,
                               TrainingPlanApprovalStatus.APPROVED.value])
                # ... but not Pending
                self.assertNotEqual(training_plan['training_plan_status'], TrainingPlanApprovalStatus.PENDING.value)


    def test_training_plan_manager_21_list_training_plans_errors(self):
        """Test training plan manager `list_training_plans` function for error cases.
        """

        # default training plans in database (not directly used, but to search among multiple entries)
        self.tp_security_manager.register_update_default_training_plans()

        training_plan_name = 'my_training_plan_name'
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')

        # add one registered training plan in database
        self.tp_security_manager.register_training_plan(training_plan_name, 'my_training_plan_description', training_plan_path)


        # Test 1 : bad parameters
        for bad_sort in [True, 7, [], ['training_plan_type'], {}, {'training_plan_type': 'training_plan_type'}]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.list_training_plans(bad_sort, None, True, None)

        for bad_status in [True, 7, {}, {'training_plan_status': 'training_plan_status'}]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.list_training_plans(None, bad_status, True, None)

        for bad_verbose in [None, 7, {}, [], ['training_plan_verbose'], {'training_plan_verbose': 'training_plan_verbose'}]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.list_training_plans(None, None, bad_verbose, None)

        # {} is ok, not search by criterion is performed
        for bad_search in [False, 7, [], ['training_plan_search'], { 'dummy': 'training_plan_id'},
                { 'by': 'training_plan_id'}, { 'text': 'search text'}]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.list_training_plans(None, None, True, bad_search)


        # Test 2 : database access error
        self.patcher_db_search.start()
        self.patcher_db_all.start()

        for status in [
                None,
                TrainingPlanApprovalStatus.PENDING,
                [],
                [TrainingPlanApprovalStatus.REJECTED, TrainingPlanApprovalStatus.APPROVED]]:
            for search in [None, {}, {'by': 'training_plan_type', 'text': 'Registered'}]:
                for sort_by in [None, 'colonne']: # non existing entry
                    for verbose in [True, False]:
                        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                            self.tp_security_manager.list_training_plans(sort_by, status, verbose, search)

        self.patcher_db_search.stop()
        self.patcher_db_all.stop()

        # remaining uncovered case
        self.patcher_db_search.start()

        for verbose in [True, False]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.list_training_plans('colonne', None, verbose, None)

        self.patcher_db_search.stop()


    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.get_training_plan_from_database')
    def test_training_plan_manager_22_reply_training_plan_status_request(self,
                                                         mock_get_training_plan,
                                                         mock_download):
        """Tests training plan manager `reply_training_plan_status_request` method (normal case scenarii)"""

        messaging = MagicMock()
        messaging.send_message.return_value = None
        default_training_plans = os.listdir(environ['DEFAULT_TRAINING_PLANS_DIR'])
        mock_download.return_value = 200, None
        mock_get_training_plan.return_value = {'training_plan_status': TrainingPlanApprovalStatus.APPROVED.value}

        msg = {
            'researcher_id': 'ssss',
            'job_id': 'xxx',
            'training_plan_url': 'file:/' + environ['DEFAULT_TRAINING_PLANS_DIR'] + '/' + default_training_plans[1],
            'command': 'training-plan-status'
        }
        # test 1: case where status code of HTTP request equals 200 AND training plan
        # has been approved
        self.tp_security_manager.reply_training_plan_status_request(msg, messaging)

        # check:
        messaging.send_message.assert_called_once_with({'researcher_id': 'ssss',
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': True,
                                                        'approval_obligation': True,
                                                        'status': TrainingPlanApprovalStatus.APPROVED.value,
                                                        'msg': "Training plan has been approved by the node," +
                                                        " training can start",
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })
        with self.assertRaises(FedbiomedMessageError):
            # should trigger a FedBiomedMessageError because 'researcher_id' should be a string
            # (and not a boolean)
            msg['researcher_id'] = True
            self.tp_security_manager.reply_training_plan_status_request(msg, messaging)

        # test 2: case where status code of HTTP request equals 200 AND training plan has
        # not been approved
        msg['researcher_id'] = 'dddd'

        for training_plan_status, message in [
                (TrainingPlanApprovalStatus.REJECTED.value, 'Training plan has been rejected by the node, training is not possible'),
                (TrainingPlanApprovalStatus.PENDING.value, 'Training plan is pending: waiting for a review')]:
            # prepare
            mock_get_training_plan.return_value = {'training_plan_status': training_plan_status}
            messaging.reset_mock()

            # test
            self.tp_security_manager.reply_training_plan_status_request(msg, messaging)

            # check
            messaging.send_message.assert_called_once_with({'researcher_id': 'dddd',
                                                            'node_id': environ['NODE_ID'],
                                                            'job_id': 'xxx',
                                                            'success': True,
                                                            'approval_obligation': True,
                                                            'status': training_plan_status,
                                                            'msg': message,
                                                            'training_plan_url': msg['training_plan_url'],
                                                            'command': 'training-plan-status'
                                                            })

        # test 3: case where "TRAINING_PLAN_APPROVAL" has not been set
        mock_get_training_plan.return_value = {'training_plan_status': TrainingPlanApprovalStatus.REJECTED.value}
        messaging.reset_mock()

        TestTrainingPlanSecurityManager.env["TRAINING_PLAN_APPROVAL"] = False
        test3_msg = 'This node does not require training plan approval (maybe for debugging purposes).'
        # action
        self.tp_security_manager.reply_training_plan_status_request(msg, messaging)

        # checks
        messaging.send_message.assert_called_once_with({'researcher_id': 'dddd',
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': True,
                                                        'approval_obligation': False,
                                                        'status': TrainingPlanApprovalStatus.REJECTED.value,
                                                        'msg': test3_msg,
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })

        # test 4: case where status code of HTTP request equals 404 (request failed)
        mock_download.return_value = 404, None
        mock_get_training_plan.return_value = {'training_plan_status': TrainingPlanApprovalStatus.PENDING.value}
        msg['researcher_id'] = '12345'

        messaging.reset_mock()
        self.tp_security_manager.reply_training_plan_status_request(msg, messaging)
        # check:
        messaging.send_message.assert_called_once_with({'researcher_id': msg['researcher_id'],
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': False,
                                                        'approval_obligation': False,
                                                        'status': 'Error',
                                                        'msg': f'Can not download training plan file. {msg["training_plan_url"]}',
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })

        # test 5: case where training plan is not registered

        for approval, message in [
                (False, 'This node does not require training plan approval (maybe for debugging purposes).'),
                (True, "Unknown training plan not in database (status Not Registered)")]:

            # prepare
            msg = {
                'researcher_id': 'ssss',
                'job_id': 'xxx',
                'training_plan_url': 'file:/' + os.path.join(self.testdir, 'test-training-plan-1.txt'),
                'command': 'training-plan-status'
            }
            TestTrainingPlanSecurityManager.env["TRAINING_PLAN_APPROVAL"] = approval

            mock_download.return_value = 200, None
            mock_get_training_plan.return_value = None
            messaging.reset_mock()

            # test
            self.tp_security_manager.reply_training_plan_status_request(msg, messaging)

            # check
            messaging.send_message.assert_called_once_with({
                'researcher_id': msg['researcher_id'],
                'node_id': environ['NODE_ID'],
                'job_id': 'xxx',
                'success': True,
                'approval_obligation': approval,
                'status': 'Not Registered',
                'msg': message,
                'training_plan_url': msg['training_plan_url'],
                'command': 'training-plan-status'})

    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.get_training_plan_from_database')
    def test_training_plan_manager_23_reply_training_plan_status_request_exception(self,
                                                                   mock_get_training_plan,
                                                                   mock_download):
        """
        Tests `reply_training_plan_status_request` method when exceptions are occuring:
        - 1: by `Repository.download_file` (FedbiomedRepositoryError)
        - 2: by `TrainingPlanSecurityManager.check_is_training_plan_approved` (Exception)
        Checks that message (that should be sent to researcher) is created accordingly
        to the triggered exception 

        """
        # test 1: tests that error triggered through `Repository.download)file` is
        # correctly handled
        # patches 
        messaging = MagicMock()
        messaging.send_message.return_value = None
        default_training_plans = os.listdir(environ['DEFAULT_TRAINING_PLANS_DIR'])
        download_exception = FedbiomedRepositoryError("mimicking an exception triggered from"
                                                      "fedbiomed.common.repository")
        mock_download.side_effect = download_exception
        mock_get_training_plan.return_value = True, {}

        msg = {
            'researcher_id': 'ssss',
            'job_id': 'xxx',
            'training_plan_url': 'file:/' + environ['DEFAULT_TRAINING_PLANS_DIR'] + '/' + default_training_plans[0],
            'command': 'training-plan-status'
        }

        download_err_msg = ErrorNumbers.FB604.value + ': An error occurred when downloading training plan file.' + \
            f' {msg["training_plan_url"]} , {str(download_exception)}'

        # action
        self.tp_security_manager.reply_training_plan_status_request(msg, messaging)

        # check
        messaging.send_message.assert_called_once_with({'researcher_id': msg['researcher_id'],
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': False,
                                                        'approval_obligation': False,
                                                        'status': 'Error',
                                                        'msg': download_err_msg,
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })
        # test 2: test that error triggered through `check_training_plan_status` method
        # of `TrainingPlanSecurityManager` is correctly handled

        # resetting `mock_download`
        mock_download.side_effect = None
        mock_download.return_value = 200, None
        messaging.reset_mock()
        # creating a new exception for `check_training_plan_status` method
        checking_training_plan_exception = FedbiomedTrainingPlanSecurityManagerError("mimicking an exception happening when calling "
                                                              "'check_is_training_plan_approved'")
        mock_get_training_plan.side_effect = checking_training_plan_exception

        checking_training_plan_err_msg = ErrorNumbers.FB606.value + ': Cannot check if training plan has been registered.' + \
            f' Details {checking_training_plan_exception}'
        # action
        self.tp_security_manager.reply_training_plan_status_request(msg, messaging)

        # check
        messaging.send_message.assert_called_once_with({'researcher_id': msg['researcher_id'],
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': False,
                                                        'approval_obligation': False,
                                                        'status': 'Error',
                                                        'msg': checking_training_plan_err_msg,
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })

        # test 3: test that any other errors (inheriting from Exception) are caught
        messaging.reset_mock()
        # creating a new exception for `check_is_training_plan_approved` method
        checking_training_plan_exception_t3 = Exception("mimicking an exception happening when calling "
                                                "'check_training_plan_status'")

        checking_training_plan_err_msg_t3 = ErrorNumbers.FB606.value + ': An unknown error occurred when downloading training plan ' +\
            f'file. {msg["training_plan_url"]} , {str(checking_training_plan_exception_t3)}'

        mock_get_training_plan.side_effect = checking_training_plan_exception_t3
        # action
        self.tp_security_manager.reply_training_plan_status_request(msg, messaging)

        # check
        messaging.send_message.assert_called_once_with({'researcher_id': msg['researcher_id'],
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': False,
                                                        'approval_obligation': False,
                                                        'status': 'Error',
                                                        'msg': checking_training_plan_err_msg_t3,
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })

    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager._create_hash')
    @patch('os.path.getctime')
    @patch('os.path.getmtime')
    def test_training_plan_manager_17_check_training_plan_not_existing(self,
                                                       getmtime_patch,
                                                       getctime_patch,
                                                       create_hash_patch):
        """Test `_check_training_plan_not_existing` function"""
        # note: if _check_training_plan_not_existing succeeds, it returns/changes nothing,
        # so we don't have to do a `assert` in this case

        # Test 1 : test with empty database, no check should raise error
        for n in [None, 'dummy_name']:
            for p in [None, 'dummy_path']:
                for h in [None, 'dummy_hash']:
                    for a in [None, 'dummy_algorithm']:
                        self.assertIsNone(self.tp_security_manager._check_training_plan_not_existing(n, p, h, a))

        # Inter-test : add training plan in database for next tests
        training_plan_name = 'my_training_plan_name'
        training_plan_path = 'my_training_plan_path'
        training_plan_hash = 'my_training_plan_hash'
        training_plan_algorithm = 'my_training_plan_algorithm'
        # patching
        getctime_patch.value = 12345
        getmtime_patch.value = 23456

        def create_hash_side_effect(path):
            return training_plan_hash, training_plan_algorithm
        create_hash_patch.side_effect = create_hash_side_effect

        self.tp_security_manager.register_training_plan(training_plan_name,
                                                        'my_training_plan_description',
                                                        training_plan_path)

        # Test 2 : test with 1 training plan in database, check with different values, no error
        for n in [None, 'dummy_name']:
            for p in [None, 'dummy_path']:
                for h in [None, 'dummy_hash']:
                    for a in [None, 'dummy_algorithm']:
                        self.assertIsNone(self.tp_security_manager._check_training_plan_not_existing(n, p, h, a))

        # Test 3 : test with 1 training plan in database, check with existing value, error raised
        for n in [None, training_plan_name]:
            for p in [None, training_plan_path]:
                for h, a in [(None, None), (training_plan_hash, training_plan_algorithm)]:
                    if all(i is None for i in (n, p, h, a)):
                        # no error occurs if we don't check against any condition ...
                        continue
                    with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                        self.tp_security_manager._check_training_plan_not_existing(n, p, h, a)

        # Inter-test : corrupt database content to prevent proper query
        with open(environ['DB_PATH'], 'w') as f:
            f.write('CORRUPTED DATABASE CONTENT')

        # Test 4 : database not readable, error raised
        for n in [None, 'dummy_name']:
            for p in [None, 'dummy_path']:
                for h in [None, 'dummy_hash']:
                    for a in [None, 'dummy_algorithm']:
                        if all(i is None for i in (n, p, h, a)):
                            continue
                        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                            self.tp_security_manager._check_training_plan_not_existing(n, p, h, a)

        # Final : empty database to enable proper cleaning
        with open(environ['DB_PATH'], 'w') as f:
            f.write('')        

    @patch('os.path.getctime')
    @patch('os.path.getmtime')
    def test_training_plan_manager_24_check_training_plan_status(self,
                                                 getmtime_patch,
                                                 getctime_patch):
        """Test `check_training_plan_status` function
        """

        # patching
        getctime_patch.value = 12345
        getmtime_patch.value = 23456

        # default training plans in database (not directly used, but to search among multiple entries)
        self.tp_security_manager.register_update_default_training_plans()

        training_plan_name = 'my_training_plan_name'
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')

        # add one registered training plan in database
        self.tp_security_manager.register_training_plan(training_plan_name, 'my_training_plan_description', training_plan_path)

        # Test 1 : successful search for registered training plan
        for status in [None, TrainingPlanStatus.REGISTERED, TrainingPlanApprovalStatus.APPROVED]:
            is_present, training_plan = self.tp_security_manager.check_training_plan_status(training_plan_path, status)
            self.assertIsInstance(is_present, bool)
            self.assertTrue(is_present)
            self.assertIsInstance(training_plan, dict)
            self.assertEqual(training_plan['name'], training_plan_name)
            self.assertEqual(training_plan['training_plan_path'], training_plan_path)

        # Test 2 : unsuccessful search for registered training plan
        for status in [TrainingPlanStatus.REQUESTED, TrainingPlanStatus.DEFAULT,
                       TrainingPlanApprovalStatus.PENDING, TrainingPlanApprovalStatus.REJECTED]:
            is_present, training_plan = self.tp_security_manager.check_training_plan_status(training_plan_path, status)
            self.assertIsInstance(is_present, bool)
            self.assertFalse(is_present)
            self.assertEqual(training_plan, None)

        # Test 3 : error because of bad status in search
        for status in [ 3, True, 'toto', {}, {'clef'}, [], ['un']]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.check_training_plan_status(training_plan_path, status)

        # Test 4 : error in database access
        self.patcher_db_get.start()

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.check_training_plan_status(training_plan_path, None)

        self.patcher_db_get.stop()

    def test_training_plan_manager_25_get_training_plan_by_name(self):
        """Test `get_training_plan_by_name` function
        """

        training_plan_name = 'my_training_plan_name'
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')

        # default training plans in database (not directly used, but to search among multiple entries)
        self.tp_security_manager.register_update_default_training_plans()

        # add one registered training plan in database
        self.tp_security_manager.register_training_plan(training_plan_name, 'my_training_plan_description', training_plan_path)

        # Test 1 : look for existing training plan
        training_plan = self.tp_security_manager.get_training_plan_by_name(training_plan_name)
        self.assertIsInstance(training_plan, dict)
        self.assertEqual(training_plan['name'], training_plan_name)

        # Test 2 : look for non existing training plan
        training_plan = self.tp_security_manager.get_training_plan_by_name('ANY DUMMY YUMMY')
        self.assertEqual(training_plan, None)

        # Test 3 : bad parameter errors
        for mpath in [None, 3, {}, { 'clef': training_plan_path }, [], [training_plan_path], self.Dummy, self.Dummy()]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.get_training_plan_by_name(mpath)

        # Test 4 : database access error
        self.patcher_db_get.start()

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.get_training_plan_by_name(training_plan_name)

        self.patcher_db_get.stop()    

    def test_training_plan_manager_26_get_training_plan_from_database(self):
        """Test `get_training_plan_from_database` function
        """

        training_plan_name = 'my_training_plan_name'
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
        same_training_plan_path = os.path.join(environ['TMP_DIR'], 'test-training-plan-1-2.txt')
        shutil.copy(training_plan_path, same_training_plan_path)

        # default training plans in database (not directly used, but to search among multiple entries)
        self.tp_security_manager.register_update_default_training_plans()

        # add one registered training plan in database
        self.tp_security_manager.register_training_plan(training_plan_name, 'my_training_plan_description', training_plan_path)

        # Test 1 : look for existing training plan
        for mpath in [training_plan_path, same_training_plan_path]:
            training_plan = self.tp_security_manager.get_training_plan_from_database(mpath)
            self.assertTrue(isinstance(training_plan, dict))
            self.assertEqual(training_plan['name'], training_plan_name)

        # Test 2 : look for non-existing training plan
        training_plan = self.tp_security_manager.get_training_plan_from_database(os.path.join(self.testdir, 'test-training-plan-2.txt'))
        self.assertIsNone(training_plan)

        # Test 3 : bad parameter errors
        for mpath in [None, 3, {}, { 'clef': training_plan_path }, [], [training_plan_path], self.Dummy, self.Dummy()]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager.get_training_plan_from_database(mpath)

        # Test 4 : database access error
        self.patcher_db_get.start()

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.get_training_plan_from_database(training_plan_path)

        self.patcher_db_get.stop()    

    def test_training_plan_manager_27_get_training_plan_by_id(self):
        """Test `get_training_plan_by_id` function
        """

        training_plan_name = 'my_training_plan_name'
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
        training_plan_id = 'my_training_plan_id_for_test'

        # default training plans in database (not directly used, but to search among multiple entries)
        self.tp_security_manager.register_update_default_training_plans()

        # add one registered training plan in database
        self.tp_security_manager.register_training_plan(training_plan_name, 'my_training_plan_description', training_plan_path, training_plan_id = training_plan_id)

        # Test 1 : look for existing training plan
        for secure in [True, False]:
            for content in [True, False]:
                training_plan = self.tp_security_manager.get_training_plan_by_id(training_plan_id, secure, content)
                self.assertIsInstance(training_plan, dict)
                self.assertEqual(training_plan['name'], training_plan_name)
                self.assertEqual(training_plan['training_plan_id'], training_plan_id)
                if not secure:
                    self.assertEqual(training_plan['training_plan_path'], training_plan_path)

        # Test 2 : look for non existing training plan
        for secure in [True, False]:
            for content in [True, False]:
                training_plan = self.tp_security_manager.get_training_plan_by_id('NON_EXISTING_TRAINING_PLAN_ID', secure, content)
                self.assertIsNone(training_plan)

        # Test 3 : bad parameter errors
        for secure in [True, False]:
            for content in [True, False]:
                for mid in [None, 3, {}, { 'training_plan_id': training_plan_id }, [], [training_plan_id], self.Dummy, self.Dummy()]:
                    with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                        self.tp_security_manager.get_training_plan_by_id(mid, secure, content)

        # Test 4 : database access error
        self.patcher_db_get.start()

        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager.get_training_plan_by_id(training_plan_id, secure, content)

        self.patcher_db_get.stop()    

    @patch('fedbiomed.common.repository.Repository.download_file')
    def test_training_plan_manager_28_reply_training_plan_approval_request(self, download_file_patch):
        """Test training plan manager `reply_training_plan_approval_request` function.
        """

        # patch
        messaging = MagicMock()
        messaging.send_message.return_value = None

        def download_file_side_effect(url, file):
            # remove leading file: url: if any
            srcfile = url.split(':', 1)[-1]
            dstfile = os.path.join(environ['TMP_DIR'], file)
            shutil.copyfile(srcfile, dstfile)
            return 200, dstfile

        download_file_patch.side_effect = download_file_side_effect

        # note: dont test bad message formatting, this is the duty of the Message class

        # Test 1 : training plan approval for non existing training plan

        # prepare
        training_plan_researcher_id = 'the researcher :%!#><|[]"*&$@!\'\\'
        training_plan_description = 'the description :%!#><|[]"*&$@!\'\\'
        training_plan_sequence = -4
        training_plan_file = os.path.join(self.testdir, 'test-training-plan-1.txt')
        msg = {
            'researcher_id': training_plan_researcher_id,
            'description': training_plan_description,
            'sequence': training_plan_sequence,
            'training_plan_url': 'file:' + training_plan_file,
            'command': 'approval'
        }

        # test
        training_plan_before = self.tp_security_manager.get_training_plan_from_database(training_plan_file)

        self.tp_security_manager.reply_training_plan_approval_request(msg, messaging)

        training_plan_after = self.tp_security_manager.get_training_plan_from_database(training_plan_file)

        # check
        messaging.send_message.assert_called_once_with({
            'researcher_id': training_plan_researcher_id,
            'node_id': environ['NODE_ID'],
            'sequence': training_plan_sequence,
            'status': 200,
            'command': 'approval',
            'success': True
        })

        self.assertEqual(training_plan_before, None)
        self.assertTrue(isinstance(training_plan_after, dict))
        self.assertEqual(training_plan_after['description'], training_plan_description)
        self.assertEqual(training_plan_after['researcher_id'], training_plan_researcher_id)
        self.assertEqual(training_plan_after['training_plan_type'], TrainingPlanStatus.REQUESTED.value)
        self.assertEqual(training_plan_after['training_plan_status'], TrainingPlanApprovalStatus.PENDING.value)

        # clean
        messaging.reset_mock()


        # Test 2 : training plan approval for existing training plan

        # prepare

        # different message : check that existing entry in database is not updated
        training_plan2_researcher_id = 'another researcher id'
        training_plan2_description = 'another description'
        training_plan2_sequence = -4
        training_plan2_file = training_plan_file  # re-using training_plan_id from test 1 for test 2
        training_plan_id = training_plan_after['training_plan_id']
        msg2 = {
            'researcher_id': training_plan2_researcher_id,
            'description': training_plan2_description,
            'sequence': training_plan2_sequence,
            'training_plan_url': 'file:' + training_plan2_file,
            'command': 'approval'
        }

        def noaction_training_plan(training_plan_id, extra_notes):
            pass

        for approval_action, approval_status in [
                (noaction_training_plan, TrainingPlanApprovalStatus.PENDING.value),
                (self.tp_security_manager.approve_training_plan, TrainingPlanApprovalStatus.APPROVED.value),
                (self.tp_security_manager.reject_training_plan, TrainingPlanApprovalStatus.REJECTED.value)]:
            # also update status
            approval_action(training_plan_id, 'dummy notes')

            # test
            self.tp_security_manager.reply_training_plan_approval_request(msg2, messaging)

            training_plan2_after = self.tp_security_manager.get_training_plan_from_database(training_plan_file)

            # check
            messaging.send_message.assert_called_once_with({
                'researcher_id': training_plan2_researcher_id,
                'node_id': environ['NODE_ID'],
                'sequence': training_plan2_sequence,
                'status': 200,
                'command': 'approval',
                'success': True
            })
            # verify existing training plan in database did not change
            self.assertTrue(isinstance(training_plan2_after, dict))
            self.assertEqual(training_plan2_after['description'], training_plan_description)
            self.assertEqual(training_plan2_after['researcher_id'], training_plan_researcher_id)
            self.assertEqual(training_plan2_after['training_plan_type'], TrainingPlanStatus.REQUESTED.value)
            self.assertEqual(training_plan2_after['training_plan_status'], approval_status)

            # clean
            messaging.reset_mock()


        # Test 3 : training plan approval with errors

        # prepare
        raise_filenotfound_error = FileNotFoundError('some additional information')
        self.patcher_shutil_move = patch('shutil.move', MagicMock(side_effect=raise_filenotfound_error))

        self.tp_security_manager.delete_training_plan(training_plan_id)

        for error in ['download', 'db_get', 'db_upsert', 'file_move']:

            # test-specific prepare
            if error == 'download':
                download_file_patch.side_effect = FedbiomedRepositoryError('any error message')
                training_plan3_status = 0

            # test
            training_plan3_before = self.tp_security_manager.get_training_plan_from_database(training_plan_file)

            # test-specific prepare
            if error == 'db_get':
                self.patcher_db_get.start()
            elif error == 'db_upsert':
                self.patcher_db_upsert.start()
            elif error == 'file_move':
                self.patcher_shutil_move.start()

            self.tp_security_manager.reply_training_plan_approval_request(msg, messaging)

            # test-specific clean
            if error == 'db_get':
                self.patcher_db_get.stop()
            elif error == 'db_upsert':
                self.patcher_db_upsert.stop()
            elif error == 'file_move':
                self.patcher_shutil_move.stop()

            training_plan3_after = self.tp_security_manager.get_training_plan_from_database(training_plan_file)

            # check
            messaging.send_message.assert_called_once_with({
                'researcher_id': training_plan_researcher_id,
                'node_id': environ['NODE_ID'],
                'sequence': training_plan_sequence,
                'status': training_plan3_status,
                'command': 'approval',
                'success': False
            })
            self.assertEqual(training_plan3_before, None)
            self.assertEqual(training_plan3_after, None)

            # clean
            messaging.reset_mock()

            # test-specific clean
            if error == 'download':
                download_file_patch.side_effect = download_file_side_effect
                training_plan3_status = 200

    def test_training_plan_manager_29_update_training_plan_status(self):
        """Test training plan manager `_update_training_plan_status` function.
        """

        training_plan_name = 'my_training_plan_name'
        training_plan_path = os.path.join(self.testdir, 'test-training-plan-1.txt')
        training_plan_id = 'my_training_plan_id_for_test'

        # add one registered training plan in database
        self.tp_security_manager.register_training_plan(training_plan_name,
                                                        'my_training_plan_description',
                                                        training_plan_path,
                                                        training_plan_id=training_plan_id)

        # Test 1 : do correct update in existing training plan
        for status in [
            TrainingPlanApprovalStatus.PENDING,
            TrainingPlanApprovalStatus.REJECTED,
            TrainingPlanApprovalStatus.APPROVED, # update 2x with same status to cover this case
            TrainingPlanApprovalStatus.APPROVED]:
            note = str(status)
            self.tp_security_manager._update_training_plan_status(training_plan_id, status, note)
            training_plan = self.tp_security_manager.get_training_plan_by_id(training_plan_id)

            self.assertEqual(training_plan['training_plan_status'], status.value)
            self.assertEqual(training_plan['notes'], note)

        # Test 2 : do update in non-existing training plan
        with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
            self.tp_security_manager._update_training_plan_status('non_existing_training_plan', TrainingPlanApprovalStatus.REJECTED, 'new_notes')

        # Test 3 : bad parameters
        for bad_id in [None, 3, ['my_training_plan'], [], {}, {'training_plan_id': 'my_training_plan'}]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._update_training_plan_status(bad_id, TrainingPlanApprovalStatus.REJECTED, 'new notes')

        for bad_status in [None, 5, [], ['mt_status'], {}, {'training_plan_status': 'my_status'}, 'Approved']:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._update_training_plan_status(training_plan_id, bad_status, 'new notes')

        for bad_notes in [5, [], ['mt_status'], {}, {'training_plan_status': 'my_status'}]:
            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._update_training_plan_status(training_plan_id, TrainingPlanApprovalStatus.REJECTED, bad_notes)

        # Test 4 : database access error
        for patch_start, patch_stop in [
                (self.patcher_db_get.start, self.patcher_db_get.stop),
                (self.patcher_db_update.start, self.patcher_db_update.stop)]:
            patch_start()

            with self.assertRaises(FedbiomedTrainingPlanSecurityManagerError):
                self.tp_security_manager._update_training_plan_status(training_plan_id, TrainingPlanApprovalStatus.REJECTED, 'new_notes')

            patch_stop()    

    def test_training_plan_manager_30_remove_sensible_keys_from_request(self):
        """Test training plan manager `_remove_sensible_keys_from_request` function.
        """

        # prepare
        key_sensible = 'training_plan_path'
        key_notsensible = 'training_plan_id'

        doc = { 
            key_sensible: 'valeur clef',
            key_notsensible: 'autre valeur'
        }
        doc2 = copy.deepcopy(doc)

        # test
        self.tp_security_manager._remove_sensible_keys_from_request(doc2)

        # check
        self.assertEqual(doc2[key_notsensible], doc[key_notsensible])
        self.assertFalse(key_sensible in doc2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
