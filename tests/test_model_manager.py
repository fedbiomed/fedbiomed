import builtins
import copy
from datetime import datetime
import os
import shutil
import tempfile
import unittest
import inspect
from unittest.mock import patch, MagicMock
# why call this in model_manager ?
#from wsgiref.util import setup_testing_defaults

import testsupport.mock_node_environ  # noqa (remove flake8 false warning)

from fedbiomed.common.constants import ErrorNumbers, HashingAlgorithms, TrainingPlanApprovalStatus, ModelTypes
from fedbiomed.common.exceptions import FedbiomedMessageError, FedbiomedModelManagerError, FedbiomedRepositoryError
from fedbiomed.common.logger import logger

from fedbiomed.node.environ import environ
from fedbiomed.node.model_manager import ModelManager


class TestModelManager(unittest.TestCase):
    """
    Unit tests for class `ModelManager` (from fedbiomed.node.model_manager)
    """

    # Raise an arbitrary error (eg: SystemError), whatever the conditions and parameters.
    # Used for mocking, when you want to have an error on some function.
    raise_some_error = SystemError('my error message')

    # dummy class for testing typing of parameters
    class Dummy():
        pass

    # before the tests
    def setUp(self):

        # This part important for setting fake values for environ -----------------
        # and properly mocking values in environ Since environ is singleton,
        # you should also mock environ objects that are called from modules e.g.
        # fedbiomed.node.model_manager.environ you should use another mock for
        # the environ object used in test functions
        self.values = copy.deepcopy(environ)

        def side_effect(arg):
            return self.values[arg]

        def side_effect_set_item(key, value):
            self.values[key] = value

        # self.environ_patch = patch('fedbiomed.node.environ.environ')
        self.environ_model_manager_patch = patch('fedbiomed.node.model_manager.environ')

        # self.environ = self.environ_patch.start()
        self.environ_model = self.environ_model_manager_patch.start()

        self.environ_model.__getitem__.side_effect = side_effect
        self.environ_model.__setitem__.side_effect = side_effect_set_item

        # patchers for causing database access errors
        # need to be (de)activated only for some pieces of test code
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

        # Build ModelManger
        self.model_manager = ModelManager()

        # get test directory to access test-model files
        self.testdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ),
            "test-model"
        )

    # after the tests
    def tearDown(self):
        # DB should be removed after each test to
        # have clear database for tests
        self.environ_model_manager_patch.stop()

        self.model_manager._tinydb.drop_table('Models')
        self.model_manager._tinydb.close()
        os.remove(environ['DB_PATH'])

    def test_model_manager_01_create_default_model_hashes(self):

        """ Testing whether created hash for model files are okay
        or not. It also tests every default with each provided hashing algorithm
        """

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        logger.info('Controlling Models Dir')
        logger.info(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:

            # set default hashing algorithm
            self.values['HASHING_ALGORITHM'] = 'SHA256'
            full_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)

            # Control return vlaues with default hashing algorithm
            hash, algortihm = self.model_manager._create_hash(full_path)
            self.assertIsInstance(hash, str, 'Hash creation is not successful')
            self.assertEqual(algortihm, 'SHA256', 'Wrong hashing algorithm')

            algorithms = HashingAlgorithms.list()
            for algo in algorithms:
                self.values['HASHING_ALGORITHM'] = algo
                hash, algortihm = self.model_manager._create_hash(full_path)
                self.assertIsInstance(hash, str, 'Hash creation is not successful')
                self.assertEqual(algortihm, algo, 'Wrong hashing algorithm')

    def test_model_manager_02_create_hash_hashing_exception(self):
        """Tests `create_hash` method is raising exception if hashing
        algorithm does not exist"""
        model_path = os.path.join(self.testdir, 'test-model-1.txt')
        self.values['HASHING_ALGORITHM'] = "AN_UNKNOWN_HASH_ALGORITHM"

        # action:
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager._create_hash(model_path)

    def test_model_manager_03_create_hash_open_exceptions(self):
        """Tests `create_hash` method is raising appropriate exception if
        cannot open and read model file (test try/catch blocks when opening
        a file)
        """
        model_path = os.path.join(self.testdir, 'test-model-1.txt')
        # test 1 : test case where model file has not been found

        # action 
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager._create_hash("a/path/that/should/not/exist/on/your/computer")

        # test 2: test case where model file cannot be read (due to a lack of privilege)
        with patch.object(builtins, 'open') as builtin_open_mock:
            builtin_open_mock.side_effect = PermissionError("mimicking a PermissionError")
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager._create_hash(model_path)

        # test 3: test where model file cannot be open and read
        with patch.object(builtins, 'open') as builtin_open_mock:
            builtin_open_mock.side_effect = OSError("mimicking a OSError")

            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager._create_hash(model_path)

    @patch('fedbiomed.node.model_manager.minify')
    def test_model_manager_04_create_hash_minify_exception(self, minify_patch):
        """Tests that `_create_hash` method is catching exception coming
        from `minify` package"""
        model_path = os.path.join(self.testdir, 'test-model-1.txt')

        minify_patch.side_effect = Exception('Mimicking an Exception triggered by `minify` package')

        # action
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager._create_hash(model_path)

    def test_model_manager_05_create_hash_param_exception(self):
        """"Tests `_create_patch` with incorrect parameters"""

        for mpath in [None, 18, -2.4, True, {}, { 'clef': 'valeur' }, [], ['un'], self.Dummy, self.Dummy()]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager._create_hash(mpath)

    def test_model_manager_06_update_default_hashes_when_algo_is_changed(self):
        """  Testing method for update/register default models when hashing
             algorithm has changed
        """

        # Single test with default hash algorithm
        self.model_manager.register_update_default_models()

        # # Multiple test with different hashing algorithms
        algorithms = HashingAlgorithms.list()
        for algo in algorithms:
            self.values['HASHING_ALGORITHM'] = algo
            self.model_manager.register_update_default_models()
            doc = self.model_manager._db.get(self.model_manager._database.model_type == "default")
            logger.info(doc)
            self.assertEqual(doc["algorithm"], algo,
                             'Hashes are not properly updated after hashing algorithm is changed')  # noqa

    def test_model_manager_07_update_default_model_deleted(self):
        """Tests `update_default_model` when a model file that had been registered 
        has been deleted
        """
        file_path = os.path.join(self.testdir, 'test-model-1.txt')
        new_default_model_path = os.path.join(environ['TMP_DIR'], 'test-model-1-2.txt')
        shutil.copy(file_path, new_default_model_path)

        # update database
        self.model_manager.register_model(
            name='test-model',
            path=new_default_model_path,
            model_type='default',
            description='desc'
        )
        # now, remove copied model from system
        os.remove(new_default_model_path)

        # check that model is in database
        model = self.model_manager._db.get(self.model_manager._database.model_path == new_default_model_path)
        self.assertIsNotNone(model)

        # action
        self.model_manager.register_update_default_models()

        # check that copied model entry has been removed
        removed_model = self.model_manager._db.get(self.model_manager._database.model_path == new_default_model_path)
        self.assertIsNone(removed_model)

    def test_model_manager_08_update_errors(self):
        """Testing error cases in function `register_update_default_models`.
        """

        # prepare: first select a hashing algorithm that is not the one set in ENVIRON
        # (first one that is different in the list)

        algo_initial = self.values['HASHING_ALGORITHM']
        for a in HashingAlgorithms.list():
            if a != algo_initial:
                algo_tempo = a
                break

        new_default_model_path = os.path.join(environ['DEFAULT_MODELS_DIR'], 'my_default_model.txt')

        for error in ['db_search', 'db_get-delete', 'db_remove-delete', 'db_get-exists', 'db_update-exists']:

            # prepare
            #
            # some test may manage update database or files, or leave them in an unclean state
            # so we need to prepare at each test

            shutil.copy(os.path.join(self.testdir, 'test-model-1.txt'), new_default_model_path)
            self.model_manager.register_update_default_models()
            models_before = self.model_manager._db.all()

            # test-specific prepare
            if error == 'db_search':
                self.patcher_db_search.start()
            elif error == 'db_get-delete':
                self.patcher_db_get.start()
                os.remove(new_default_model_path)
            elif error == 'db_remove-delete':
                self.patcher_db_remove.start()
                os.remove(new_default_model_path)
            elif error == 'db_get-exists':
                self.patcher_db_get.start()
                self.values['HASHING_ALGORITHM'] = algo_tempo
            elif error == 'db_update-exists':
                self.patcher_db_update.start()
                self.values['HASHING_ALGORITHM'] = algo_tempo

            # test and check
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.register_update_default_models()

            # test-specific clean
            if error == 'db_search':
                self.patcher_db_search.stop()
            elif error == 'db_get-delete':
                self.patcher_db_get.stop()
            elif error == 'db_remove-delete':
                self.patcher_db_remove.stop()
            elif error == 'db_get-exists':
                self.patcher_db_get.stop()
                self.values['HASHING_ALGORITHM'] = algo_initial
            elif error == 'db_update-exists':
                self.patcher_db_update.stop()
                self.values['HASHING_ALGORITHM'] = algo_initial

            # check models where not modified by failed operation
            models_after = self.model_manager._db.all()
            self.assertEqual(models_before, models_after)

        # clean
        if os.path.exists(new_default_model_path):
            os.remove(new_default_model_path)
        self.model_manager.register_update_default_models()


    def test_model_manager_09_update_modified_model_files(self):
        """ Testing update of modified default models """

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])

        # Test with only first file

        for model in default_models:
            file_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            self.model_manager.register_update_default_models()
            doc = self.model_manager._db.get(self.model_manager._database.model_path == file_path)

            # Open the file in append & read mode ('a+')
            with open(file_path, "a+") as file:
                lines = file.readlines()  # lines is list of line, each element '...\n'
                lines.insert(0, "\nprint('Hello world') \t \n")  # you can use any index if you know the line index
                file.seek(0)  # file pointer locates at the beginning to write the whole file again
                file.writelines(lines)

            self.model_manager.register_update_default_models()
            docAfter = self.model_manager._db.get(self.model_manager._database.model_path == file_path)

            self.assertNotEqual(doc['hash'], docAfter['hash'], "Hash couldn't updated after file has modified")

    def test_model_manager_10_register_model(self):
        """ Testing registering method for new models """

        # We should import environ to get fake values

        self.model_manager.register_update_default_models()

        model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')
        model_file_2 = os.path.join(self.testdir, 'test-model-2.txt')

        self.model_manager.register_model(
            name='test-model',
            path=model_file_1,
            model_type='registered',
            description='desc'
        )

        # When same model file wants to be added it should raise and exception
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                name='test-model-2',
                path=model_file_1,
                model_type='registered',
                description='desc')

        # When same model wants to be added with same name  and different file
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                name='test-model',
                path=model_file_2,
                model_type='registered',
                description='desc')

        # Wrong types
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                name='test-model-2',
                path=model_file_2,
                model_type=False,
                description='desc')

        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                name='tesasdsad',
                path=model_file_2,
                model_type=False,
                description=False)

        # Wrong model type
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                name='tesasdsad',
                path=model_file_2,
                model_type='123123',
                description='desc')

    @patch('fedbiomed.node.model_manager.ModelManager._check_model_not_existing')
    def test_model_manager_11_register_model_DB_error(self, check_model_not_existing_patch):
        """ Testing registering method for new models continued, case where model is corrupted"""

        # patch
        check_model_not_existing_patch.value = None

        model_file_2 = os.path.join(self.testdir, 'test-model-2.txt')

        # Cannot access corrupted database
        with open(environ['DB_PATH'], 'w') as f:
            f.write('CORRUPTED DATABASE CONTENT')

        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                name='test-model-3',
                path=model_file_2,
                model_type='registered',
                description='desc')        

        with open(environ['DB_PATH'], 'w') as f:
            f.write('') 

    @patch('fedbiomed.node.model_manager.ModelManager._create_hash')
    def test_model_manager_12_check_hashes_for_registerd_models(self,
                                                                create_hash_patch):
        """
        Tests `hashes_for_registered_models` method, with 3 settings
        - Test 1: no models are registered
        - Test 2: models are registered and are stored on computer
        - Test 3: model is no longer stored on computer.
        """
        # patch
        def create_hash_side_effect(path):
            return f'a correct unique hash {path}', 'a correct hashing algo'
        create_hash_patch.side_effect = create_hash_side_effect

        # test 1: case where there is no model registered
        self.model_manager.check_hashes_for_registered_models()

        # check that no models are not registered
        self.assertListEqual(self.model_manager._db.search(self.model_manager._database.model_type.all('registered')),
                             [])

        # test 2: case where default models have been registered
        # we will check that models have correct hashes and hashing algorithm
        # in the database
        self.model_manager.register_update_default_models()

        model_file_1_path = os.path.join(self.testdir, 'test-model-1.txt')

        # copying model (we will delete it afterward)
        model_file_copied_path = os.path.join(environ['TMP_DIR'], 'copied-test-model-1.txt')
        shutil.copy(model_file_1_path, model_file_copied_path)
        self.model_manager.register_model(
            name='test-model',
            path=model_file_copied_path,
            model_type='registered',
            description='desc',
            model_id='test-model-id'
        )

        # update database with non-existing hash and hashing algorithms
        models = self.model_manager._db.search(self.model_manager._database.model_type.all('registered'))
        for model in models:
            self.model_manager._db.update({'hash': "an incorrect hash",
                                           'algorithm': "incorrect_hashing_algorithm"},
                                          self.model_manager._database.model_id.all(model["model_id"]))

        self.model_manager.check_hashes_for_registered_models()

        # checks
        models = self.model_manager._db.search(self.model_manager._database.model_type.all('registered'))

        for model in models:
            correct_hash, correct_hashing_algo = create_hash_side_effect(model['model_path'])
            self.assertEqual(model['hash'], correct_hash)
            self.assertEqual(model['algorithm'], correct_hashing_algo)

        # Test 3: here we are testing that a file that has been removed on
        # the system is also removed from the database

        # remove the model file stored on the system
        # FIXME: should we skip the remaining tests if a PermissionError is triggered
        os.remove(model_file_copied_path)

        # action
        self.model_manager.check_hashes_for_registered_models()
        removed_model = self.model_manager._db.get(self.model_manager._database.name == 'test-model')
        # check that the model has been removed
        self.assertIsNone(removed_model)

    @patch('fedbiomed.node.model_manager.ModelManager._create_hash')
    def test_model_manager_13_check_hashes_for_registerd_models_DB_error(self, create_hash_patch):
        """
        Tests `hashes_for_registered_models` method, with 3 settings, causing database access errors
        - Test 1: no model registered, cannot read database
        - Test 2: one model registered, but cannot update database
        - Test 3: model is no longer stored on computer.
        """
        # patch
        def create_hash_side_effect(path):
            return f'a correct unique hash {path}', 'a correct hashing algo'
        create_hash_patch.side_effect = create_hash_side_effect

        # Test 1: no model registered, cannot read database
        self.patcher_db_search.start()

        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.check_hashes_for_registered_models()

        self.patcher_db_search.stop()

        # Test 2: one model registered, but cannot update database

        # register model
        model_file_1_path = os.path.join(self.testdir, 'test-model-1.txt')
        model_file_copied_path = os.path.join(environ['TMP_DIR'], 'copied-test-model-1.txt')
        shutil.copy(model_file_1_path, model_file_copied_path)
        self.model_manager.register_model(
            name='test-model',
            path=model_file_copied_path,
            model_type='registered',
            description='desc',
            model_id='test-model-id'
        )
        # update database with other hash and hashing algorithms
        models = self.model_manager._db.search(self.model_manager._database.model_type.all('registered'))
        for model in models:
            self.model_manager._db.update({'hash': "different hash",
                                           'algorithm': "different_hashing_algorithm"},
                                          self.model_manager._database.model_id.all(model["model_id"]))

        self.patcher_db_update.start()

        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.check_hashes_for_registered_models()

        self.patcher_db_update.stop()

        # Test 3 : one registered model, but file has been removed

        # remove the model file stored on the system
        os.remove(model_file_copied_path)

        self.patcher_db_remove.start()

        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.check_hashes_for_registered_models()

        self.patcher_db_remove.stop()

    def test_model_manager_14_create_txt_model_from_py(self):
        """Test model manager: tests if txt file can be created from py file"""
        # initialisation: creating a *.py file
        randomfolder = tempfile.mkdtemp()
        if not os.access(randomfolder, os.W_OK):
            self.skipTest("Test skipped cause temporary directory not writtable")
        else:
            file = os.path.join(environ['TMP_DIR'], 'model.py')
            code_source = \
                "class TestClass:\n" + \
                "   def __init__(self, **kwargs):\n" + \
                "       self._kwargs = kwargs\n" + \
                "   def load_state(self, state :str):\n" + \
                "       self._state = state\n"
            with open(file, 'w') as f:
                f.write(code_source)

            # action
            txt_model_path = self.model_manager.create_txt_model_from_py(file)

            # checks
            # tests if `txt_model` has a *.txt extension
            _, ext = os.path.splitext(txt_model_path)
            self.assertEqual(ext, '.txt')

            # check if content is the same in *.txt file and in *.py file
            with open(txt_model_path, 'r') as f:
                code = f.read()

            self.assertEqual(code, code_source)

    def test_model_manager_15_update_model_normal_case(self, ):
        """Tests method `update_model_hash` in the normal case scenario"""

        # database initialisation
        default_model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')
        self.model_manager.register_model(
            name='test-model',
            path=default_model_file_1,
            model_type='registered',
            description='desc',
            model_id='test-model-id'
        )
        # value to update the database
        file_modification_date_timestamp = 987654321.1234567
        file_modification_date_literal = datetime.fromtimestamp(file_modification_date_timestamp). \
            strftime("%d-%m-%Y %H:%M:%S.%f")
        file_creation_date_timestamp = 1234567890.1234567
        file_creation_date_literal = datetime.fromtimestamp(file_creation_date_timestamp). \
            strftime("%d-%m-%Y %H:%M:%S.%f")
        model_hash = 'a hash'
        model_hashing_algorithm = 'a_hashing_algorithm'

        # action
        default_model_file_2 = os.path.join(self.testdir, 'test-model-2.txt')
        with (patch.object(ModelManager, '_create_hash',
                           return_value=(model_hash, model_hashing_algorithm)),
              patch.object(os.path, 'getmtime', return_value=file_modification_date_timestamp),
              patch.object(os.path, 'getctime', return_value=file_creation_date_timestamp)):
            self.model_manager.update_model_hash('test-model-id', default_model_file_2)

        # checks
        # first, we are accessing to the updated model
        updated_model = self.model_manager._db.get(self.model_manager._database.name == 'test-model')

        # we are then checking that each entry in the database is correct
        self.assertEqual(updated_model['hash'], model_hash)
        self.assertEqual(updated_model['algorithm'], model_hashing_algorithm)
        self.assertEqual(updated_model['date_modified'], file_modification_date_literal)
        self.assertEqual(updated_model['date_created'], file_creation_date_literal)
        self.assertEqual(updated_model['model_path'], default_model_file_2)

    def test_model_manager_16_update_model_exception1(self):
        """Tests method `update_model_hash` in error cases """

        # Test 1 : update of a default model

        # database preparation
        default_model_file_path = os.path.join(self.testdir, 'test-model-1.txt')
        self.model_manager.register_model(
            name='test-model',
            path=default_model_file_path,
            model_type='default',
            description='desc',
            model_id='test-model-id'
        )
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.update_model_hash(model_id='test-model-id',
                                                 path=default_model_file_path)

    def test_model_manager_17_update_model_exception2(self):
        """Tests method `update_model_hash` in error cases continued """

        # Test 2 : database access error

        # prepare
        for patch_start, patch_stop in [
                (self.patcher_db_get.start, self.patcher_db_get.stop),
                (self.patcher_db_update.start, self.patcher_db_update.stop)]: 
            for model_type in [ModelTypes.REGISTERED.value, ModelTypes.REQUESTED.value]:
                model_file_path = os.path.join(self.testdir, 'test-model-1.txt')
                self.model_manager.register_model(
                    name='test-model',
                    path=model_file_path,
                    model_type=model_type,
                    description='desc',
                    model_id='test-model-id'
                )
                model_file_path_new = os.path.join(self.testdir, 'test-model-2.txt')

                patch_start()

                # test + check
                with self.assertRaises(FedbiomedModelManagerError):
                    self.model_manager.update_model_hash(model_id='test-model-id',
                                                         path=model_file_path_new)

                # clean
                patch_stop()
                self.model_manager.delete_model('test-model-id')


    def test_model_manager_18_delete_registered_models(self):
        """ Testing delete operation for model manager """

        model_file_path = os.path.join(self.testdir, 'test-model-1.txt')

        self.model_manager.register_model(
            name='test-model-1',
            path=model_file_path,
            model_type='registered',
            description='desc'
        )

        # Get registered model
        model_1 = self.model_manager._db.get(self.model_manager._database.name == 'test-model-1')

        # Delete model
        self.model_manager.delete_model(model_1['model_id'])

        # Check model is removed
        model_1_r = self.model_manager._db.get(self.model_manager._database.name == 'test-model-1')
        self.assertIsNone(model_1_r, "Registered model is not removed")

        # Load default models
        self.model_manager.register_update_default_models()

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:
            model_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            model = self.model_manager._db.get(self.model_manager._database.model_path == model_path)

            # Check delete method removed default models (it shouldnt)
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.delete_model(model['model_id'])

    def test_model_manager_19_delete_model_more_cases(self):
        """Test model manager `delete_model` function: more cases.
        """

        model_name = 'mymodel_name'
        model_path = os.path.join(self.testdir, 'test-model-1.txt')
        model_id = 'mymodel_id_for_test'


        # Test 1 : correct model removal from database

        # add one registered model in database
        self.model_manager.register_model(model_name, 'mymodel_description', model_path, model_id = model_id)
        model1 = self.model_manager.get_model_by_name(model_name)

        # test
        self.model_manager.delete_model(model_id)
        model2 = self.model_manager.get_model_by_name(model_name)

        # check
        self.assertNotEqual(model1, None)
        self.assertEqual(model2, None)


        # Test 2 : try remove non existing model
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.delete_model('non_existing_model')


        # Test 3 : bad parameter type
        # test + check
        for bad_id in [None, 3, ['my_model'], [], {}, {'model_id': 'my_model'}]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.delete_model(bad_id)


        # Test 4 : database access error
        self.model_manager.register_model(model_name, 'mymodel_description', model_path, model_id = model_id)
        model1 = self.model_manager.get_model_by_name(model_name)

        # test + check        
        self.assertNotEqual(model1, None)

        for patch_start, patch_stop in [
                (self.patcher_db_get.start, self.patcher_db_get.stop),
                (self.patcher_db_remove.start, self.patcher_db_remove.stop)]:
            patch_start()

            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.delete_model(model_id)

            patch_stop()

    def test_model_manager_20_list_models_correct(self):
        """ Testing list method of model manager for correct request cases """

        self.model_manager.register_update_default_models()
        models = self.model_manager.list_models(verbose=False)
        self.assertIsInstance(models, list, 'Could not get list of models properly')

        # Check with verbose
        models = self.model_manager.list_models(verbose=True)
        self.assertIsInstance(models, list, 'Could not get list of models properly in verbose mode')

        # do some tests on the first model of models contained in database
        self.assertNotIn('model_path', models[0])
        self.assertNotIn('hash', models[0])
        self.assertNotIn('date_modified', models[0])
        self.assertNotIn('date_created', models[0])

        # check by sorting results ()
        # list of fields we are going to sort in alphabetical order
        sort_by_fields = ['date_last_action',
                          'model_type',
                          'model_status',
                          'algorithm', 
                          'researcher_id']

        for sort_by_field in sort_by_fields:
            models_sorted_by_modified_date = self.model_manager.list_models(sort_by=sort_by_field,
                                                                            )
            for i in range(len(models_sorted_by_modified_date) - 1):

                # do not compare if values extracted are set to None
                if models_sorted_by_modified_date[i][sort_by_field] is not None \
                   and models_sorted_by_modified_date[i + 1][sort_by_field] is not None:
                    self.assertLessEqual(models_sorted_by_modified_date[i][sort_by_field],
                                         models_sorted_by_modified_date[i + 1][sort_by_field])

        # check with results filtered on `model_status` field
        # first, register a model
        model_file_path = os.path.join(self.testdir, 'test-model-1.txt')

        self.model_manager.register_model(
            name='test-model-1',
            path=model_file_path,
            model_type='registered',
            description='desc'
        )
        # second, reject it 
        _, model_to_reject = self.model_manager.check_model_status(model_file_path, ModelTypes.REGISTERED)
        self.model_manager.reject_model(model_to_reject['model_id'])

        # action: gather only rejected models
        rejected_models = self.model_manager.list_models(select_status=TrainingPlanApprovalStatus.REJECTED)

        self.assertIn('test-model-1', [x['name'] for x in rejected_models])
        self.assertNotIn(TrainingPlanApprovalStatus.APPROVED.value,
                         [x['model_status'] for x in rejected_models])
        # gather only pending models (there should be no model with pending status,
        # so request returns empty list)
        pending_models = self.model_manager.list_models(select_status=TrainingPlanApprovalStatus.PENDING,
                                                        verbose=False)
        self.assertEqual(pending_models, [])

        # filtering with more than one status (get only REJECTED and APPROVAL model)
        # plus sorting
        for sort_by in [None, 'model_id', 'NON_EXISTING_KEY']:
            rejected_and_approved_models = self.model_manager.list_models(sort_by=sort_by,
                                                                          select_status=[TrainingPlanApprovalStatus.REJECTED,
                                                                                         TrainingPlanApprovalStatus.APPROVED],
                                                                          verbose=False)
            for model in rejected_and_approved_models:
                # Model status should be either Rejected or Approved...
                self.assertIn(model['model_status'],
                              [TrainingPlanApprovalStatus.REJECTED.value,
                               TrainingPlanApprovalStatus.APPROVED.value])
                # ... but not Pending
                self.assertNotEqual(model['model_status'], TrainingPlanApprovalStatus.PENDING.value)


    def test_model_manager_21_list_models_errors(self):
        """Test model manager `list_models` function for error cases.
        """

        # default models in database (not directly used, but to search among multiple entries)
        self.model_manager.register_update_default_models()

        model_name = 'mymodel_name'
        model_path = os.path.join(self.testdir, 'test-model-1.txt')

        # add one registered model in database
        self.model_manager.register_model(model_name, 'mymodel_description', model_path)


        # Test 1 : bad parameters
        for bad_sort in [True, 7, [], ['model_type'], {}, {'model_type': 'model_type'}]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.list_models(bad_sort, None, True, None)

        for bad_status in [True, 7, {}, {'model_status': 'model_status'}]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.list_models(None, bad_status, True, None)

        for bad_verbose in [None, 7, {}, [], ['model_verbose'], {'model_verbose': 'model_verbose'}]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.list_models(None, None, bad_verbose, None)

        # {} is ok, not search by criterion is performed
        for bad_search in [False, 7, [], ['model_search'], { 'dummy': 'model_id'},
                { 'by': 'model_id'}, { 'text': 'search text'}]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.list_models(None, None, True, bad_search)


        # Test 2 : database access error
        self.patcher_db_search.start()
        self.patcher_db_all.start()

        for status in [
                None,
                TrainingPlanApprovalStatus.PENDING,
                [],
                [TrainingPlanApprovalStatus.REJECTED, TrainingPlanApprovalStatus.APPROVED]]:
            for search in [None, {}, {'by': 'model_type', 'text': 'Registered'}]:
                for sort_by in [None, 'colonne']: # non existing entry
                    for verbose in [True, False]:
                        with self.assertRaises(FedbiomedModelManagerError):
                            self.model_manager.list_models(sort_by, status, verbose, search)

        self.patcher_db_search.stop()
        self.patcher_db_all.stop()

        # remaining uncovered case
        self.patcher_db_search.start()

        for verbose in [True, False]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.list_models('colonne', None, verbose, None)       

        self.patcher_db_search.stop()


    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('fedbiomed.node.model_manager.ModelManager.get_model_from_database')
    def test_model_manager_22_reply_model_status_request(self,
                                                         mock_get_model,
                                                         mock_download):
        """Tests model manager `reply_model_status_request` method (normal case scenarii)"""

        messaging = MagicMock()
        messaging.send_message.return_value = None
        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        mock_download.return_value = 200, None
        mock_get_model.return_value = {'model_status': TrainingPlanApprovalStatus.APPROVED.value}

        msg = {
            'researcher_id': 'ssss',
            'job_id': 'xxx',
            'training_plan_url': 'file:/' + environ['DEFAULT_MODELS_DIR'] + '/' + default_models[0],
            'command': 'training-plan-status'
        }
        # test 1: case where status code of HTTP request equals 200 AND model
        # has been approved
        self.model_manager.reply_model_status_request(msg, messaging)

        # check:
        messaging.send_message.assert_called_once_with({'researcher_id': 'ssss',
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': True,
                                                        'approval_obligation': True,
                                                        'status': TrainingPlanApprovalStatus.APPROVED.value,
                                                        'msg': "Model has been approved by the node," +
                                                        " training can start",
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })
        with self.assertRaises(FedbiomedMessageError):
            # should trigger a FedBiomedMessageError because 'researcher_id' should be a string
            # (and not a boolean)
            msg['researcher_id'] = True
            self.model_manager.reply_model_status_request(msg, messaging)

        # test 2: case where status code of HTTP request equals 200 AND model has
        # not been approved
        msg['researcher_id'] = 'dddd'

        for model_status, message in [
                (TrainingPlanApprovalStatus.REJECTED.value, 'Model has been rejected by the node, training is not possible'),
                (TrainingPlanApprovalStatus.PENDING.value, 'Model is pending: waiting for a review')]:
            # prepare
            mock_get_model.return_value = {'model_status': model_status}
            messaging.reset_mock()

            # test
            self.model_manager.reply_model_status_request(msg, messaging)

            # check
            messaging.send_message.assert_called_once_with({'researcher_id': 'dddd',
                                                            'node_id': environ['NODE_ID'],
                                                            'job_id': 'xxx',
                                                            'success': True,
                                                            'approval_obligation': True,
                                                            'status': model_status,
                                                            'msg': message,
                                                            'training_plan_url': msg['training_plan_url'],
                                                            'command': 'training-plan-status'
                                                            })

        # test 3: case where "MODEL_APPROVAL" has not been set 
        mock_get_model.return_value = {'model_status': TrainingPlanApprovalStatus.REJECTED.value}
        messaging.reset_mock()

        self.values["MODEL_APPROVAL"] = False
        test3_msg = 'This node does not require model approval (maybe for debuging purposes).'
        # action
        self.model_manager.reply_model_status_request(msg, messaging)

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
        mock_get_model.return_value = {'model_status': TrainingPlanApprovalStatus.PENDING.value}
        msg['researcher_id'] = '12345'

        messaging.reset_mock()
        self.model_manager.reply_model_status_request(msg, messaging)
        # check:
        messaging.send_message.assert_called_once_with({'researcher_id': msg['researcher_id'],
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': False,
                                                        'approval_obligation': False,
                                                        'status': 'Error',
                                                        'msg': f'Can not download model file. {msg["training_plan_url"]}',
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })

        # test 5: case where model is not registered

        for approval, message in [
                (False, 'This node does not require model approval (maybe for debuging purposes).'),
                (True, "Unknown model / model not in database (status Not Registered)")]:

            # prepare
            msg = {
                'researcher_id': 'ssss',
                'job_id': 'xxx',
                'training_plan_url': 'file:/' + os.path.join(self.testdir, 'test-model-1.txt'),
                'command': 'training-plan-status'
            }
            self.values["MODEL_APPROVAL"] = approval

            mock_download.return_value = 200, None
            mock_get_model.return_value = None
            messaging.reset_mock()

            # test
            self.model_manager.reply_model_status_request(msg, messaging)

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
    @patch('fedbiomed.node.model_manager.ModelManager.get_model_from_database')
    def test_model_manager_23_reply_model_status_request_exception(self,
                                                                   mock_get_model,
                                                                   mock_download):
        """
        Tests `reply_model_status_request` method when exceptions are occuring:
        - 1: by `Repository.download_file` (FedbiomedRepositoryError)
        - 2: by `ModelManager.check_is_model_approved` (Exception)
        Checks that message (that should be sent to researcher) is created accordingly
        to the triggered exception 

        """
        # test 1: tests that error triggered through `Repository.download)file` is
        # correctly handled
        # patches 
        messaging = MagicMock()
        messaging.send_message.return_value = None
        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        download_exception = FedbiomedRepositoryError("mimicking an exception triggered from"
                                                      "fedbiomed.common.repository")
        mock_download.side_effect = download_exception
        mock_get_model.return_value = True, {}

        msg = {
            'researcher_id': 'ssss',
            'job_id': 'xxx',
            'training_plan_url': 'file:/' + environ['DEFAULT_MODELS_DIR'] + '/' + default_models[0],
            'command': 'training-plan-status'
        }

        download_err_msg = ErrorNumbers.FB604.value + ': An error occurred when downloading model file.' + \
            f' {msg["training_plan_url"]} , {str(download_exception)}'

        # action
        self.model_manager.reply_model_status_request(msg, messaging)

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
        # test 2: test that error triggered through `check_model_status` method
        # of `ModelManager` is correctly handled

        # resetting `mock_download`
        mock_download.side_effect = None
        mock_download.return_value = 200, None
        messaging.reset_mock()
        # creating a new exception for `check_model_status` method
        checking_model_exception = FedbiomedModelManagerError("mimicking an exception happening when calling "
                                                              "'check_is_model_approved'")
        mock_get_model.side_effect = checking_model_exception

        checking_model_err_msg = ErrorNumbers.FB606.value + ': Cannot check if model has been registered.' + \
            f' Details {checking_model_exception}'
        # action
        self.model_manager.reply_model_status_request(msg, messaging)

        # check
        messaging.send_message.assert_called_once_with({'researcher_id': msg['researcher_id'],
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': False,
                                                        'approval_obligation': False,
                                                        'status': 'Error',
                                                        'msg': checking_model_err_msg,
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })

        # test 3: test that any other errors (inhereting from Exception) are caught
        messaging.reset_mock()
        # creating a new exception for `check_is_model_approved` method
        checking_model_exception_t3 = Exception("mimicking an exception happening when calling "
                                                "'check_model_status'")

        checking_model_err_msg_t3 = ErrorNumbers.FB606.value + ': An unknown error occured when downloading model ' +\
            f'file. {msg["training_plan_url"]} , {str(checking_model_exception_t3)}'

        mock_get_model.side_effect = checking_model_exception_t3
        # action
        self.model_manager.reply_model_status_request(msg, messaging)

        # check
        messaging.send_message.assert_called_once_with({'researcher_id': msg['researcher_id'],
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': False,
                                                        'approval_obligation': False,
                                                        'status': 'Error',
                                                        'msg': checking_model_err_msg_t3,
                                                        'training_plan_url': msg['training_plan_url'],
                                                        'command': 'training-plan-status'
                                                        })

    @patch('fedbiomed.node.model_manager.ModelManager._create_hash')
    @patch('os.path.getctime')
    @patch('os.path.getmtime')
    def test_model_manager_17_check_model_not_existing(self,
                                                       getmtime_patch,
                                                       getctime_patch,
                                                       create_hash_patch):
        """Test `_check_model_not_existing` function
        """
        # note: if _check_model_not_existing succeeds, it returns/changes nothing
        # so we don't have to do a `assert` in this case

        # Test 1 : test with empty database, no check should raise error
        for n in [None, 'dummy_name']:
            for p in [None, 'dummy_path']:
                for h in [None, 'dummy_hash']:
                    for a in [None, 'dummy_algorithm']:
                        self.assertIsNone(self.model_manager._check_model_not_existing(n, p, h, a))


        # Inter-test : add model in database for next tests
        model_name = 'mymodel_name'
        model_path = 'mymodel_path'
        model_hash = 'mymodel_hash'
        model_algorithm = 'mymodel_algorithm'
        # patching
        getctime_patch.value = 12345
        getmtime_patch.value = 23456

        def create_hash_side_effect(path):
            return model_hash, model_algorithm
        create_hash_patch.side_effect = create_hash_side_effect

        self.model_manager.register_model(model_name, 'mymodel_description', model_path)

        # Test 2 : test with 1 model in database, check with different values, no error
        for n in [None, 'dummy_name']:
            for p in [None, 'dummy_path']:
                for h in [None, 'dummy_hash']:
                    for a in [None, 'dummy_algorithm']:
                        self.assertIsNone(self.model_manager._check_model_not_existing(n, p, h, a))        

        # Test 3 : test with 1 model in database, check with existing value, error raised
        for n in [None, model_name]:
            for p in [None, model_path]:
                for h, a in [(None, None), (model_hash, model_algorithm)]:
                    if all(i is None for i in (n, p, h, a)):
                        # no error occurs if we don't check against any condition ...
                        continue
                    with self.assertRaises(FedbiomedModelManagerError):
                        self.model_manager._check_model_not_existing(n, p, h, a)

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
                        with self.assertRaises(FedbiomedModelManagerError):
                            self.model_manager._check_model_not_existing(n, p, h, a)

        # Final : empty database to enable proper cleaning
        with open(environ['DB_PATH'], 'w') as f:
            f.write('')        

    @patch('os.path.getctime')
    @patch('os.path.getmtime')
    def test_model_manager_24_check_model_status(self,
                                                 getmtime_patch,
                                                 getctime_patch):
        """Test `check_model_status` function
        """

        # patching
        getctime_patch.value = 12345
        getmtime_patch.value = 23456

        # default models in database (not directly used, but to search among multiple entries)
        self.model_manager.register_update_default_models()

        model_name = 'mymodel_name'
        model_path = os.path.join(self.testdir, 'test-model-1.txt')

        # add one registered model in database
        self.model_manager.register_model(model_name, 'mymodel_description', model_path)

        # Test 1 : successful search for registered model
        for status in [None, ModelTypes.REGISTERED, TrainingPlanApprovalStatus.APPROVED]:
            is_present, model = self.model_manager.check_model_status(model_path, status)
            self.assertIsInstance(is_present, bool)
            self.assertTrue(is_present)
            self.assertIsInstance(model, dict)
            self.assertEqual(model['name'], model_name)
            self.assertEqual(model['model_path'], model_path)

        # Test 2 : unsuccessful search for registered model
        for status in [ModelTypes.REQUESTED, ModelTypes.DEFAULT,
                       TrainingPlanApprovalStatus.PENDING, TrainingPlanApprovalStatus.REJECTED]:
            is_present, model = self.model_manager.check_model_status(model_path, status)
            self.assertIsInstance(is_present, bool)
            self.assertFalse(is_present)
            self.assertEqual(model, None)      

        # Test 3 : error because of bad status in search
        for status in [ 3, True, 'toto', {}, {'clef'}, [], ['un']]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.check_model_status(model_path, status)

        # Test 4 : error in database access
        self.patcher_db_get.start()

        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.check_model_status(model_path, None)

        self.patcher_db_get.stop()

    def test_model_manager_25_get_model_by_name(self):
        """Test `get_model_by_name` function
        """

        model_name = 'mymodel_name'
        model_path = os.path.join(self.testdir, 'test-model-1.txt')

        # default models in database (not directly used, but to search among multiple entries)
        self.model_manager.register_update_default_models()

        # add one registered model in database
        self.model_manager.register_model(model_name, 'mymodel_description', model_path)

        # Test 1 : look for existing model
        model = self.model_manager.get_model_by_name(model_name)
        self.assertIsInstance(model, dict)
        self.assertEqual(model['name'], model_name)

        # Test 2 : look for non existing model
        model = self.model_manager.get_model_by_name('ANY DUMMY YUMMY')
        self.assertEqual(model, None)

        # Test 3 : bad parameter errors
        for mpath in [None, 3, {}, { 'clef': model_path }, [], [model_path], self.Dummy, self.Dummy()]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.get_model_by_name(mpath)            

        # Test 4 : database access error
        self.patcher_db_get.start()

        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.get_model_by_name(model_name)

        self.patcher_db_get.stop()    

    def test_model_manager_26_get_model_from_database(self):
        """Test `get_model_from_database` function
        """

        model_name = 'mymodel_name'
        model_path = os.path.join(self.testdir, 'test-model-1.txt')
        same_model_path = os.path.join(environ['TMP_DIR'], 'test-model-1-2.txt')
        shutil.copy(model_path, same_model_path)

        # default models in database (not directly used, but to search among multiple entries)
        self.model_manager.register_update_default_models()

        # add one registered model in database
        self.model_manager.register_model(model_name, 'mymodel_description', model_path)

        # Test 1 : look for existing model
        for mpath in [model_path, same_model_path]:
            model = self.model_manager.get_model_from_database(mpath)
            self.assertTrue(isinstance(model, dict))
            self.assertEqual(model['name'], model_name) 

        # Test 2 : look for non existing model
        model = self.model_manager.get_model_from_database(os.path.join(self.testdir, 'test-model-2.txt'))
        self.assertIsNone(model)

        # Test 3 : bad parameter errors
        for mpath in [None, 3, {}, { 'clef': model_path }, [], [model_path], self.Dummy, self.Dummy()]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager.get_model_from_database(mpath)            

        # Test 4 : database access error
        self.patcher_db_get.start()

        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.get_model_from_database(model_path)

        self.patcher_db_get.stop()    

    def test_model_manager_27_get_model_by_id(self):
        """Test `get_model_by_id` function
        """

        model_name = 'mymodel_name'
        model_path = os.path.join(self.testdir, 'test-model-1.txt')
        model_id = 'mymodel_id_for_test'

        # default models in database (not directly used, but to search among multiple entries)
        self.model_manager.register_update_default_models()

        # add one registered model in database
        self.model_manager.register_model(model_name, 'mymodel_description', model_path, model_id = model_id)

        # Test 1 : look for existing model
        for secure in [True, False]:
            for content in [True, False]:
                model = self.model_manager.get_model_by_id(model_id, secure, content)
                self.assertIsInstance(model, dict)
                self.assertEqual(model['name'], model_name)
                self.assertEqual(model['model_id'], model_id)
                if not secure:
                    self.assertEqual(model['model_path'], model_path)

        # Test 2 : look for non existing model
        for secure in [True, False]:
            for content in [True, False]:
                model = self.model_manager.get_model_by_id('NON_EXISTING_MODEL_ID', secure, content)
                self.assertIsNone(model)

        # Test 3 : bad parameter errors
        for secure in [True, False]:
            for content in [True, False]:
                for mid in [None, 3, {}, { 'model_id': model_id }, [], [model_id], self.Dummy, self.Dummy()]:
                    with self.assertRaises(FedbiomedModelManagerError):
                        self.model_manager.get_model_by_id(mid, secure, content)            

        # Test 4 : database access error
        self.patcher_db_get.start()

        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.get_model_by_id(model_id, secure, content)

        self.patcher_db_get.stop()    

    @patch('fedbiomed.common.repository.Repository.download_file')
    def test_model_manager_28_reply_model_approval_request(self, download_file_patch):
        """Test model manager `reply_model_approval_request` function.
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

        # Test 1 : model approval for non existing model

        # prepare
        model_researcher_id = 'the researcher :%!#><|[]"*&$@!\'\\'
        model_description = 'the description :%!#><|[]"*&$@!\'\\'
        model_sequence = -4
        model_file = os.path.join(self.testdir, 'test-model-1.txt')
        msg = {
            'researcher_id': model_researcher_id,
            'description': model_description,
            'sequence': model_sequence,
            'training_plan_url': 'file:' + model_file,
            'command': 'approval'
        }

        # test
        model_before = self.model_manager.get_model_from_database(model_file)

        self.model_manager.reply_model_approval_request(msg, messaging)

        model_after = self.model_manager.get_model_from_database(model_file)

        # check
        messaging.send_message.assert_called_once_with({
            'researcher_id': model_researcher_id,
            'node_id': environ['NODE_ID'],
            'sequence': model_sequence,
            'status': 200,
            'command': 'approval',
            'success': True
        })
        self.assertEqual(model_before, None)
        self.assertTrue(isinstance(model_after, dict))
        self.assertEqual(model_after['description'], model_description)
        self.assertEqual(model_after['researcher_id'], model_researcher_id)
        self.assertEqual(model_after['model_type'], ModelTypes.REQUESTED.value)
        self.assertEqual(model_after['model_status'], TrainingPlanApprovalStatus.PENDING.value)

        # clean
        messaging.reset_mock()


        # Test 2 : model approval for existing model

        # prepare

        # different message : check that existing entry in database is not updated
        model2_researcher_id = 'another researcher id'
        model2_description = 'another description'
        model2_sequence = -4
        model2_file = model_file  # re-using model_id from test 1 for test 2
        model_id = model_after['model_id']
        msg2 = {
            'researcher_id': model2_researcher_id,
            'description': model2_description,
            'sequence': model2_sequence,
            'training_plan_url': 'file:' + model2_file,
            'command': 'approval'
        }

        def noaction_model(model_id, extra_notes):
            pass

        for approval_action, approval_status in [
                (noaction_model, TrainingPlanApprovalStatus.PENDING.value),
                (self.model_manager.approve_model, TrainingPlanApprovalStatus.APPROVED.value),
                (self.model_manager.reject_model, TrainingPlanApprovalStatus.REJECTED.value)]:
            # also update status
            approval_action(model_id, 'dummy notes')

            # test
            self.model_manager.reply_model_approval_request(msg2, messaging)

            model2_after = self.model_manager.get_model_from_database(model_file)

            # check
            messaging.send_message.assert_called_once_with({
                'researcher_id': model2_researcher_id,
                'node_id': environ['NODE_ID'],
                'sequence': model2_sequence,
                'status': 200,
                'command': 'approval',
                'success': True
            })
            # verify existing model in database did not change
            self.assertTrue(isinstance(model2_after, dict))
            self.assertEqual(model2_after['description'], model_description)
            self.assertEqual(model2_after['researcher_id'], model_researcher_id)
            self.assertEqual(model2_after['model_type'], ModelTypes.REQUESTED.value)
            self.assertEqual(model2_after['model_status'], approval_status)

            # clean
            messaging.reset_mock()


        # Test 3 : model approval with errors

        # prepare
        raise_filenotfound_error = FileNotFoundError('some additional information')
        self.patcher_shutil_move = patch('shutil.move', MagicMock(side_effect=raise_filenotfound_error))

        self.model_manager.delete_model(model_id)

        for error in ['download', 'db_get', 'db_upsert', 'file_move']:

            # test-specific prepare
            if error == 'download':
                download_file_patch.side_effect = FedbiomedRepositoryError('any error message')
                model3_status = 0

            # test
            model3_before = self.model_manager.get_model_from_database(model_file)

            # test-specific prepare
            if error == 'db_get':
                self.patcher_db_get.start()
            elif error == 'db_upsert':
                self.patcher_db_upsert.start()
            elif error == 'file_move':
                self.patcher_shutil_move.start()

            self.model_manager.reply_model_approval_request(msg, messaging)

            # test-specific clean
            if error == 'db_get':
                self.patcher_db_get.stop()
            elif error == 'db_upsert':
                self.patcher_db_upsert.stop()
            elif error == 'file_move':
                self.patcher_shutil_move.stop()

            model3_after = self.model_manager.get_model_from_database(model_file)

            # check
            messaging.send_message.assert_called_once_with({
                'researcher_id': model_researcher_id,
                'node_id': environ['NODE_ID'],
                'sequence': model_sequence,
                'status': model3_status,
                'command': 'approval',
                'success': False
            })
            self.assertEqual(model3_before, None)
            self.assertEqual(model3_after, None)

            # clean
            messaging.reset_mock()

            # test-specific clean
            if error == 'download':
                download_file_patch.side_effect = download_file_side_effect
                model3_status = 200

    def test_model_manager_29_update_model_status(self):
        """Test model manager `_update_model_status` function.
        """

        model_name = 'mymodel_name'
        model_path = os.path.join(self.testdir, 'test-model-1.txt')
        model_id = 'mymodel_id_for_test'

        # add one registered model in database
        self.model_manager.register_model(model_name, 'mymodel_description', model_path, model_id = model_id)


        # Test 1 : do correct update in existing model
        for status in [
            TrainingPlanApprovalStatus.PENDING,
            TrainingPlanApprovalStatus.REJECTED,
            TrainingPlanApprovalStatus.APPROVED, # update 2x with same status to cover this case
            TrainingPlanApprovalStatus.APPROVED]:
            note = str(status)
            self.model_manager._update_model_status(model_id, status, note)
            model = self.model_manager.get_model_by_id(model_id)

            self.assertEqual(model['model_status'], status.value)
            self.assertEqual(model['notes'], note)


        # Test 2 : do update in non-existing model
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager._update_model_status('non_existing_model', TrainingPlanApprovalStatus.REJECTED, 'new_notes')


        # Test 3 : bad parameters
        for bad_id in [None, 3, ['my_model'], [], {}, {'model_id': 'my_model'}]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager._update_model_status(bad_id, TrainingPlanApprovalStatus.REJECTED, 'new notes')

        for bad_status in [None, 5, [], ['mt_status'], {}, {'model_status': 'my_status'}, 'Approved']:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager._update_model_status(model_id, bad_status, 'new notes')

        for bad_notes in [5, [], ['mt_status'], {}, {'model_status': 'my_status'}]:
            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager._update_model_status(model_id, TrainingPlanApprovalStatus.REJECTED, bad_notes)


        # Test 4 : database access error
        for patch_start, patch_stop in [
                (self.patcher_db_get.start, self.patcher_db_get.stop),
                (self.patcher_db_update.start, self.patcher_db_update.stop)]:
            patch_start()

            with self.assertRaises(FedbiomedModelManagerError):
                self.model_manager._update_model_status(model_id, TrainingPlanApprovalStatus.REJECTED, 'new_notes')

            patch_stop()    

    def test_model_manager_30_remove_sensible_keys_from_request(self):
        """Test model manager `_remove_sensible_keys_from_request` function.
        """

        # prepare
        key_sensible = 'model_path'
        key_notsensible = 'model_id' 

        doc = { 
            key_sensible: 'valeur clef',
            key_notsensible: 'autre valeur'
        }
        doc2 = copy.deepcopy(doc)

        # test
        self.model_manager._remove_sensible_keys_from_request(doc2)

        # check
        self.assertEqual(doc2[key_notsensible], doc[key_notsensible])
        self.assertFalse(key_sensible in doc2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
