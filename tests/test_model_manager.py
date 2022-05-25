import builtins
import copy
from datetime import datetime
import os
import shutil
import tempfile
import unittest
import inspect
from unittest.mock import patch, MagicMock

import testsupport.mock_node_environ  # noqa (remove flake8 false warning)

from fedbiomed.common.constants import ErrorNumbers, HashingAlgorithms, ModelApprovalStatus, ModelTypes
from fedbiomed.common.exceptions import FedbiomedMessageError, FedbiomedModelManagerError, FedbiomedRepositoryError
from fedbiomed.common.logger import logger

from fedbiomed.node.environ import environ
from fedbiomed.node.model_manager import ModelManager


class TestModelManager(unittest.TestCase):
    """
    Unit tests for class `ModelManager` (from fedbiomed.node.model_manager)
    """

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
        # ---------------------------------------------------------------------------

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

    def test_model_manager_05_update_default_hashes_when_algo_is_changed(self):
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

    def test_model_manager_06_update_default_model_deleted(self):
        """Tests `update_default_model` when a model file that had been registered 
        has been deleted
        """
        file_path = os.path.join(self.testdir, 'test-model-1.txt')
        new_default_model_path = os.path.join(self.testdir, 'test-model-1-2.txt')
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

    def test_model_manager_07_update_modified_model_files(self):
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

    def test_model_manager_08_register_model(self):
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

        # Wrong types -------------------------------------
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

    @patch('fedbiomed.node.model_manager.ModelManager._create_hash')
    def test_model_manager_09_check_hashes_for_registerd_models(self,
                                                                create_hash_patch):
        """
        Tests `hashes_for_registered_models` method, with 3 settings
        - Test 1: no models are registered
        - Test 2: models are registered and are stored on computer
        - Test 3: model is no longer stored on computer.
        """
        # patch
        correct_hash = 'a correct hash'
        correct_hashing_algo = 'a correct hashing algorithm'
        create_hash_patch.return_value = correct_hash, correct_hashing_algo

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
        model_file_copied_path = os.path.join(self.testdir, 'copied-test-model-1.txt')
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

    def test_model_manager_10_checking_model_register(self):
        """ Testing check model is registered or not """

        model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')
        model_file_2 = os.path.join(self.testdir, 'test-model-2.txt')

        self.model_manager.register_model(
            name='test-model',
            path=model_file_1,
            model_type=ModelTypes.REGISTERED.value,
            description='desc'
        )

        # Load default datasets
        self.model_manager.register_update_default_models()

        # Test when model is not registered (ie not present in the database)
        approve, model = self.model_manager.check_is_model_registered(model_file_2)
        self.assertFalse(approve, "Model has been registered but it hasnot been registered")
        self.assertIsNone(model, "Model has been registered but it hasnot been registered")

        # Test when model is a registered (either registered or default)
        approve, model = self.model_manager.check_is_model_registered(model_file_1)
        self.assertTrue(approve, "Model hasn't been registered but it should have been")
        self.assertIsNotNone(model, "Model hasn't been registered but it should have been")

        # Test when default models is not allowed / not approved
        self.values['ALLOW_DEFAULT_MODELS'] = False

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:
            model_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            approve, model = self.model_manager.check_is_model_registered(model_path)
            self.assertFalse(approve, "Model has been registered but it shouldn't have been")
            self.assertIsNone(model, "Model has been registered but it shouldn't have been")

    def test_model_manager_11_check_if_model_requested(self):
        """Tests if model has been requested or not"""
        model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')
        model_file_2 = os.path.join(self.testdir, 'test-model-2.txt')

        self.model_manager.register_model(
            name='test-model',
            path=model_file_1,
            model_type=ModelTypes.REQUESTED.value,
            description='desc'
        )

        self.assertTrue(self.model_manager.check_is_model_requested(model_file_1))

        self.assertFalse(self.model_manager.check_is_model_requested(model_file_2))

        # adding model_file_2 to database as registered
        self.model_manager.register_model(
            name='test-model-2',
            path=model_file_2,
            model_type=ModelTypes.REGISTERED.value,
            description='desc'
        )

        self.assertFalse(self.model_manager.check_is_model_requested(model_file_2))
        
        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:
            model_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            self.assertFalse(self.model_manager.check_is_model_requested(model_path))

    def test_model_manager_12_create_txt_model_from_py(self):
        
        # initialisation: creating a *.py file
        randomfolder = tempfile.mkdtemp()
        if not os.access(randomfolder, os.W_OK):
            self.skipTest("Test skipped cause temporary directory not writtable")
        else:
            file = 'model.py'
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
            ## tests if `txt_model` has a *.txt extension
            _, ext = os.path.splitext(txt_model_path)
            self.assertEqual(ext, '.txt')

            # check if content is the same in *.txt file and in *.py file
            with open(txt_model_path, 'r') as f:
                code = f.read()

            self.assertEqual(code, code_source)

    def test_model_manager_13_update_model_normal_case(self, ):
        """Tests method `update_model` in the normal case scenario"""

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
        with (patch.object(ModelManager, '_create_hash',
                           return_value=(model_hash, model_hashing_algorithm)),
              patch.object(os.path, 'getmtime', return_value=file_modification_date_timestamp),
              patch.object(os.path, 'getctime', return_value=file_creation_date_timestamp)):
            self.model_manager.update_model_hash('test-model-id', default_model_file_1)

        # checks
        # first, we are accessing to the updated model
        updated_model = self.model_manager._db.get(self.model_manager._database.name == 'test-model')

        # we are then checking that each entry in the database is correct
        self.assertEqual(updated_model['hash'], model_hash)
        self.assertEqual(updated_model['algorithm'], model_hashing_algorithm)
        self.assertEqual(updated_model['date_modified'], file_modification_date_literal)
        self.assertEqual(updated_model['date_created'], file_creation_date_literal)
        self.assertEqual(updated_model['model_path'], default_model_file_1)

    def test_model_manager_14_update_model_exception(self):
        """Tests method `update_model` """
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

    def test_model_manager_15_delete_registered_models(self):
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

    def test_model_manager_14_list_models(self):
        """ Testing list method of model manager """

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
        ## list of fields we are going to sort in alphabetical order
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
        ## first, register a model
        model_file_path = os.path.join(self.testdir, 'test-model-1.txt')

        self.model_manager.register_model(
            name='test-model-1',
            path=model_file_path,
            model_type='registered',
            description='desc'
        )
        ## second, reject it 
        _, model_to_reject = self.model_manager.check_is_model_registered(model_file_path)
        self.model_manager.reject_model(model_to_reject['model_id'])
        
        # action: gather only rejected models
        rejected_models = self.model_manager.list_models(select_status=ModelApprovalStatus.REJECTED)
        
        self.assertIn('test-model-1', [x['name'] for x in rejected_models])
        self.assertNotIn(ModelApprovalStatus.APPROVED.value,
                         [x['model_status'] for x in rejected_models])
        # gather only pending models (there should be no model with pending status,
        # so request returns empty list)
        pending_models = self.model_manager.list_models(select_status=ModelApprovalStatus.PENDING,
                                                        verbose=False)
        self.assertEqual(pending_models, [])
        
        ## filtering with more than one status (get only REJECTED and APPROVAL model)
        rejected_and_approved_models = self.model_manager.list_models(select_status=[ModelApprovalStatus.REJECTED,
                                                                                     ModelApprovalStatus.APPROVED],
                                                                      verbose=False)
        for model in rejected_and_approved_models:
            # Model status should be either Rejected or Approved...
            self.assertIn(model['model_status'],
                          [ModelApprovalStatus.REJECTED.value,
                           ModelApprovalStatus.APPROVED.value])
            # ... but not Pending
            self.assertNotEqual(model['model_status'], ModelApprovalStatus.PENDING.value)

    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('fedbiomed.node.model_manager.ModelManager.get_model_from_database')
    def test_model_manager_15_reply_model_status_request(self,
                                                         mock_get_model,
                                                         mock_download):
        """Tests model manager `reply_model_status_request` method (normal case scenarii)"""

        messaging = MagicMock()
        messaging.send_message.return_value = None
        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        mock_download.return_value = 200, None
        mock_get_model.return_value = {'model_status': ModelApprovalStatus.APPROVED.value}

        msg = {
            'researcher_id': 'ssss',
            'job_id': 'xxx',
            'model_url': 'file:/' + environ['DEFAULT_MODELS_DIR'] + '/' + default_models[0],
            'command': 'model-status'
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
                                                        'status': ModelApprovalStatus.APPROVED.value,
                                                        'msg': "Model has been approved by the node," +
                                                        " training can start",
                                                        'model_url': msg['model_url'],
                                                        'command': 'model-status'
                                                        })
        with self.assertRaises(FedbiomedMessageError):
            # should trigger a FedBiomedMessageError because 'researcher_id' should be a string
            # (and not a boolean)
            msg['researcher_id'] = True
            self.model_manager.reply_model_status_request(msg, messaging)

        # test 2: case where status code of HTTP request equals 200 AND model has
        # not been approved
        msg['researcher_id'] = 'dddd'
        mock_get_model.return_value = {'model_status': ModelApprovalStatus.REJECTED.value}
        messaging.reset_mock()
        # action
        self.model_manager.reply_model_status_request(msg, messaging)

        # check
        messaging.send_message.assert_called_once_with({'researcher_id': 'dddd',
                                                        'node_id': environ['NODE_ID'],
                                                        'job_id': 'xxx',
                                                        'success': True,
                                                        'approval_obligation': True,
                                                        'status': ModelApprovalStatus.REJECTED.value,
                                                        'msg': "Model has been rejected by the node," +
                                                        " training is not possible",
                                                        'model_url': msg['model_url'],
                                                        'command': 'model-status'
                                                        })
        # test 3: case where "MODEL_APPROVAL" has not been set 
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
                                                        'status': ModelApprovalStatus.REJECTED.value,
                                                        'msg': test3_msg,
                                                        'model_url': msg['model_url'],
                                                        'command': 'model-status'
                                                        })

        # test 4: case where status code of HTTP request equals 404 (request failed)
        mock_download.return_value = 404, None
        mock_get_model.return_value = {'model_status': ModelApprovalStatus.PENDING.value}
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
                                                        'msg': f'Can not download model file. {msg["model_url"]}',
                                                        'model_url': msg['model_url'],
                                                        'command': 'model-status'
                                                        })

    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('fedbiomed.node.model_manager.ModelManager.get_model_from_database')
    def test_model_manager_16_reply_model_status_request_exception(self,
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
            'model_url': 'file:/' + environ['DEFAULT_MODELS_DIR'] + '/' + default_models[0],
            'command': 'model-status'
        }

        download_err_msg = ErrorNumbers.FB604.value + ': An error occured when downloading model file.' + \
                           f' {msg["model_url"]} , {str(download_exception)}'

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
                                                        'model_url': msg['model_url'],
                                                        'command': 'model-status'
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
                                                        'model_url': msg['model_url'],
                                                        'command': 'model-status'
                                                        })

        # test 3: test that any other errors (inhereting from Exception) are caught
        messaging.reset_mock()
        # creating a new exception for `check_is_model_approved` method
        checking_model_exception_t3 = Exception("mimicking an exception happening when calling "
                                                "'check_model_status'")

        checking_model_err_msg_t3 = ErrorNumbers.FB606.value + ': An unknown error occured when downloading model file.' +\
            f' {msg["model_url"]} , {str(checking_model_exception_t3)}'

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
                                                        'model_url': msg['model_url'],
                                                        'command': 'model-status'
                                                        })


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
