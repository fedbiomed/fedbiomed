import os
import shutil
import unittest
import inspect
from unittest.mock import patch, MagicMock
from fedbiomed.common.exceptions import FedbiomedModelManagerError

import testsupport.mock_node_environ
from datetime import datetime
from fedbiomed.node.environ import environ
from fedbiomed.node.model_manager import ModelManager, HASH_FUNCTIONS
from fedbiomed.common.constants import HashingAlgorithms
from fedbiomed.common.logger import logger


class TestModelManager(unittest.TestCase):
    """
    Test `ModelManager`
    Args:
        unittest ([type]): [description]
    """

    # before the tests
    def setUp(self):

        # This part important for setting fake values for environ -----------------
        # and properly mocking values in environ Since environ is singleton,
        # you should also mock environ objects that is called from modolues e.g
        # fedbiomed.node.model_manager.environ you should use another mock for
        # the environ object used in test functions
        self.values = environ
        def side_effect(arg):
            return self.values[arg]

        def side_effect_set_item(key, value):
            self.values[key] = value

        # self.environ_patch = patch('fedbiomed.node.environ.environ')
        self.environ_model_manager_patch = patch('fedbiomed.node.model_manager.environ')

        # self.environ = self.environ_patch.start()
        self.environ_model = self.environ_model_manager_patch.start()

        # self.environ.__getitem__.side_effect = side_effect
        # self.environ.__setitem__.side_effect = side_effect_set_item

        self.environ_model.__getitem__.side_effect = side_effect
        self.environ_model.__setitem__.side_effect = side_effect_set_item
        # ---------------------------------------------------------------------------

        # Build ModelManger
        self.model_manager = ModelManager()

        # get test directory to acces test-model files
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
#        self.environ_patch.stop()
        self.environ_model_manager_patch.stop()

        self.model_manager._tinydb.drop_table('Models')
        self.model_manager._tinydb.close()
        os.remove(environ['DB_PATH'])

    def test_model_manager_01_create_default_model_hashes(self):

        """ Testing whether created hash for model files are okay
        or not. It also tests every default with each provided hashing algorithim
        """
        # We should import environ to get fake values
        #fromfedbiomed.node.environ import environ


        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        logger.info('Controlling Models Dir')
        logger.info(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:

            #set default hashing algorithm
            environ['HASHING_ALGORITHM'] = 'SHA256'
            full_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)


            # Control return vlaues with default hashing algorithm
            hash, algortihm = self.model_manager._create_hash(full_path)
            self.assertIsInstance(hash, str , 'Hash creation is not successful')
            self.assertEqual(algortihm, 'SHA256' , 'Wrong hashing algorithm')

            algortihms = HashingAlgorithms.list()
            for algo in algortihms:
                self.values['HASHING_ALGORITHM'] = algo
                hash, algortihm = self.model_manager._create_hash(full_path)
                self.assertIsInstance(hash, str , 'Hash creation is not successful')
                self.assertEqual(algortihm, algo , 'Wrong hashing algorithm')

                # Test unkown hashing algorithm
                # with self.assertRaises(FedbiomedModelManagerError):
                #     hash, algortihm = self.model_manager._create_hash(full_path, 'sss')

    def test_model_manager_02_update_default_hashes_when_algo_is_changed(self):

        """  Testing method for update/register default models when hashing
             algorithm has changed
        """
        # We should import environ to get fake values



        # Single test with default hash algorithm
        self.model_manager.register_update_default_models()

        # # Multiple test with different hashing algorithms
        algortihms = HashingAlgorithms.list()
        for algo in algortihms:
            self.values['HASHING_ALGORITHM'] = algo
            self.model_manager.register_update_default_models()
            doc = self.model_manager._db.get(self.model_manager._database.model_type == "default")
            logger.info(doc)
            self.assertEqual(doc["algorithm"], algo,
                             'Hashes are not properly updated after hashing algorithm is changed')

    def test_model_manager_03_update_modified_model_files(self):

        """ Testing update of modified default models """

        # We should import environ to get fake values

        logger.info('Print Infoooo')
        logger.info(environ['DB_PATH'])

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])

        # Test with only first file

        for model in default_models:

            file_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            self.model_manager.register_update_default_models()
            doc = self.model_manager._db.get(self.model_manager._database.model_path == file_path)

            # Open the file in append & read mode ('a+')
            with open(file_path, "a+") as file:
                lines = file.readlines()     # lines is list of line, each element '...\n'
                lines.insert(0, "\nprint('Hello world') \t \n")  # you can use any index if you know the line index
                file.seek(0)                 # file pointer locates at the beginning to write the whole file again
                file.writelines(lines)

            self.model_manager.register_update_default_models()
            docAfter = self.model_manager._db.get(self.model_manager._database.model_path == file_path)

            self.assertNotEqual(doc['hash'] , docAfter['hash'] , "Hash couldn't updated after file has modified")



    def test_model_manager_04_register_model(self):

        """ Testing registering method for new models """

        # We should import environ to get fake values

        self.model_manager.register_update_default_models()

        model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')
        model_file_2 = os.path.join(self.testdir, 'test-model-2.txt')

        self.model_manager.register_model(
            name = 'test-model',
            path = model_file_1,
            model_type = 'registered',
            description=  'desc'
        )

        # When same model file wants to be added it should raise and exception
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                                        name = 'test-model-2',
                                        path = model_file_1,
                                        model_type = 'registered',
                                        description=  'desc')

        # When same model wants to be added with same name  and different file
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                                        name = 'test-model',
                                        path = model_file_2,
                                        model_type = 'registered',
                                        description=  'desc')

        # Wrong types -------------------------------------
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                                        name = 'test-model-2',
                                        path = model_file_2,
                                        model_type = False,
                                        description=  'desc')
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                                        name = 'tesasdsad',
                                        path = model_file_2,
                                        model_type = False,
                                        description=  False)
        # Worng model type
        with self.assertRaises(FedbiomedModelManagerError):
            self.model_manager.register_model(
                                        name = 'tesasdsad',
                                        path = model_file_2,
                                        model_type = '123123',
                                        description=  'desc')

    @patch('fedbiomed.node.model_manager.ModelManager._create_hash')
    def test_model_manager_05_check_hashes_for_registerd_models(self,
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
            name = 'test-model',
            path = model_file_copied_path,
            model_type = 'registered',
            description=  'desc',
            model_id = 'test-model-id'
        )
        
        # update database with non-existing hash and hashing algorithms
        models = self.model_manager._db.search(self.model_manager._database.model_type.all('registered'))
        for model in models:
            self.model_manager._db.update( {'hash' : "an incorrect hash",
                                            'algorithm' : "incorrect_hashing_algorithm" },
                                        self.model_manager._database.model_id.all(model["model_id"]))
            
        self.model_manager.check_hashes_for_registered_models()
        
        # checks
        models = self.model_manager._db.search(self.model_manager._database.model_type.all('registered'))
        
        for model in models:

            self.assertEqual(model['hash'], correct_hash)
            self.assertEqual(model['algorithm'], correct_hashing_algo)
        
        # Test 3: here we are testing that a file that has been removed on
        # the system is also removed from the database
        
        # remove a model on the system
        os.remove(model_file_copied_path)
       
        # action
        self.model_manager.check_hashes_for_registered_models()
        removed_model = self.model_manager._db.get(self.model_manager._database.name == 'test-model')
        # check that the model has been removed
        self.assertIsNone(removed_model)

    def test_model_manager_06_checking_model_approve(self):
        """ Testing check model is approved or not """

        model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')
        model_file_2 = os.path.join(self.testdir, 'test-model-2.txt')

        self.model_manager.register_model(
            name = 'test-model',
            path = model_file_1,
            model_type = 'registered',
            description=  'desc',
        )

        # Load default datasets
        self.model_manager.register_update_default_models()

        # Test when model is not approved
        approve, model = self.model_manager.check_is_model_approved(model_file_2)
        self.assertFalse(approve,  "Model has been approved but it it shouldn't have been")
        self.assertIsNone(model, "Model has been approved but it it shouldn't have been")

        # Test when model is approved model
        approve, model = self.model_manager.check_is_model_approved(model_file_1)
        self.assertTrue(approve , "Model hasn't been approved but it should have been")
        self.assertIsNotNone(model , "Model hasn't been approved but it should have been")


        # Test when default models is not allowed / not approved
        environ['ALLOW_DEFAULT_MODELS'] = False

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:
            model_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            approve, model = self.model_manager.check_is_model_approved(model_path)
            self.assertFalse(approve , "Model has been approved but it shouldn't have been")
            self.assertIsNone(model , "Model has been approved but it shouldn't have been")

    #@patch('fedbiomed.node.model_manager.ModelManager._create_hash')
    def test_model_manager_07_update_model_normal_case(self,):
                                                       #create_hash_patch
                                                       #):
        """Tests method `update_model` in the normal case scenario"""
        
        # database initialisation
        default_model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')
        self.model_manager.register_model(
            name = 'test-model',
            path = default_model_file_1,
            model_type = 'registered',
            description=  'desc',
            model_id = 'test-model-id'
        )
        file_modification_date = 987654321.1234567
        file_creation_date = '1234567890.1234567'
        model_hash = 'a hash'
        model_hashing_algorithm = 'a_hashing_algorithm'

        # patches
        #create_hash_patch.return_value = model_hash, model_hashing_algorithm
        
        # action
        with (patch.object(ModelManager, '_create_hash', return_value=(model_hash, model_hashing_algorithm)),
              patch.object(os.path, 'getmtime', return_value=file_modification_date) as getmtime_mock):

            self.model_manager.update_model('test-model-id', default_model_file_1)

        # checks
        updated_model = self.model_manager._db.get(self.model_manager._database.name == 'test-model')
        
        self.assertEqual(updated_model['hash'], model_hash)
        self.assertEqual(updated_model['date_modified'], datetime.fromtimestamp(file_modification_date).strftime("%d-%m-%Y %H:%M:%S.%f"))
        
    def test_model_manager_08_update_model_exception(self):
        pass

    def test_model_manager_08_delete_registered_models(self):

        """ Testing delete opration for model manager """


        model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')

        self.model_manager.register_model(
            name = 'test-model-1',
            path = model_file_1,
            model_type = 'registered',
            description=  'desc'
        )

        # Get registered model
        model_1 = self.model_manager._db.get(self.model_manager._database.name == 'test-model-1')
        
        # Delete model
        self.model_manager.delete_model(model_1['model_id'])

        # Check model is removed
        model_1_r = self.model_manager._db.get(self.model_manager._database.name == 'test-model-1')
        self.assertIsNone(model_1_r , "Registered model is not removed")

        # Load default models
        self.model_manager.register_update_default_models()

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:
            model_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            model = self.model_manager._db.get(self.model_manager._database.model_path == model_path)

            # Check delete method removed default models (it shouldnt)
            with self.assertRaises(Exception):
                self.model_manager.delete_model(model['model_id'])

    def test_model_manager_09_list_appoved_models(self):
        """ Testing list method of model manager """

        self.model_manager.register_update_default_models()
        models = self.model_manager.list_approved_models(verbose=False)
        self.assertIsInstance(models, list , 'Could not get list of models properly')

        # Check with verbose
        models = self.model_manager.list_approved_models(verbose=True)
        self.assertIsInstance(models, list , 'Could not get list of models properly in verbose mode')

    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    def test_model_manager_10_reply_model_status_request(self,
                                                         mock_checking,
                                                         mock_download):


        messaging = MagicMock()
        messaging.send_message.return_value = None
        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        mock_download.return_value = 200, None
        mock_checking.return_value = True, {}

        msg = {
            'researcher_id' : 'ssss',
            'job_id' : 'xxx',
            'model_url' : 'file:/' + environ['DEFAULT_MODELS_DIR'] + '/' + default_models[0],
            'command' : 'model-status'
        }
        self.model_manager.reply_model_status_request( msg, messaging)

        with self.assertRaises(Exception):
            msg['researcher_id'] = True
            self.model_manager.reply_model_status_request( msg, messaging)

        mock_download.return_value = 404, None
        mock_checking.return_value = True, {}
        msg['researcher_id'] = '12345'
        self.model_manager.reply_model_status_request( msg, messaging)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
