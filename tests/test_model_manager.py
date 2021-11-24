
# Managing NODE, RESEARCHER environ mock before running tests
from fedbiomed.node.environ import environ
from testsupport.environ_fake import Environ
from fedbiomed.common.constants import HashingAlgorithms
import os
from fedbiomed.node.model_manager import ModelManager
import unittest
import inspect
from unittest.mock import patch, MagicMock
from fedbiomed.common.logger import logger

class TestModelManager(unittest.TestCase):
    """
    Test `Monitormanager` 
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
        self.values = Environ().values()
        def side_effect(arg):
            return self.values[arg]

        def side_effect_set_item(key, value):
            self.values[key] = value

        self.environ_patch = patch('fedbiomed.node.environ.environ')
        self.environ_model_manager_patch = patch('fedbiomed.node.model_manager.environ')

        self.environ = self.environ_patch.start()
        self.environ_model = self.environ_model_manager_patch.start()

        self.environ.__getitem__.side_effect = side_effect
        self.environ.__setitem__.side_effect = side_effect_set_item

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
        pass

    # after the tests
    def tearDown(self):
        # DB should be removed after each test to
        # have clear database for tests
        self.environ_patch.stop()
        self.environ_model_manager_patch.stop()
        
        self.model_manager._tinydb.drop_table('Models')

        pass

    def test_create_default_model_hashes(self):

        """ Testing whether created hash for model files are okay 
        or not. It also tests every default with each provided hashing algorithim 
        """
        # We should import environ to get fake values 
        from fedbiomed.node.environ import environ


        self.model_manager = ModelManager()
        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        logger.info('Controlling Models Dir')
        logger.info(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:
            
            #set default hashing algorithm
            environ['HASHING_ALGORITHM'] = 'SHA256'
            full_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            

            # Create has with default hashing algorithm
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
                with self.assertRaises(Exception):
                    hash, algortihm = self.model_manager._create_hash(full_path, 'sss')  
                        
    def test_update_default_hashes_when_algo_is_changed(self):

        """  Testing method for update/register default models when hashing
             algorithm has changed
        """
        # We should import environ to get fake values 
        from fedbiomed.node.environ import environ
        


        # Single test with default hash algorithm 
        self.model_manager.register_update_default_models()

        # # Multiple test with different hashing algorithms
        algortihms = HashingAlgorithms.list()
        for algo in algortihms:
            self.values['HASHING_ALGORITHM'] = algo
            self.model_manager.register_update_default_models()
            doc = self.model_manager._db.get(self.model_manager._database.model_type == "default")
            logger.info(doc)
            self.assertEqual(doc["algorithm"], algo, 'Hashes are not properly updated after hashing algorithm is changed')

    def test_update_modified_model_files(self):
        
        """ Testing update of modified default models """
        
        # We should import environ to get fake values 
        from fedbiomed.node.environ import environ

        logger.info('Print Infoooo')
        logger.info(environ['DB_PATH'])
        self.model_manager = ModelManager()

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

        

    def test_register_model(self):
        
        """ Testing registering method for new models """

        # We should import environ to get fake values 
        from fedbiomed.node.environ import environ

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
        with self.assertRaises(Exception):
            self.model_manager.register_model(
                                        name = 'test-model-2',
                                        path = model_file_1,
                                        model_type = 'registered',
                                        description=  'desc')

        # When same model wants to be added with same name  and different file
        with self.assertRaises(Exception):
            self.model_manager.register_model(
                                        name = 'test-model',
                                        path = model_file_2,
                                        model_type = 'registered',
                                        description=  'desc')

        # Wrong types -------------------------------------
        with self.assertRaises(Exception):
            self.model_manager.register_model(
                                        name = 'test-model-2',
                                        path = model_file_2,
                                        model_type = False,
                                        description=  'desc')
        with self.assertRaises(Exception):
            self.model_manager.register_model(
                                        name = 'tesasdsad',
                                        path = model_file_2,
                                        model_type = False,
                                        description=  False)
        # Worng model type 
        with self.assertRaises(Exception):
            self.model_manager.register_model(
                                        name = 'tesasdsad',
                                        path = model_file_2,
                                        model_type = '123123',
                                        description=  'desc')


    def test_checking_model_approve(self):

        """ Testing check model is approved or not """
        from fedbiomed.node.environ import environ

        model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')
        model_file_2 = os.path.join(self.testdir, 'test-model-2.txt')

        self.model_manager.register_model(
            name = 'test-model',
            path = model_file_1,
            model_type = 'registered',
            description=  'desc'
        )

        # Load default datasets
        self.model_manager.register_update_default_models()

        # Test when model is not approved
        approve, model = self.model_manager.check_is_model_approved(model_file_2)
        self.assertEqual(approve , False , "Model has been approved but it it shouldn't have been")
        self.assertEqual(model , None , "Model has been approved but it it shouldn't have been")

        # Test when model is approved model
        approve, model = self.model_manager.check_is_model_approved(model_file_1)
        self.assertEqual(approve , True , "Model hasn't been approved but it should have been")
        self.assertIsInstance(model , object , "Model hasn't been approved but it should have been")

        
        # Test when default models is not allowed / not approved
        environ['ALLOW_DEFAULT_MODELS'] = False
        
        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:
            model_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            approve, model = self.model_manager.check_is_model_approved(model_path)
            self.assertEqual(approve , False , "Model has been approved but it shouldn't have been")
            self.assertEqual(model , None , "Model has been approved but it shouldn't have been")

    def test_delete_registered_models(self):

        """ Testing delete opration for model manager """
        from fedbiomed.node.environ import environ


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
        self.assertEqual(model_1_r , None , "Registered model is not removed")

        # Load default models 
        self.model_manager.register_update_default_models()

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:
            model_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            model = self.model_manager._db.get(self.model_manager._database.model_path == model_path)
            
            # Check delete method removed default models (it shouldnt)
            with self.assertRaises(Exception):
                self.model_manager.delete_model(model['model_id'])

    def test_list_models(self):
        
        """ Testing list method of model manager """

        self.model_manager.register_update_default_models()
        models = self.model_manager.list_approved_models(verbose=False)
        self.assertIsInstance(models, list , 'Could not get list of models properly')

        # Check with verbose
        models = self.model_manager.list_approved_models(verbose=True)
        self.assertIsInstance(models, list , 'Could not get list of models properly in verbose mode')

    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    def test_reply_model_status_request(self, mock_checking , mock_download):
        
        from fedbiomed.node.environ import environ


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