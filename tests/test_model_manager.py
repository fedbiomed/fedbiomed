# Managing NODE, RESEARCHER environ mock before running tests
from array import array
from testsupport.delete_environ import delete_environ
# Detele environ. It is necessary to rebuild environ for required component
delete_environ()
import testsupport.mock_common_environ
# Import environ for node, since tests will be runing for node component
from fedbiomed.node.environ import environ


from fedbiomed.common.constants import HashingAlgorithms
import os
from fedbiomed.node.model_manager import ModelManager
import unittest
import inspect


class TestMonitor(unittest.TestCase):
    """
    Test `Monitormanager` 
    Args:
        unittest ([type]): [description]
    """

    # before the tests
    def setUp(self):

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

        # Set default 
        environ['HASHING_ALGORITHM'] = "SHA256"

        # DB should be removed after each test to
        # have clear database for tests
        os.remove(environ['DB_PATH'])
        
        pass

    def test_create_default_model_hashes(self):

        """ Testing whether created hash for model files are okay 
        or not. It also tests every default with each provided hashing algorithim 
        """

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])

        for model in default_models:
            
            #set default hashing algorithm
            environ['HASHING_ALGORITHM'] = 'SHA256'


            full_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            
            # Create has with default hashing algorithm
            hash, algortihm = self.model_manager._create_hash(full_path)    
            self.assertIsInstance(hash, str , 'Hash creation is not successful')
            self.assertEqual(algortihm, 'SHA256' , 'Wrong hashing algorithm')

            # Create has with each provided hashing algorithm

            algortihms = HashingAlgorithms.list()
            for algo in algortihms:
                environ['HASHING_ALGORITHM'] = algo
                hash, algortihm = self.model_manager._create_hash(full_path)    
                self.assertIsInstance(hash, str , 'Hash creation is not successful')
                self.assertEqual(algortihm, algo , 'Wrong hashing algorithm')

                # Test unkown hashing algorithm
                environ['HASHING_ALGORITHM'] = 'sss' # Undefined hashing algorithm
                with self.assertRaises(Exception):
                    hash, algortihm = self.model_manager._create_hash(full_path)  
          
                
    def test_update_default_hashes_when_algo_is_changed(self):

        """  Testing method for update/register default models when hashing
             algorithm has changed
        """

        # Single test with default hash algorithm 
        self.model_manager.register_update_default_models()

        # Multiple test with different hashing algorithms
        algortihms = HashingAlgorithms.list()
        for algo in algortihms:
            environ['HASHING_ALGORITHM'] = algo
            self.model_manager.register_update_default_models()
            doc = self.model_manager.db.get(self.model_manager.database.model_type == "default")
            self.assertEqual(doc["algorithm"], algo, 'Hashes are not properly updated after hashing algorithm is changed')
        

    def test_update_modified_model_files(self):
        
        """ Testing update of modified default models """

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])

        # Test with only first file 

        for model in default_models:

            file_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            self.model_manager.register_update_default_models()
            doc = self.model_manager.db.get(self.model_manager.database.model_path == file_path)

            # Open the file in append & read mode ('a+')
            with open(file_path, "a+") as file:
                lines = file.readlines()     # lines is list of line, each element '...\n'
                lines.insert(0, "\nprint('Hello world') \t \n")  # you can use any index if you know the line index
                file.seek(0)                 # file pointer locates at the beginning to write the whole file again
                file.writelines(lines)
            
            self.model_manager.register_update_default_models()
            docAfter = self.model_manager.db.get(self.model_manager.database.model_path == file_path)

            self.assertNotEqual(doc['hash'] , docAfter['hash'] , "Hash couldn't updated after file has modified")

        

    def test_register_model(self):
        
        """ Testing registering method for new models """

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

        model_file_1 = os.path.join(self.testdir, 'test-model-1.txt')
        model_file_2 = os.path.join(self.testdir, 'test-model-2.txt')

        self.model_manager.register_model(
            name = 'test-model-1',
            path = model_file_1,
            model_type = 'registered',
            description=  'desc'
        )

        # Get registered model
        model_1 = self.model_manager.db.get(self.model_manager.database.name == 'test-model-1')
       
        # Delete model 
        self.model_manager.delete_model(model_1['model_id'])
        
        # Check model is removed
        model_1_r = self.model_manager.db.get(self.model_manager.database.name == 'test-model-1')
        self.assertEqual(model_1_r , None , "Registered model is not removed")

        # Load default models 
        self.model_manager.register_update_default_models()

        default_models = os.listdir(environ['DEFAULT_MODELS_DIR'])
        for model in default_models:
            model_path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            model = self.model_manager.db.get(self.model_manager.database.model_path == model_path)
            
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



if __name__ == '__main__':  # pragma: no cover
    unittest.main()