# Managing NODE, RESEARCHER environ mock before running tests
import builtins
import logging
from typing import Any, Dict, List
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ
# Import environ for Node, since tests will be running for Node component
from fedbiomed.node.environ import environ

import sys, os

import unittest
from unittest.mock import MagicMock, patch

from fedbiomed.node.round import Round
from fedbiomed.common.logger import logger, DEFAULT_LOG_LEVEL

# importing fake (=dummy) classes
from testsupport.fake_classes.fake_training_plan import FakeModel
from testsupport.fake_classes.fake_message import FakeNodeMessages
from testsupport.fake_classes.fake_uuid import FakeUuid

class TestRound(unittest.TestCase):
    
    # values and attributes for dummy classes
    URL_MSG = 'http://url/where/my/file?is=True'
    
    @classmethod
    def setUpClass(cls):
        """Sets up values in the test once """
        # we define here common side effect functions
        def node_msg_side_effect(msg: Dict[str, Any]) -> Dict[str, Any]:
            fake_node_msg = FakeNodeMessages(msg)
            return fake_node_msg
        
        cls.node_msg_side_effect = node_msg_side_effect

    @patch('fedbiomed.common.repository.Repository.__init__')
    @patch('fedbiomed.node.model_manager.ModelManager.__init__')
    def setUp(self,
              model_manager_patch,
              reporistory_patch):
        model_manager_patch.return_value = None
        reporistory_patch.return_value = None

        # instantiate logger (we will see if exceptions are logged)
        logger.setLevel(DEFAULT_LOG_LEVEL)
        # instanciate Round class 
        self.r1 = Round(model_url='http://somewhere/where/my/model?is_stored=True',
                        model_class='MyTrainingPlan',
                        params_url='https://url/to/model/params?ok=True')

        self.r1.training_kwargs = {}
        params = {'path': 'my/dataset/path',
                           'dataset_id': 'id_1234'}
        self.r1.dataset = params
        self.r1.job_id = '1234'
        self.r1.researcher_id = '1234'
        dummy_monitor = MagicMock()
        self.r1.monitor = dummy_monitor

        self.r2 = Round(model_url='http://a/b/c/model',
                        model_class='another_training_plan',
                        params_url='https://to/my/model/params')
        self.r2.training_kwargs = {}
        self.r2.dataset = params
        self.r2.monitor = dummy_monitor
        
        sys.path.insert(0, environ['TMP_DIR'])

    def tearDown(self) -> None:
        sys.path.remove(environ['TMP_DIR'])

    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('builtins.eval')
    @patch('builtins.exec')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_run_model_training_01(self,
                                   uuid_patch,
                                   repository_download_patch,
                                   model_manager_patch,
                                   builtin_exec_patch,
                                   builtin_eval_patch,
                                   repository_upload_patch,
                                   node_msg_patch
                                   ):
        """tests correct execution and message parameters.
        Besides  tests the training time.
         """
        # Tests details:
        # - Test 1: normal case scenario where no model_kwargs has been passed during model instantiation
        # - Test 2: normal case scenario where model_kwargs has been passed when during model instantiation
       
        FakeModel.SLEEPING_TIME = 1

        # initalisation of side effect function
        
        def repository_side_effect(model_url: str, model_name: str):
            return 200, 'my_python_model'

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.side_effect = repository_side_effect
        model_manager_patch.return_value = (True, {'name': "model_name"})
        builtin_exec_patch.return_value = None
        builtin_eval_patch.return_value = FakeModel
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        
        # test 1: case where argument `model_kwargs` = None
        # action!
        msg_test1 = self.r1.run_model_training()
        
        # check results
        self.assertTrue(msg_test1.get('success', False))
        self.assertEqual(msg_test1.get('params_url', False), TestRound.URL_MSG)
        self.assertEqual(msg_test1.get('command', False), 'train')
        
        # timing test
        self.assertAlmostEqual(
            msg_test1.get('timing', {'rtime_training': 0}).get('rtime_training'),
            FakeModel.SLEEPING_TIME,
            places=1
                    )
        
        # test 2: redo test 1 but with the case where `model_kwargs` != None
        FakeModel.SLEEPING_TIME = 0
        self.r2.model_kwargs = {'param1': 1234,
                                'param2': [1, 2, 3, 4],
                                'param3': None}
        msg_test2 = self.r2.run_model_training()
        
        # check values in message (output of `run_model_training`)
        self.assertTrue(msg_test2.get('success', False))
        self.assertEqual(TestRound.URL_MSG, msg_test2.get('params_url', False))
        self.assertEqual('train', msg_test2.get('command', False))
        
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('builtins.eval')
    @patch('builtins.exec')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_run_model_training_02_correct_model_calls(self,
                                                       uuid_patch,
                                                       repository_download_patch,
                                                       model_manager_patch,
                                                       builtin_exec_patch,
                                                       builtin_eval_patch,
                                                       repository_upload_patch,
                                                       node_msg_patch):
        """tests if all methods of `model` have been called after instanciating"""
        # `run_model_training`, when no issues are found
        # methods tested:
        #  - model.load
        #  - model.save
        #  - model.training_routine
        #  - model.after_training_params
        #  - model.set_dataset

        FakeModel.SLEEPING_TIME = 0
        
        MODEL_NAME = "my_model"
        MODEL_PARAMS = [1, 2, 3, 4]

        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, MODEL_NAME)
        model_manager_patch.return_value = (True, {'name': "model_name"})
        builtin_exec_patch.return_value = None
        builtin_eval_patch.return_value = FakeModel
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect

        self.r1.training_kwargs = {}
        self.r1.dataset = {'path': 'my/dataset/path',
                           'dataset_id': 'id_1234'}

        # arguments of `save` method
        _model_filename = environ['TMP_DIR'] + '/node_params_1234.pt'
        _model_results = {
            'researcher_id': self.r1.researcher_id,
            'job_id': self.r1.job_id,
            'model_params': MODEL_PARAMS,
            'history': self.r1.monitor.history,
            'node_id': environ['NODE_ID']
        }
        
        # define context managers for each model method
        # we are mocking every methods of our dummy model FakeModel,
        # and we will check if there are called when running
        # `run_model_training` 
        with (
            patch.object(FakeModel, 'load') as mock_load,
            patch.object(FakeModel, 'set_dataset') as mock_set_dataset,
            patch.object(FakeModel, 'training_routine') as mock_training_routine,
            patch.object(FakeModel, 'after_training_params', return_value=MODEL_PARAMS) as mock_after_training_params,
            patch.object(FakeModel, 'save') as mock_save
              ):
            msg = self.r1.run_model_training()
            
        # test if all methods have been called once with the good arguments
        mock_load.assert_called_once_with(MODEL_NAME,
                                          to_params=False)
        mock_set_dataset.assert_called_once_with(self.r1.dataset.get('path'))

        mock_training_routine.assert_called_once_with( monitor=self.r1.monitor,
                                                       node_args=None)

        mock_after_training_params.assert_called_once()
        mock_save.assert_called_once_with(_model_filename, _model_results)

    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_run_model_training_03_test_with_real_model(self,
                                                        uuid_patch,
                                                        repository_download_patch,
                                                        model_manager_patch,
                                                        repository_upload_patch,
                                                        node_msg_patch):


        """tests normal case scenario with a real model file"""
        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        model_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect

        # create dummy_model
        dummy_training_plan_test = \
            "class MyTrainingPlan:\n" + \
            "   def __init__(self, **kwargs):\n" + \
            "       self._kwargs = kwargs\n" + \
            "   def load(self, *args, **kwargs):\n" + \
            "       pass \n" + \
            "   def save(self, *args, **kwargs):\n" + \
            "       pass\n" + \
            "   def training_routine(self, *args, **kwargs):\n" + \
            "       pass\n" + \
            "   def set_dataset(self, *args, **kwargs):\n" + \
            "       pass\n" + \
            "   def after_training_params(self):\n" + \
            "       return [1,2,3,4]\n"

        module_file_path = os.path.join(environ['TMP_DIR'],
                                        'my_model_' + str(FakeUuid.VALUE) + '.py')

        # creating file for toring dummy training plan
        with open(module_file_path, "w") as f:
            f.write(dummy_training_plan_test)

        # action
        msg_test = self.r1.run_model_training()

         ## checks
        self.assertTrue(msg_test.get('success', False))
        self.assertEqual(TestRound.URL_MSG, msg_test.get('params_url', False))
        self.assertEqual('train', msg_test.get('command', False))

        # remove model file
        os.remove(module_file_path)

    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')    
    def test_run_model_training_04_bad_http_status(self,
                                                   uuid_patch,
                                                   repository_download_patch,
                                                   model_manager_patch,
                                                   repository_upload_patch,
                                                   node_msg_patch): 
        """tests failures and exceptions during the download file process"""
        # Tests details:
        # Test 1: tests case where downloading model file fails
        # Test 2: tests case where downloading model paraeters fails
        FakeModel.SLEEPING_TIME = 0
        
        # initalisation of side effects functions

        def download_repo_answers_gene() -> int:
            """Generates different values of connections:
            First one is HTTP code 200, second one is HTTP code 404
            Raises: StopIteration, if called more than twice
            """
            for i in [200, 404]:
                yield i

        def repository_side_effect_test_1(model_url: str, model_name: str):
            """Returns HTTP 404 error, mimicking an error happened during 
            download process"""
            return 404, 'my_python_model'

        download_repo_answers_iter = iter(download_repo_answers_gene())
        # initialisation of patchers 

        uuid_patch.return_value = FakeUuid()
        repository_download_patch.side_effect = repository_side_effect_test_1
        model_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect

        # test 1: case where first call to `Repository.download` generates HTTP
        # status 404 (when downloading model_file)
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            msg_test_1 = self.r1.run_model_training()

        # checks:
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training`
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_1.get('msg'))

        self.assertFalse(msg_test_1.get('success', True))

        # test 2: case where second call to `Repository.download` generates HTTP
        # status 404 (when downloading params_file)
        # overwriting side effect function for second test:
        def repository_side_effect_2(model_url: str, model_name: str):
            """Returns different values when called
            First call: returns (200, 'my_python_model') mimicking a first download 
                that happened without encoutering any issues
            Second call: returns (404, 'my_python_model') mimicking a second download
                that failed
            Third Call (or more): raises StopIteration (due to generator)
                
            """
            val = next(download_repo_answers_iter)
            return val, 'my_python_model'

        repository_download_patch.side_effect = repository_side_effect_2

        # action
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            msg_test_2 = self.r1.run_model_training()

        # checks:
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training`
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_2.get('msg'))
        self.assertFalse(msg_test_2.get('success', True))

        # test 3: check if unknown exception is raised and caught during the download
        # files process

        def repository_side_effect_3(model_url: str, model_name: str):
            raise Exception('mimicking an error during download files process')
        
        repository_download_patch.side_effect = repository_side_effect_3
        model_manager_patch.return_value = (True, {'name': "model_name"})
        
        # action
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            msg_test_3 = self.r1.run_model_training()
        ## checks
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_3.get('msg'))
        self.assertFalse(msg_test_3.get('success', True))

    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_run_model_training_05_import_error(self,
                                                uuid_patch,
                                                repository_download_patch,
                                                model_manager_patch,
                                                repository_upload_patch,
                                                node_msg_patch):
        """tests case where the import/loading of the model have failed"""

        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        model_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect

        # test 1: tests raise of exception during model import
        def exec_side_effect(*args, **kwargs):
            """Overriding the behaviour of `exec` builitin function,
            and raises an Exception"""
            raise Exception("mimicking an exception happening when loading file")

        # patching builtin objects & looking for generated logs
        # NB: this is the only way I have found to use
        # both patched bulitins functions and assertLogs
        with (self.assertLogs('fedbiomed', logging.ERROR) as captured,
              patch.object(builtins, 'exec', return_value = None),
              patch.object(builtins, 'eval') as eval_patch):
            eval_patch.side_effect = exec_side_effect
            msg_test_1 = self.r1.run_model_training()

        # checks:
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training`
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_1.get('msg'))

        self.assertFalse(msg_test_1.get('success', True))

        # test 2: tests raise of Exception during loading parameters
        # into model instance

        # Here we creating a new class inheriting from the FakeModel,
        # but overriding `load` through classes inheritance
        # when `load` is called, an Exception will be raised
        # 
        class FakeModelRaiseExceptionWhenLoading(FakeModel):
            def load(self, **kwargs):
                """Mimicks an exception happening in the `load`
                method

                Raises:
                    Exception: 
                """
                raise Exception('mimicking an error happening during model training')

        # action
        with (self.assertLogs('fedbiomed', logging.ERROR) as captured,
              patch.object(builtins, 'exec', return_value=None),
              patch.object(builtins, 'eval', return_value=FakeModelRaiseExceptionWhenLoading) 
              ):

            msg_test_2 = self.r1.run_model_training()

        # checks:
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training`
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_2.get('msg'))
        self.assertFalse(msg_test_2.get('success', True))

        # test 3: tests raise of Exception during model training
        # into model instance
        
        # Here we are creating a new class inheriting from the FakeModel,
        # but overriding `training_routine` through classes inheritance
        # when `training_routine` is called, an Exception will be raised
        # 
        class FakeModelRaiseExceptionInTraining(FakeModel):
            def training_routine(self, **kwargs):
                """Mimicks an exception happening in the `training_routine`
                method

                Raises:
                    Exception: 
                """
                raise Exception('mimicking an error happening during model training')
        
        # action
        with (self.assertLogs('fedbiomed', logging.ERROR) as captured,
              patch.object(builtins, 'exec', return_value=None),
              patch.object(builtins, 'eval', return_value= FakeModelRaiseExceptionInTraining)):
            msg_test_3 = self.r1.run_model_training()

        # checks :
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training``
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_3.get('msg'))

        self.assertFalse(msg_test_3.get('success', True))

    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_run_model_training_07_upload_file_fails(self,
                                                     uuid_patch,
                                                     repository_download_patch,
                                                     model_manager_patch,
                                                     repository_upload_patch,
                                                     node_msg_patch):

        """tests case where uploading model parameters file fails"""
        FakeModel.SLEEPING_TIME = 0
        
        # declaration of side effect functions

        def upload_side_effect(*args, **kwargs):
            """Raises an exception when calling this function"""
            raise Exception("mimicking an error happening during upload")
        
        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        model_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.side_effect = upload_side_effect
        node_msg_patch.side_effect = TestRound.node_msg_side_effect

        # action
        with (self.assertLogs('fedbiomed', logging.ERROR) as captured,
              patch.object(builtins, 'exec', return_value=None),
              patch.object(builtins, 'eval', return_value=FakeModel) 
                ):
            msg_test = self.r1.run_model_training()

         ## checks
         # checks if message logged is the message returned as a reply
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test.get('msg'))

        self.assertFalse(msg_test.get('success', True))

        
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('builtins.eval')
    @patch('builtins.exec')
    @patch('fedbiomed.node.model_manager.ModelManager.check_is_model_approved')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4') 
    def test_run_model_training_08_bad_training_argument(self,
                                                         uuid_patch,
                                                         repository_download_patch,
                                                         model_manager_patch,
                                                         builtin_exec_patch,
                                                         builtin_eval_patch,
                                                         repository_upload_patch,
                                                         node_msg_patch):
        """tests case where training plan contains node_side arguments"""
        FakeModel.SLEEPING_TIME = 1

        # initialisation of patchers 
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, "my_model")
        model_manager_patch.return_value = (True, {'name': "model_name"})
        builtin_exec_patch.return_value = None
        builtin_eval_patch.return_value = FakeModel
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        
        # adding into `training_kwargs` node_side arguments
        self.r1.training_kwargs = {'param': 1234,
                                   'monitor': MagicMock(),
                                   'node_args': [1, 2, 3, 4]}
        # action!
        msg = self.r1.run_model_training()
        
        # check if 'monitor' and 'node_args' entries have been removed 
        #  in `training_kwargs` (for security reasons, see Round for further details)
        
        self.assertFalse(self.r1.training_kwargs.get('monitor', False))
        self.assertFalse(self.r1.training_kwargs.get('node_args', False))
        

if __name__ == '__main__': # pragma: no cover
    unittest.main()
