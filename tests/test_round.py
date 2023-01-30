import builtins
import copy
import inspect
import logging
import os
from typing import Any, Dict
import unittest
from unittest.mock import MagicMock, patch

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from testsupport.fake_training_plan import FakeModel
from testsupport.fake_message import FakeMessages
from testsupport.fake_uuid import FakeUuid


from fedbiomed.node.environ import environ
from fedbiomed.node.round import Round
from fedbiomed.common.logger import logger
from fedbiomed.common.data import DataManager, DataLoadingPlanMixin, DataLoadingPlan
from fedbiomed.common.constants import DatasetTypes
from testsupport.testing_data_loading_block import ModifyGetItemDP, LoadingBlockTypesForTesting


# Needed to access length of dataset from Round class
class FakeLoader:
    dataset = [1, 2, 3, 4, 5]


class TestRound(NodeTestCase):

    # values and attributes for dummy classes
    URL_MSG = 'http://url/where/my/file?is=True'

    @classmethod
    def setUpClass(cls):
        """Sets up values in the test once """

        # Sets mock environ for the test -------------------
        super().setUpClass()
        # --------------------------------------------------

        # we define here common side effect functions
        def node_msg_side_effect(msg: Dict[str, Any]) -> Dict[str, Any]:
            fake_node_msg = FakeMessages(msg)
            return fake_node_msg

        cls.node_msg_side_effect = node_msg_side_effect

    @patch('fedbiomed.common.repository.Repository.__init__')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.__init__')
    def setUp(self,
              tp_security_manager_patch,
              repository_patch):
        tp_security_manager_patch.return_value = None
        repository_patch.return_value = None

        # instantiate logger (we will see if exceptions are logged)
        # we are setting the logger level to "ERROR" to output
        # logs messages
        logger.setLevel("ERROR")
        # instanciate Round class
        self.r1 = Round(training_plan_url='http://somewhere/where/my/model?is_stored=True',
                        training_plan_class='MyTrainingPlan',
                        params_url='https://url/to/model/params?ok=True',
                        training_kwargs={},
                        training=True
                        )
        params = {'path': 'my/dataset/path',
                  'dataset_id': 'id_1234'}
        self.r1.dataset = params
        self.r1.job_id = '1234'
        self.r1.researcher_id = '1234'
        dummy_monitor = MagicMock()
        self.r1.history_monitor = dummy_monitor

        self.r2 = Round(training_plan_url='http://a/b/c/model',
                        training_plan_class='another_training_plan',
                        params_url='https://to/my/model/params',
                        training_kwargs={},
                        training=True)
        self.r2.dataset = params
        self.r2.history_monitor = dummy_monitor


    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('importlib.import_module')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_01_run_model_training_normal_case(self,
                                                     uuid_patch,
                                                     repository_download_patch,
                                                     tp_security_manager_patch,
                                                     import_module_patch,
                                                     repository_upload_patch,
                                                     node_msg_patch,
                                                     mock_split_test_train_data,
                                                     ):
        """tests correct execution and message parameters.
        Besides  tests the training time.
         """
        # Tests details:
        # - Test 1: normal case scenario where no model_kwargs has been passed during model instantiation
        # - Test 2: normal case scenario where model_kwargs has been passed when during model instantiation

        FakeModel.SLEEPING_TIME = 1

        # initalisation of side effect function

        def repository_side_effect(training_plan_url: str, model_name: str):
            return 200, 'my_python_model'

        class FakeModule:
            MyTrainingPlan = FakeModel
            another_training_plan = FakeModel

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.side_effect = repository_side_effect
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        import_module_patch.return_value = FakeModule
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_test_train_data.return_value = (FakeLoader, FakeLoader)

        # test 1: case where argument `model_kwargs` = None
        # action!
        msg_test1 = self.r1.run_model_training()

        # check results
        self.assertTrue(msg_test1.get('success', False))
        self.assertEqual(msg_test1.get('params_url', False), TestRound.URL_MSG)
        self.assertEqual(msg_test1.get('command', False), 'train')

        # timing test - does not always work with self.assertAlmostEqual
        self.assertGreaterEqual(
            msg_test1.get('timing', {'rtime_training': 0}).get('rtime_training'),
            FakeModel.SLEEPING_TIME
        )
        self.assertLess(
            msg_test1.get('timing', {'rtime_training': 0}).get('rtime_training'),
            FakeModel.SLEEPING_TIME * 1.1
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

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('importlib.import_module')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_02_run_model_training_correct_model_calls(self,
                                                             uuid_patch,
                                                             repository_download_patch,
                                                             tp_security_manager_patch,
                                                             import_module_patch,
                                                             repository_upload_patch,
                                                             node_msg_patch,
                                                             mock_split_train_and_test_data):
        """tests if all methods of `model` have been called after instanciating
        (in run_model_training)"""
        # `run_model_training`, when no issues are found
        # methods tested:
        #  - model.load
        #  - model.save
        #  - model.training_routine
        #  - model.after_training_params
        #  - model.set_dataset_path

        FakeModel.SLEEPING_TIME = 0
        MODEL_NAME = "my_model"
        MODEL_PARAMS = [1, 2, 3, 4]

        class FakeModule:
            MyTrainingPlan = FakeModel

        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, MODEL_NAME)
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        import_module_patch.return_value = FakeModule
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = (FakeLoader, FakeLoader)

        self.r1.training_kwargs = {}
        self.r1.dataset = {'path': 'my/dataset/path',
                           'dataset_id': 'id_1234'}

        # arguments of `save` method
        _model_filename = environ['TMP_DIR'] + '/node_params_1234.pt'
        _model_results = {
            'researcher_id': self.r1.researcher_id,
            'job_id': self.r1.job_id,
            'model_params': MODEL_PARAMS,
            'node_id': environ['NODE_ID'],
            'optimizer_args': {}
        }

        # define context managers for each model method
        # we are mocking every methods of our dummy model FakeModel,
        # and we will check if there are called when running
        # `run_model_training`
        with (
                patch.object(FakeModel, 'load') as mock_load,
                patch.object(FakeModel, 'set_dataset_path') as mock_set_dataset,
                patch.object(FakeModel, 'training_routine') as mock_training_routine,
                patch.object(FakeModel, 'after_training_params', return_value=MODEL_PARAMS) as mock_after_training_params,  # noqa
                patch.object(FakeModel, 'save') as mock_save
        ):
            _ = self.r1.run_model_training()

            # test if all methods have been called once with the good arguments
            mock_load.assert_called_once_with(MODEL_NAME,
                                              to_params=False)


            # Check set train and test data split function is called
            # Set dataset is called in set_train_and_test_data
            # mock_set_dataset.assert_called_once_with(self.r1.dataset.get('path'))
            mock_split_train_and_test_data.assert_called_once()

            # Since set training data return None, training_routine should be called as None
            mock_training_routine.assert_called_once_with( history_monitor=self.r1.history_monitor,
                                                           node_args=None)

            mock_after_training_params.assert_called_once()
            mock_save.assert_called_once_with(_model_filename, _model_results)

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_03_test_run_model_training_with_real_model(self,
                                                              uuid_patch,
                                                              repository_download_patch,
                                                              tp_security_manager_patch,
                                                              repository_upload_patch,
                                                              node_msg_patch,
                                                              mock_split_train_and_test_data):
        """tests normal case scenario with a real model file"""
        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = (True, True)

        # create dummy_model
        dummy_training_plan_test = \
            "class MyTrainingPlan:\n" + \
            "   dataset = [1,2,3,4]\n" + \
            "   def __init__(self, **kwargs):\n" + \
            "       self._kwargs = kwargs\n" + \
            "       self._kwargs = kwargs\n" + \
            "       self._kwargs = kwargs\n" + \
            "   def post_init(self, model_args, training_args, optimizer_args=None, aggregator_args=None):\n" + \
            "       pass\n" + \
            "   def load(self, *args, **kwargs):\n" + \
            "       pass \n" + \
            "   def save(self, *args, **kwargs):\n" + \
            "       pass\n" + \
            "   def training_routine(self, *args, **kwargs):\n" + \
            "       pass\n" + \
            "   def set_data_loaders(self, *args, **kwargs):\n" + \
            "       self.testing_data_loader = MyTrainingPlan\n" + \
            "       self.training_data_loader = MyTrainingPlan\n" + \
            "       pass\n" + \
            "   def set_dataset_path(self, *args, **kwargs):\n" + \
            "       pass\n" + \
            "   def optimizer_args(self):\n" + \
            "       pass\n" + \
            "   def after_training_params(self):\n" + \
            "       return [1,2,3,4]\n"


        module_file_path = os.path.join(environ['TMP_DIR'],
                                        'training_plan_' + str(FakeUuid.VALUE) + '.py')

        # creating file for toring dummy training plan
        with open(module_file_path, "w") as f:
            f.write(dummy_training_plan_test)

        # action
        msg_test = self.r1.run_model_training()
        print("MESSAGE", msg_test)
        # checks
        self.assertTrue(msg_test.get('success', False))
        self.assertEqual(TestRound.URL_MSG, msg_test.get('params_url', False))
        self.assertEqual('train', msg_test.get('command', False))

        # remove model file
        os.remove(module_file_path)

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_rounds_04_run_model_training_bad_http_status(self,
                                                          uuid_patch,
                                                          repository_download_patch,
                                                          tp_security_manager_patch,
                                                          repository_upload_patch,
                                                          node_msg_patch,
                                                          mock_split_train_and_test_data):
        """tests failures and exceptions during the download file process
        (in run_model_training)"""
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

        def repository_side_effect_test_1(training_plan_url: str, model_name: str):
            """Returns HTTP 404 error, mimicking an error happened during
            download process"""
            return 404, 'my_python_model'

        download_repo_answers_iter = iter(download_repo_answers_gene())
        # initialisation of patchers

        uuid_patch.return_value = FakeUuid()
        repository_download_patch.side_effect = repository_side_effect_test_1
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = None

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
        def repository_side_effect_2(training_plan_url: str, model_name: str):
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

        def repository_side_effect_3(training_plan_url: str, model_name: str):
            raise Exception('mimicking an error during download files process')

        repository_download_patch.side_effect = repository_side_effect_3
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})

        # action
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            msg_test_3 = self.r1.run_model_training()

        # checks
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_3.get('msg'))
        self.assertFalse(msg_test_3.get('success', True))

    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_05_run_model_training_model_not_approved(self,
                                                            uuid_patch,
                                                            repository_download_patch,
                                                            tp_security_manager_patch):
        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        tp_security_manager_patch.return_value = (False, {'name': "model_name"})
        # action
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            msg_test = self.r1.run_model_training()
            # checks
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test.get('msg'))

        self.assertFalse(msg_test.get('success', True))

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_06_run_model_training_import_error(self,
                                                      uuid_patch,
                                                      repository_download_patch,
                                                      tp_security_manager_patch,
                                                      repository_upload_patch,
                                                      node_msg_patch,
                                                      mock_split_train_and_test_data):
        """tests case where the import/loading of the model have failed"""

        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = None

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

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_07_run_model_training_upload_file_fails(self,
                                                           uuid_patch,
                                                           repository_download_patch,
                                                           tp_security_manager_patch,
                                                           repository_upload_patch,
                                                           node_msg_patch,
                                                           mock_split_train_and_test_data):

        """tests case where uploading model parameters file fails"""
        FakeModel.SLEEPING_TIME = 0

        # declaration of side effect functions

        def upload_side_effect(*args, **kwargs):
            """Raises an exception when calling this function"""
            raise Exception("mimicking an error happening during upload")

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.side_effect = upload_side_effect
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = None

        # action
        with (self.assertLogs('fedbiomed', logging.ERROR) as captured,
              patch.object(builtins, 'exec', return_value=None),
              patch.object(builtins, 'eval', return_value=FakeModel)
              ):
            msg_test = self.r1.run_model_training()

        # checks if message logged is the message returned as a reply
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test.get('msg'))

        self.assertFalse(msg_test.get('success', True))

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.reply_create')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('builtins.eval')
    @patch('builtins.exec')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_08_run_model_training_bad_training_argument(self,
                                                               uuid_patch,
                                                               repository_download_patch,
                                                               tp_security_manager_patch,
                                                               builtin_exec_patch,
                                                               builtin_eval_patch,
                                                               repository_upload_patch,
                                                               node_msg_patch,
                                                               mock_split_train_and_test_data):
        """tests case where training plan contains node_side arguments"""
        FakeModel.SLEEPING_TIME = 1

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, "my_model")
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        builtin_exec_patch.return_value = None
        builtin_eval_patch.return_value = FakeModel
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = None

    @patch('inspect.signature')
    def test_round_09_data_loading_plan(self,
                                        patch_inspect_signature,
                                        ):
        """Test that Round correctly handles a DataLoadingPlan during training"""
        class MyDataset(DataLoadingPlanMixin):
            def __init__(self):
                super().__init__()

            def __getitem__(self, item):
                return self.apply_dlb('orig-value', LoadingBlockTypesForTesting.MODIFY_GETITEM)

            @staticmethod
            def get_dataset_type() -> DatasetTypes:
                return DatasetTypes.TEST

        patch_inspect_signature.return_value = inspect.Signature(parameters={})

        my_dataset = MyDataset()
        data_loader_mock = MagicMock()
        data_loader_mock.dataset = my_dataset

        data_manager_mock = MagicMock(spec=DataManager)
        data_manager_mock.split = MagicMock()
        data_manager_mock.split.return_value = (data_loader_mock, None)
        data_manager_mock.dataset = my_dataset

        r3 = Round(training_kwargs={})
        r3.training_plan = MagicMock()
        r3.training_plan.training_data.return_value = data_manager_mock

        training_data_loader, _ = r3._split_train_and_test_data(test_ratio=0.)
        dataset = training_data_loader.dataset
        self.assertEqual(dataset[0], 'orig-value')

        dlp = DataLoadingPlan({LoadingBlockTypesForTesting.MODIFY_GETITEM: ModifyGetItemDP()})
        r4 = Round(training_kwargs={},
                   dlp_and_loading_block_metadata=dlp.serialize()
                   )
        r4.training_plan = MagicMock()
        r4.training_plan.training_data.return_value = data_manager_mock

        training_data_loader, _ = r4._split_train_and_test_data(test_ratio=0.)
        dataset = training_data_loader.dataset
        self.assertEqual(dataset[0], 'modified-value')


    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_10_download_aggregator_args(self, uuid_patch, repository_download_patch, ):
        uuid_patch.return_value = FakeUuid()
        
        repository_download_patch.side_effect = ((200, "my_model_var"+ str(i)) for i in range(3, 5))
        success, _ = self.r1.download_aggregator_args()
        self.assertEqual(success, True)
        # if attribute `aggregator_args` is None, then do nothing
        repository_download_patch.assert_not_called()

        aggregator_args = {'var1': 1, 
                            'var2': [1, 2, 3, 4],
                            'var3': {'url': 'http://to/var/3',},
                            'var4': {'url': 'http://to/var/4'}}
        self.r1.aggregator_args = copy.deepcopy(aggregator_args)
        
        success, error_msg = self.r1.download_aggregator_args()
        self.assertEqual(success, True)
        self.assertEqual(error_msg, '')
        
        for var in ('var1', 'var2'):
            self.assertEqual(self.r1.aggregator_args[var], aggregator_args[var])
        
        for var in ('var3', 'var4'):
            self.assertNotIn('url', self.r1.aggregator_args[var].keys())
            self.assertEqual(self.r1.aggregator_args[var]['param_path'], 'my_model_' + var)

    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_11_download_file(self, uuid_patch, repository_download_patch):
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, "my_model")
        file_path = 'path/to/my/downloaded/files'
        success, param_path, msg = self.r1.download_file('http://some/url/to/some/files', file_path)
        self.assertEqual(success, True)
        self.assertEqual(param_path, 'my_model')
        self.assertEqual(msg, '')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
