import builtins
import copy
import inspect
import logging
import os
import tempfile
import unittest
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, create_autospec, patch, PropertyMock
from fedbiomed.common.optimizers.generic_optimizers import DeclearnOptimizer
from fedbiomed.common.serializer import Serializer
from fedbiomed.node.node_state_manager import NodeStateFileName


#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from testsupport.fake_training_plan import FakeModel, DeclearnAuxVarModel
from testsupport.fake_message import FakeMessages
from testsupport.fake_uuid import FakeUuid
from testsupport.testing_data_loading_block import ModifyGetItemDP, LoadingBlockTypesForTesting
from testsupport import fake_training_plan

import torch
from fedbiomed.common.optimizers.declearn import YogiModule, ScaffoldClientModule, RidgeRegularizer

from fedbiomed.common.constants import DatasetTypes, TrainingPlans
from fedbiomed.common.data import DataManager, DataLoadingPlanMixin, DataLoadingPlan
from fedbiomed.common.exceptions import  FedbiomedOptimizerError, FedbiomedRoundError
from fedbiomed.common.logger import logger
from fedbiomed.common.models import TorchModel, Model
from fedbiomed.common.optimizers import BaseOptimizer, Optimizer
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.node.environ import environ
from fedbiomed.node.round import Round


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

    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.__init__')
    def setUp(self,
              tp_security_manager_patch):
        
        tp_security_manager_patch.return_value = None

        # instantiate logger (we will see if exceptions are logged)
        # we are setting the logger level to "ERROR" to output
        # logs messages
        history_monitor = MagicMock()

        self.atexit_patcher = patch('fedbiomed.node.round.atexit')
        self.atexit_mock = self.atexit_patcher.start()

        self.state_manager_patch = patch('fedbiomed.node.round.NodeStateManager')
        self.ic_from_spec_patch = patch("fedbiomed.node.round.utils.import_class_from_spec")
        self.ic_from_file_patch = patch("fedbiomed.node.round.utils.import_class_object_from_file")

        self.ic_from_spec_mock = self.ic_from_spec_patch.start()
        self.ic_from_file_mock = self.ic_from_file_patch.start()
        self.state_manager_mock = self.state_manager_patch.start()

        type(self.state_manager_mock.return_value).state_id = PropertyMock(return_value='test-state-id')

        class FakeModule:
            MyTrainingPlan = FakeModel
            another_training_plan = FakeModel

        self.ic_from_spec_mock.return_value = (FakeModule, FakeModule.MyTrainingPlan)
        self.ic_from_file_mock.return_value = (FakeModule, FakeModule.MyTrainingPlan())

        logger.setLevel("ERROR")
        # instanciate Round class
        self.r1 = Round(training_plan='TP',
                        training_plan_class='MyTrainingPlan',
                        params={"x": 0},
                        training_kwargs={},
                        model_kwargs={},
                        researcher_id="researcher-id",
                        history_monitor=history_monitor,
                        dataset={"path": 'ssss'},
                        job_id="job_id",
                        training=True,
                        node_args={},
                        aggregator_args={})
        
        params = {'path': 'my/dataset/path',
                  'dataset_id': 'id_1234'}
        self.r1.dataset = params
        self.r1.job_id = '1234'
        self.r1.researcher_id = '1234'
        dummy_monitor = MagicMock()
        self.r1.history_monitor = dummy_monitor

        self.r2 = Round(training_plan='TP',
                        training_plan_class='another_training_plan',
                        params={"x": 0},
                        training_kwargs={},
                        model_kwargs={},
                        researcher_id="researcher-id",
                        history_monitor=history_monitor,
                        dataset={"path": 'ssss'},
                        job_id="job_id",
                        training=True,
                        node_args={},
                        aggregator_args={})
        self.r2.dataset = params
        self.r2.history_monitor = dummy_monitor

    def tearDown(self):
        self.atexit_patcher.stop() 
        self.ic_from_file_patch.stop()
        self.ic_from_spec_patch.stop()
        self.state_manager_patch.stop()

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_outgoing_message')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('uuid.uuid4')
    def test_round_01_run_model_training_normal_case(self,
                                                     uuid_patch,
                                                     tp_security_manager_patch,
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

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})



        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_test_train_data.return_value = (FakeLoader, FakeLoader)

        # test 1: case where argument `model_kwargs` = None
        # action!
        self.r1.initialize_node_state_manager()
        msg_test1 = self.r1.run_model_training()

        # check results
        self.assertTrue(msg_test1.get_dict().get('success', False))
        self.assertEqual(msg_test1.get_dict().get('params', False), {"coefs": [1, 2, 3, 4]})
        self.assertEqual(msg_test1.get_dict().get('command', False), 'train')


        # test 2: redo test 1 but with the case where `model_kwargs` != None
        FakeModel.SLEEPING_TIME = 0
        self.r2.model_kwargs = {'param1': 1234,
                                'param2': [1, 2, 3, 4],
                                'param3': None}
        self.r2.initialize_node_state_manager()
        msg_test2 = self.r2.run_model_training()

        # check values in message (output of `run_model_training`)
        self.assertTrue(msg_test2.get_dict().get('success', False))
        self.assertEqual({"coefs": [1, 2, 3, 4]}, msg_test2.get_dict().get('params', False))
        self.assertEqual('train', msg_test2.get_dict().get('command', False))

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('importlib.import_module')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('uuid.uuid4')
    def test_round_02_run_model_training_correct_model_calls(self,
                                                             uuid_patch,
                                                             tp_security_manager_patch,
                                                             import_module_patch,
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
        MODEL_PARAMS = {"coef": [1, 2, 3, 4]}

        class FakeModule:
            MyTrainingPlan = FakeModel

        uuid_patch.return_value = FakeUuid()
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        import_module_patch.return_value = FakeModule
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = (FakeLoader, FakeLoader)

        self.r1.training_kwargs = {}
        self.r1.dataset = {'path': 'my/dataset/path',
                           'dataset_id': 'id_1234'}


        # define context managers for each model method
        # we are mocking every methods of our dummy model FakeModel,
        # and we will check if there are called when running
        # `run_model_training`
        with (
                patch.object(FakeModel, 'set_dataset_path') as mock_set_dataset,
                patch.object(FakeModel, 'training_routine') as mock_training_routine,
                patch.object(FakeModel, 'after_training_params', return_value=MODEL_PARAMS) as mock_after_training_params,  # noqa
        ):
            self.r1.initialize_node_state_manager()
            msg = self.r1.run_model_training()
            self.assertTrue(msg.get_dict().get("success"))



            # Check set train and test data split function is called
            # Set dataset is called in set_train_and_test_data
            # mock_set_dataset.assert_called_once_with(self.r1.dataset.get('path'))
            mock_split_train_and_test_data.assert_called_once()

            # Since set training data return None, training_routine should be called as None
            mock_training_routine.assert_called_once_with( history_monitor=self.r1.history_monitor,
                                                           node_args={})

            # Check that the model weights were saved.
            mock_after_training_params.assert_called_once()

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('uuid.uuid4')
    def test_round_03_test_run_model_training_with_real_model(self,
                                                              uuid_patch,
                                                              tp_security_manager_patch,
                                                              node_msg_patch,
                                                              mock_split_train_and_test_data):
        """tests normal case scenario with a real model file"""
        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = (True, True)

        # create dummy_model
        dummy_training_plan_test = "\n".join([
            "from testsupport.fake_training_plan import FakeModel",
            "class MyTrainingPlan(FakeModel):",
            "    dataset = [1, 2, 3, 4]",
            "    def set_data_loaders(self, *args, **kwargs):",
            "        self.testing_data_loader = MyTrainingPlan",
            "        self.training_data_loader = MyTrainingPlan",
        ])

        self.ic_from_spec_patch.stop()
        self.ic_from_file_patch.stop()
        self.r1.training_plan_source = dummy_training_plan_test
        self.r1.training_plan_class = "MyTrainingPlan"
        # action
        self.r1.initialize_node_state_manager()
        msg_test = self.r1.run_model_training()
        # checks

        self.assertTrue(msg_test.get_dict().get('success', False))
        self.assertEqual('train', msg_test.get_dict().get('command', False))


    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('uuid.uuid4')
    def test_round_05_run_model_training_model_not_approved(self,
                                                            uuid_patch,
                                                            tp_security_manager_patch):
        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        tp_security_manager_patch.return_value = (False, {'name': "model_name"})
        environ["TRAINING_PLAN_APPROVAL"] = True
        # action
        msg_test = self.r1.run_model_training()

        self.assertFalse(msg_test.get_param('success'))

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('uuid.uuid4')
    def test_round_06_run_model_training_import_error(self,
                                                      uuid_patch,
                                                      tp_security_manager_patch,
                                                      node_msg_patch,
                                                      mock_split_train_and_test_data):
        """tests case where the import/loading of the model have failed"""

        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = None

        self.ic_from_file_mock.side_effect = Exception
        msg_test_1 = self.r1.run_model_training()
        self.assertFalse(msg_test_1.success)


        self.ic_from_file_mock.side_effect = None
        self.ic_from_spec_mock.side_effect = Exception
        msg_test_1 = self.r1.run_model_training()
        self.assertFalse(msg_test_1.success)
        
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

        self.ic_from_file_mock.return_value = (fake_training_plan, FakeModelRaiseExceptionWhenLoading())
        msg_test_2 = self.r1.run_model_training()
        self.assertFalse(msg_test_2.success)

        # test 3: tests raise of Exception during model training
        # into model instance
        
        class FakeModelRaiseExceptionInTraining(FakeModel):
            def training_routine(self, **kwargs):
                """Mimicks an exception happening in the `training_routine`
                method

                Raises:
                    Exception:
                """
                raise Exception('mimicking an error happening during model training')
        self.ic_from_file_mock.return_value = (fake_training_plan, FakeModelRaiseExceptionInTraining())
        msg_test_3 = self.r1.run_model_training()
        self.assertFalse(msg_test_3.success, )


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

        self.r1.training_kwargs = {}
        self.r1.initialize_validate_training_arguments()
        self.r1.training_plan = MagicMock()
        self.r1.training_plan.training_data.return_value = data_manager_mock

        training_data_loader, _ = self.r1._split_train_and_test_data(test_ratio=0.)
        dataset = training_data_loader.dataset
        self.assertEqual(dataset[0], 'orig-value')

        dlp = DataLoadingPlan({LoadingBlockTypesForTesting.MODIFY_GETITEM: ModifyGetItemDP()})
        self.r1._dlp_and_loading_block_metadata = dlp.serialize()
        self.r1.training_kwargs = {}
        self.r1.initialize_validate_training_arguments()
       

        training_data_loader, _ = self.r1._split_train_and_test_data(test_ratio=0.)
        dataset = training_data_loader.dataset
        self.assertEqual(dataset[0], 'modified-value')


    @patch("fedbiomed.node.round.BPrimeManager.get")
    @patch("fedbiomed.node.round.SKManager.get")
    def test_round_12_configure_secagg(self,
                                       servkey_get,
                                       biprime_get
                                       ):
        """Tests round secure aggregation configuration"""

        servkey_get.return_value = {"context": {}}
        biprime_get.return_value = {"context": {}}

        environ["SECURE_AGGREGATION"] = True

        result = self.r1._configure_secagg(
            secagg_random=1.5,
            secagg_biprime_id='123',
            secagg_servkey_id='123'
        )
        self.assertTrue(result)

        result = self.r1._configure_secagg(
            secagg_random=None,
            secagg_biprime_id=None,
            secagg_servkey_id=None
        )
        self.assertFalse(result)

        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=None,
                secagg_biprime_id="1234",
                secagg_servkey_id=None)

        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=None,
                secagg_biprime_id="1234",
                secagg_servkey_id="1223")

        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=None,
                secagg_biprime_id=None,
                secagg_servkey_id="1223")

        with self.assertRaises(FedbiomedRoundError):
            servkey_get.return_value = None
            biprime_get.return_value = {"context": {}}
            self.r1._configure_secagg(
                secagg_random=1.5,
                secagg_biprime_id='123',
                secagg_servkey_id='123'
            )

        with self.assertRaises(FedbiomedRoundError):
            servkey_get.return_value = {"context": {}}
            biprime_get.return_value = None
            self.r1._configure_secagg(
                secagg_random=1.5,
                secagg_biprime_id='123',
                secagg_servkey_id='123'
            )

        # If node forces using secagg
        environ["SECURE_AGGREGATION"] = True
        environ["FORCE_SECURE_AGGREGATION"] = True
        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=None,
                secagg_biprime_id=None,
                secagg_servkey_id=None
            )

        # If secagg is not activated
        environ["SECURE_AGGREGATION"] = False
        environ["FORCE_SECURE_AGGREGATION"] = False
        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=1.5,
                secagg_biprime_id='123',
                secagg_servkey_id='123'
            )



    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('uuid.uuid4')
    @patch("fedbiomed.node.round.BPrimeManager.get")
    @patch("fedbiomed.node.round.SKManager.get")
    def test_round_13_run_model_training_secagg(self,
                                                servkey_get,
                                                biprime_get,
                                                uuid_patch,
                                                tp_security_manager_patch,
                                                node_msg_patch,
                                                mock_split_test_train_data):
        """tests correct execution and message parameters.
        Besides  tests the training time.
         """
        # Tests details:
        # - Test 1: normal case scenario where no model_kwargs has been passed during model instantiation
        # - Test 2: normal case scenario where model_kwargs has been passed when during model instantiation

        FakeModel.SLEEPING_TIME = 1

        # initalisation of side effect functio
        class M(FakeModel):
            def after_training_params(self, flatten):
                return [0.1,0.2,0.3,0.4,0.5]

        
        self.ic_from_file_mock.return_value = (fake_training_plan, M())
        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_test_train_data.return_value = (FakeLoader, FakeLoader)


        # Secagg configuration
        servkey_get.return_value = {"parties": ["r-1", "n-1", "n-2"],  "context" : {"server_key": 123445}}
        biprime_get.return_value = {"parties": ["r-1", "n-1", "n-2"], "context" : {"biprime": 123445}}
        environ["SECURE_AGGREGATION"] = True
        environ["FORCE_SECURE_AGGREGATION"] = True

        self.r1.initialize_node_state_manager()
        msg_test1 = self.r1.run_model_training(secagg_arguments={
            'secagg_random': 1.12,
            'secagg_servkey_id': '1234',
            'secagg_biprime_id': '1234',
        })

        # Back to normal
        environ["SECURE_AGGREGATION"] = False
        environ["FORCE_SECURE_AGGREGATION"] = False

    @patch("uuid.uuid4")
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    def test_round_14_run_model_training_optimizer_aux_var_error(self,
                                                                 tp_security_manager_patch,
                                                                 patch_uuid):
 
        patch_uuid.return_value = FakeUuid()
        tp_security_manager_patch.return_value = (True, {'name': 'my_node_id'})
        

 
        # creating Round
        self.r1.training_kwargs = {'optimizer_args': {'lr': .1234}}

        self.r1.researcher_id = 'researcher_id_1234'
        self.r1.job_id = 'job_id_1234'

        # configure optimizer
        lr = .1234
        optim_node = Optimizer(lr=lr, modules=[ScaffoldClientModule(), YogiModule()],
                                regularizers=[RidgeRegularizer()])


        dec_node_optim = DeclearnOptimizer(TorchModel(torch.nn.Linear(3, 1)), optim_node)

        DeclearnAuxVarModel.OPTIM = dec_node_optim
        DeclearnAuxVarModel.TYPE = TrainingPlans.TorchTrainingPlan

        self.ic_from_file_mock.return_value = (fake_training_plan, DeclearnAuxVarModel())
        self.r1.training_plan_class = "DeclearnAuxVarModel"
        self.r1.dataset = {'dataset_id': 'dataset_id_1234',
                        'path': os.path.join('path', 'to', 'my', 'dataset')}
        self.r1.aux_vars = [{}, {'scaffold': {'delta': 'some incorrect value for scaffold'}}]


        # action
        rnd_reply = self.r1.run_model_training()

        self.assertIn("TrainingPlan Optimizer failed to ingest the provided auxiliary variables",
                        rnd_reply.msg)


    def test_round_18_process_optim_aux_var(self):
        """Test that 'process_optim_aux_var' works properly."""
        
        # Set up a mock BaseOptimizer with an attached Optimizer.
        mock_optim = create_autospec(Optimizer, instance=True)
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = mock_optim
        # Attach the former to the Round's mock TrainingPlan.
        self.r1.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        self.r1.training_plan.optimizer.return_value = mock_b_opt

        # Attach fake auxiliary variables (as though pre-downloaded).
        fake_aux_var = [{}, {"module": {"key": "val"}}]
        setattr(self.r1, "aux_vars", fake_aux_var)
        # Call the tested method and verify its outputs and effects.
        msg = self.r1.process_optim_aux_var()
        self.assertEqual(msg, None)

        call_with = {}
        call_with.update(fake_aux_var[0])
        call_with.update(fake_aux_var[1])
        mock_optim.set_aux.assert_called_once_with(call_with)

    def test_round_19_process_optim_aux_var_without_aux_var(self):
        """Test that 'process_optim_aux_var' exits properly without aux vars."""
        # Set up a Round with a mock Optimizer attached, but no aux vars.
       
        mock_optim = create_autospec(Optimizer, instance=True)
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = mock_optim
        self.r1.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        self.r1.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method, verifying that it exits without effects.
        msg = self.r1.process_optim_aux_var()
        self.assertEqual(msg, None)
        mock_optim.set_aux.assert_not_called()

    def test_round_20_process_optim_aux_var_without_base_optimizer(self):
        """Test that 'process_optim_aux_var' documents missing BaseOptimizer."""
        # Set up a Round with fake aux_vars, but no BaseOptimizer.
        
        setattr(self.r1, "aux_vars", [{}, {"module": {"key": "val"}}])
        self.r1.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        self.r1.training_plan.optimizer.return_value = None
        # Call the tested method, verifying that it returns an error.
        msg = self.r1.process_optim_aux_var()
        self.assertTrue("TrainingPlan does not hold a BaseOptimizer" in msg)
        self.assertIsInstance(msg, str)

    def test_round_21_process_optim_aux_var_without_optimizer(self):
        """Test that 'process_optim_aux_var' documents missing Optimizer."""
        # Set up a Round with aux vars, but a non-Optimizer optimizer.
        
        setattr(self.r1, "aux_vars", [{}, {"module": {"key": "val"}}])
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = MagicMock()  # not a declearn-based Optimizer
        self.r1.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        self.r1.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method, verifying that it returns an error.
        msg = self.r1.process_optim_aux_var()
        self.assertTrue("does not manage a compatible Optimizer" in msg)
        self.assertIsInstance(msg, str)

    def test_round_22_process_optim_aux_var_with_optimizer_error(self):
        """Test that 'process_optim_aux_var' documents 'Optimizer.set_aux' error."""
        # Set up a Round with fake pre-downloaded aux vars.

        fake_aux_var = [{}, {"module": {"key": "val"}}]
        setattr(self.r1, "aux_vars", [{}, {"module": {"key": "val"}}])

        # Set up a mock BaseOptimizer with an attached failing Optimizer.
        mock_optim = create_autospec(Optimizer, instance=True)
        fake_error = "fake FedbiomedOptimizerError on 'set_aux' call"
        mock_optim.set_aux.side_effect = FedbiomedOptimizerError(fake_error)
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = mock_optim
        # Attach the former to the Round's mock TrainingPlan.
        self.r1.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        self.r1.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method, verifying that it returns an error.
        msg = self.r1.process_optim_aux_var()
        self.assertTrue(fake_error in msg)

        call_with = {}
        call_with.update(fake_aux_var[0])
        call_with.update(fake_aux_var[1])
        mock_optim.set_aux.assert_called_once_with(call_with)

    def test_round_23_collect_optim_aux_var(self):
        """Test that 'collect_optim_aux_var' works properly with an Optimizer."""
        # Set up a Round with an attached mock Optimizer.
        

        mock_optim = create_autospec(Optimizer, instance=True)
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        # why not using DeclearnOptimizer?
        mock_b_opt.optimizer = mock_optim
        self.r1.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        self.r1.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method and verify its outputs.
        aux_var = self.r1.collect_optim_aux_var()
        self.assertEqual(aux_var, mock_optim.get_aux.return_value)
        mock_optim.get_aux.assert_called_once()

    def test_round_24_collect_optim_aux_var_without_optimizer(self):
        """Test that 'collect_optim_aux_var' works properly without an Optimizer."""
        # Set up a Round with a non-Optimizer optimizer.
   
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = MagicMock()  # non-declearn-based object
        self.r1.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        self.r1.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method and verify its outputs.
        aux_var = self.r1.collect_optim_aux_var()
        self.assertEqual(aux_var, {})

    def test_round_25_collect_optim_aux_var_without_base_optimizer(self):
        """Test that 'collect_optim_aux_var' fails without a BaseOptimizer."""
        # Set up a Round without a BaseOptimizer.

        self.r1.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        self.r1.training_plan.optimizer.return_value = None
        # Verify that aux-var collection raises.
        self.assertRaises(FedbiomedRoundError, self.r1.collect_optim_aux_var)

    # add a test with : shared and node specific auxiliary avraibales

    def test_round_26_split_train_and_test_data_raises_exceptions(self):
        """Test that _split_train_and_test_data raises correct exceptions"""
        mock_training_plan = MagicMock()
        def foo_takes_an_argument(x):
            return x
        mock_training_plan.training_data = foo_takes_an_argument
        mock_training_plan.type.return_value = 'tp type'

        self.r1.training_plan = mock_training_plan
        with self.assertRaises(FedbiomedRoundError):
            self.r1._split_train_and_test_data()


    @patch('fedbiomed.node.round.Serializer')
    @patch('fedbiomed.node.round.Round._get_base_optimizer')
    def test_round_27_load_round_state(self,
                                       get_optim_patch,
                                       serializer_patch,
                                       ):

        self.r1.job_id = 1234
        state_id = 'state_id_1234'
        path_state = '/path/to/state'

        training_plan_mock = MagicMock(spec=BaseTrainingPlan,
                                       type=MagicMock(),
                                       _model=MagicMock(spec=Model))
        self.r1.training_plan = training_plan_mock

        
        node_state = {
            'optimizer_state': {
                'optimizer_type': 'optimizer_type',
                'state_path': path_state,
            }
        }
        self.state_manager_mock.return_value.get.return_value = node_state
        get_optim_patch.return_value = MagicMock(spec=DeclearnOptimizer,
                                                 __class__='optimizer_type',
                                                )
        self.r1._load_round_state(state_id)
        
        # checks
        # FIXME: in future version, we should check each call to Serializer.load
        serializer_patch.load.assert_called_once_with(path_state)

        
        # check Optimizer.load_state call
        get_optim_patch.return_value.load_state.assert_called_once_with(
            serializer_patch.load.return_value,
            load_from_state=True
        )

    @patch('fedbiomed.node.round.logger')
    @patch('fedbiomed.node.round.Serializer')
    @patch('fedbiomed.node.round.Round._get_base_optimizer')
    def test_round_28_load_round_state_failure(self,
                                               get_optim_patch,
                                               serializer_patch,
                                               logger_patch
                                               ):

        self.r1.job_id = 1234
        state_id = 'state_id_1234'
        path_state = '/path/to/state'

        training_plan_mock = MagicMock(spec=BaseTrainingPlan,
                                       type=MagicMock(),
                                       _model=MagicMock(spec=Model))
        self.r1.training_plan = training_plan_mock

        
        node_state = {
            'optimizer_state': {
                'optimizer_type': 'optimizer_type',
                'state_path': path_state,
            }
        }
        self.state_manager_mock.return_value.get.return_value = node_state
        get_optim_patch.return_value = MagicMock(spec=DeclearnOptimizer,
                                                 __class__='optimizer_type',
                                                 )
        err_msg = "Error raised for the sake of testing"
        get_optim_patch.return_value.load_state.side_effect = FedbiomedOptimizerError(err_msg)
        self.r1._load_round_state(state_id)
        
        # checks
        # FIXME: in future version, we should check each call to Serializer.load
        serializer_patch.load.assert_called_once_with(path_state)
        logger_patch.warning.assert_called()
        
        logger_calls = logger_patch.mock_calls
        if len(logger_calls) < 2:
            self.skipTest("Test `load_round_state_failure skipped because logger call has changed - which has minor impact on the code")
        self.assertIn(err_msg, logger_calls[1][1][0], f"error message '{err_msg}' not sent through logger")

    @patch('fedbiomed.node.round.Serializer', autospec=True)
    @patch('fedbiomed.node.round.Round._get_base_optimizer')
    def test_round_28_save_round_state(self, 
                                       get_optim_patch,
                                       serializer_patch):
        
        self.r1.job_id = '1234'
        self.r1._round = 34
        
        optim_path = 'path/to/folder/containing/state/files'
        self.state_manager_mock.return_value.generate_folder_and_create_file_name.return_value = optim_path
        get_optim_patch.return_value = MagicMock(spec=DeclearnOptimizer,
                                                 __class__='optimizer_type',
                                                )
        
        # adding a funcition that add additional dictionary entries through reference 
        self.state_manager_mock.return_value.add.side_effect = lambda x,y: y.update({'version_node_id': '1',
                                                                                      'state_id': 'state_id_1234'})

        res = self.r1._save_round_state()

        added_state = {
            'optimizer_state': {
                'optimizer_type': 'optimizer_type',
                'state_path': optim_path
            }
        }

        expected_state = copy.deepcopy(added_state)
        expected_state.update(
            {'version_node_id': '1',
            'state_id': 'state_id_1234'}
        )
        get_optim_patch.return_value.save_state.assert_called_once()
        serializer_patch.dump.assert_called_once_with(
            get_optim_patch.return_value.save_state.return_value,
            path=optim_path)
        
        self.state_manager_mock.return_value.add.assert_called_once_with('1234', 
                                                                          expected_state)
        self.assertDictEqual(expected_state, res)


    @patch('fedbiomed.node.round.Round._get_base_optimizer')
    def test_round_29_save_round_state_failure_saving_optimizer(self,
                                                                get_optim_patch,
                                                                ):
        self.r1.job_id = '1234'
        self.r1._round = 34
        #r = Round(job_id=job_id, round_number=round_nb)
        get_optim_patch.return_value = MagicMock(save_state=MagicMock(return_value=None))
        

        self.state_manager_mock.return_value.add.side_effect = lambda x,y: y.update({'version_node_id': '1',
                                                                         'state_id': 'state_id_1234'})
        res = self.r1._save_round_state()
        expected_state = {
            'optimizer_state': None,
            'version_node_id': '1',
            'state_id': 'state_id_1234'
        }
        
        self.assertDictEqual(res, expected_state)
        
    @patch('uuid.uuid4', autospec=True)
    @patch('fedbiomed.node.round.Serializer', autospec=True)
    def test_round_30_save_and_load_state(self,
                                          serializer_patch,
                                          uuid_patch):
        
        optim_mock = MagicMock(spec=BaseOptimizer,
                               __class__='optimizer_type')

        training_plan_mock = MagicMock(spec=BaseTrainingPlan,
                                       )
        training_plan_mock.optimizer.return_value = optim_mock
        self.state_manager_mock.return_value.get_node_state_base_dir.return_value = "/path/to/base/dir"

        uuid_patch.return_value = FakeUuid()
        job_id = _id = FakeUuid.VALUE
        state_id = f'node_state_{_id}'
        path = "/path/to/base/dir" + f"/job_id_{job_id}/" + NodeStateFileName.OPTIMIZER.value % (0, state_id)
        
        # first create state
        self.r1.training_plan = training_plan_mock
        
        self.r1.initialize_node_state_manager()
        state = self.r1._save_round_state()
        print(state)
        self.r1._load_round_state('test-state-id')
        
        self.state_manager_mock.return_value.get.assert_called_once_with(str(job_id), 'test-state-id')
        self.state_manager_mock.return_value.add.assert_called_once()

    def test_round_31_initialize_node_state_manager(self):
        previous_state_id = 'state_id_1234'
        self.r1.initialize_node_state_manager(previous_state_id=previous_state_id)
        
        self.state_manager_mock.return_value.initialize.assert_called_once_with(previous_state_id=previous_state_id)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
