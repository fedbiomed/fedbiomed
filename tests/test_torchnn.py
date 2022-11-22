import unittest
import os
import logging
import re
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import Module
from testsupport.base_fake_training_plan import BaseFakeTrainingPlan
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.metrics import MetricTypes


# define TP outside of test class to avoid indentation problems when exporting class to file
class TrainingPlan(TorchTrainingPlan):
    def __init__(self):
        super(TrainingPlan, self).__init__()
        self.lin1 = nn.Linear(4, 2)

    def test_method(self):
        return True


class FakeDPController:
    def validate_and_fix_model(self, model):
        return model

    def before_training(self, model, optimizer, loader):
        return model, optimizer, loader

class TestTorchnn(unittest.TestCase):
    """
    Test the Torchnn class
    """

    model = Module()
    optimizer = Adam([torch.zeros([2, 4])])

    class FakeTrainingArgs:

        def pure_training_arguments(self):
            return {"dry_run": True, "num_updates": 1, "batch_size": 10, "log_interval": 10,
                    "epochs": None, "batch_maxnum": None}

        def optimizer_arguments(self):
            return {"lr": 0.0001}

        def dp_arguments(self):
            return None

    class CustomDataset(Dataset):
        """ Create PyTorch Dataset for test purposes """

        def __init__(self):
            self.X_train = [[1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3]]
            self.Y_train = [1, 2, 3, 4, 5, 6]

        def __len__(self):
            return len(self.Y_train)

        def __getitem__(self, idx):
            return self.X_train[idx], self.Y_train[idx]

    # before the tests
    def setUp(self):

        self.patcher = patch.multiple(TorchTrainingPlan, __abstractmethods__=set())
        self.patcher.start()

        self.TrainingPlan = TrainingPlan
        self.params = {'one': 1, '2': 'two'}
        self.tmpdir = '.'

    # after the tests
    def tearDown(self):
        self.patcher.stop()
        pass

    #
    # TODO : add tests for checking the training payload
    #
    def test_torch_training_plan_01_save_model(self):
        """Test save model method of troch traning plan"""
        tp1 = TorchTrainingPlan()
        modulename = 'tmp_model'
        file = self.tmpdir + os.path.sep + modulename + '.py'

        if os.path.isfile(file):
            os.remove(file)

        tp1.save_code(file)
        self.assertTrue(os.path.isfile(file))
        os.remove(file)

    @patch("fedbiomed.common.training_plans.TorchTrainingPlan._configure_dependencies")
    @patch("fedbiomed.common.training_plans.TorchTrainingPlan._configure_model_and_optimizer")
    @patch("fedbiomed.common.training_plans._torchnn.deepcopy")
    def test_torch_training_plan_02_post_init(self, mock_deepcopy, conf_optimizer_model, conf_deps):

        mock_deepcopy.return_value = []
        conf_optimizer_model.return_value = None
        conf_deps.return_value = None

        tp = TorchTrainingPlan()
        tp._model = Module()
        tp.post_init({}, TestTorchnn.FakeTrainingArgs())

        self.assertEqual(tp._log_interval, 10)
        self.assertEqual(tp._num_updates, 1)
        self.assertEqual(tp._dry_run, True)

        conf_optimizer_model.assert_called_once()
        conf_deps.assert_called_once()
        mock_deepcopy.assert_called_once()

    @patch('fedbiomed.common.training_plans.BaseTrainingPlan.add_dependency')
    def test_torch_training_plan_03_configure_deps(self, add_dependency):
        """Test private method configure dependencies """
        add_dependency.return_value = None

        # Test default init dependencies
        tp = TorchTrainingPlan()
        add_dependency.reset_mock()
        tp._configure_dependencies()
        add_dependency.assert_called_once()

        # Wrong 1 -----------------------------------------------------------------
        class FakeWrongTP(BaseFakeTrainingPlan):
            def init_dependencies(self, invalid):
                pass

        tp = FakeWrongTP()
        with self.assertRaises(FedbiomedTrainingPlanError):
            tp._configure_dependencies()

        # Wrong 2 -----------------------------------------------------------------
        class FakeWrongTP(BaseFakeTrainingPlan):
            def init_dependencies(self):
                return None

        tp = FakeWrongTP()
        with self.assertRaises(FedbiomedTrainingPlanError):
            tp._configure_dependencies()

    def test_torch_training_plan_04_configure_model_and_optimizer_1(self):
        """Tests method for configuring model and optimizer """

        tp = TorchTrainingPlan()

        # Special methods without arguments ----------------------------------------------
        class FakeTP(BaseFakeTrainingPlan):
            def init_model(self):
                return TestTorchnn.model

            def init_optimizer(self):
                return TestTorchnn.optimizer

        tp = FakeTP()
        tp._optimizer_args = {}
        tp._model_args = {}
        tp._dp_controller = FakeDPController()
        tp._configure_model_and_optimizer()
        self.assertEqual(tp._optimizer, TestTorchnn.optimizer)
        self.assertEqual(tp._model, TestTorchnn.model)
        # ---------------------------------------------------------------------------------

    def test_torch_training_plan_05_configure_model_and_optimizer_2(self):
        """Tests method for configuring model and optimizer with arguments """

        class FakeTP(BaseFakeTrainingPlan):
            def init_model(self, model_args):
                return TestTorchnn.model

            def init_optimizer(self, optimizer_args):
                return TestTorchnn.optimizer

        tp = FakeTP()
        tp._optimizer_args = {}
        tp._model_args = {}
        tp._dp_controller = FakeDPController()
        tp._configure_model_and_optimizer()
        self.assertEqual(tp._optimizer, TestTorchnn.optimizer)
        self.assertEqual(tp._model, TestTorchnn.model)
        # -----------------------------------------------------------------------------------

    def test_torch_training_plan_06_configure_model_and_optimizer_test_invalid_types(self):
        """Tests method for configuring model and optimizer when they return invalid types """

        class FakeTP(BaseFakeTrainingPlan):
            def init_model(self, model_args):
                return None

            def init_optimizer(self, optimizer_args):
                return TestTorchnn.optimizer

        tp = FakeTP()
        tp._optimizer_args = {}
        tp._model_args = {}
        tp._dp_controller = FakeDPController()

        with self.assertRaises(FedbiomedTrainingPlanError):
            tp._configure_model_and_optimizer()

        # -----------------------------------------------------------------------------------

        class FakeTP(BaseFakeTrainingPlan):
            def init_model(self, model_args):
                return TestTorchnn.model

            def init_optimizer(self, optimizer_args):
                return None

        tp = FakeTP()
        tp._optimizer_args = {}
        tp._model_args = {}
        tp._dp_controller = FakeDPController()
        with self.assertRaises(FedbiomedTrainingPlanError):
            tp._configure_model_and_optimizer()

    def test_torch_training_plan_07_configure_model_and_optimizer_test_invalid_types(self):
        """Tests method for configuring model and optimizer with wrong number of arguments """

        class FakeTP(BaseFakeTrainingPlan):
            def init_model(self, model_args, x):
                return None

            def init_optimizer(self, optimizer_args):
                return TestTorchnn.optimizer

        tp = FakeTP()
        tp._optimizer_args = {}
        tp._model_args = {}
        with self.assertRaises(FedbiomedTrainingPlanError):
            tp._configure_model_and_optimizer()

        # -----------------------------------------------------------------------------------
        class FakeTP(BaseFakeTrainingPlan):
            def init_model(self, model_args):
                return TestTorchnn.model

            def init_optimizer(self, optimizer_args, x):
                return None

        tp = FakeTP()
        tp._optimizer_args = {}
        tp._model_args = {}
        tp._dp_controller = FakeDPController()

        with self.assertRaises(FedbiomedTrainingPlanError):
            tp._configure_model_and_optimizer()

    def test_torch_training_plan_08_getters(self):
        """Tests getter methods. """

        tp = TorchTrainingPlan()
        tp._model = TestTorchnn.model
        tp._optimizer = TestTorchnn.optimizer

        m = tp.model()
        self.assertEqual(m, TestTorchnn.model)

        o = tp.optimizer()
        self.assertEqual(o, TestTorchnn.optimizer)

        ma = {"a": 12}
        ta = {"t": 13}
        oa = {"y": 14}
        ip = {"s": 15}
        tp._model_args = ma
        tp._training_args = ta
        tp._optimizer_args = oa
        tp._init_params = ip

        r_ma = tp.model_args()
        r_ta = tp.training_args()
        r_oa = tp.optimizer_args()
        r_ip = tp.initial_parameters()

        self.assertEqual(r_ma, ma)
        self.assertEqual(r_oa, oa)
        self.assertEqual(r_ta, ta)
        self.assertEqual(r_ip, ip)

    def test_torch_training_plan_09_save_and_load_params(self):
        """ Test save and load parameters """
        tp1 = TorchTrainingPlan()
        tp1._model = torch.nn.Module()
        paramfile = self.tmpdir + '/tmp_params.pt'

        if os.path.isfile(paramfile):
            os.remove(paramfile)

        # save/load from/to variable
        tp1.save(paramfile, self.params)
        self.assertTrue(os.path.isfile(paramfile))
        params2 = tp1.load(paramfile, True)

        self.assertTrue(type(params2) is dict)
        self.assertEqual(self.params, params2)

        # save/load from/to object params
        tp1.save(paramfile)
        tp2 = TorchTrainingPlan()
        tp2._model = torch.nn.Module()
        tp2.load(paramfile)
        self.assertTrue(type(params2) is dict)

        sd1 = tp1.model().state_dict()
        sd2 = tp2.model().state_dict()

        # verify we have an equivalent state dict
        for key in sd1:
            self.assertTrue(key in sd2)

        for key in sd2:
            self.assertTrue(key in sd1)

        for (key, value) in sd1.items():
            self.assertTrue(torch.all(torch.isclose(value, sd2[key])))

        os.remove(paramfile)

    @patch('torch.nn.Module.__call__')
    def test_torch_training_plan_10_testing_routine(self,
                                         patch_model_call):

        history_monitor = MagicMock()
        history_monitor.add_scalar = MagicMock(return_value=None)
        tp = TorchTrainingPlan()
        tp._model = torch.nn.Module()

        # Create custom test data and set data loader for training plan
        test_dataset = TestTorchnn.CustomDataset()
        data_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        # Patch predict call (self(data))
        patch_model_call.return_value = torch.tensor(test_dataset.Y_train)

        # Raises error if there is no testing data loader is defined ----------------------------------
        with self.assertRaises(FedbiomedTrainingPlanError):
            tp.testing_routine(metric=MetricTypes.ACCURACY,
                               metric_args={},
                               history_monitor=history_monitor,
                               before_train=True)

        # Run testing routine -------------------------------------------------------------------------
        tp.set_data_loaders(test_data_loader=data_loader, train_data_loader=data_loader)
        tp.testing_routine(metric=MetricTypes.ACCURACY,
                           metric_args={},
                           history_monitor=history_monitor,
                           before_train=True)
        history_monitor.add_scalar.assert_called_once_with(metric={'ACCURACY': 1.0},
                                                           iteration=1,
                                                           epoch=None,
                                                           test=True,
                                                           test_on_local_updates=False,
                                                           test_on_global_updates=True,
                                                           total_samples=6,
                                                           batch_samples=6,
                                                           num_batches=1)
        history_monitor.add_scalar.reset_mock()

        # If metric is None --------------------------------------------------------------------------------
        tp.testing_routine(metric=None,
                           metric_args={},
                           history_monitor=history_monitor,
                           before_train=True)
        history_monitor.add_scalar.assert_called_once_with(metric={'ACCURACY': 1.0},
                                                           iteration=1,
                                                           epoch=None,
                                                           test=True,
                                                           test_on_local_updates=False,
                                                           test_on_global_updates=True,
                                                           total_samples=6,
                                                           batch_samples=6,
                                                           num_batches=1)
        history_monitor.add_scalar.reset_mock()

        # If prediction raises an exception
        patch_model_call.side_effect = Exception
        with self.assertRaises(FedbiomedTrainingPlanError):
            tp.testing_routine(metric=MetricTypes.ACCURACY,
                               metric_args={},
                               history_monitor=history_monitor,
                               before_train=True)
        patch_model_call.side_effect = None

        # Testing routine with testing step ---------------------------------------------------------------------
        class TrainingPlanWithTestingStep(BaseFakeTrainingPlan):
            def __init__(self):
                super(TrainingPlanWithTestingStep, self).__init__()

            def testing_step(self, data, target):  # noqa
                return {'Metric': 12}

        tp = TrainingPlanWithTestingStep()
        tp._model = torch.nn.Module()
        tp.set_data_loaders(test_data_loader=data_loader, train_data_loader=data_loader)
        tp.testing_routine(metric=MetricTypes.ACCURACY,
                           metric_args={},
                           history_monitor=history_monitor,
                           before_train=True)
        history_monitor.add_scalar.assert_called_once_with(metric={'Metric': 12.0},
                                                           iteration=1,
                                                           epoch=None,
                                                           test=True,
                                                           test_on_local_updates=False,
                                                           test_on_global_updates=True,
                                                           total_samples=6,
                                                           batch_samples=6,
                                                           num_batches=1)
        with patch.object(TrainingPlanWithTestingStep, 'testing_step') as patch_testing_step:
            patch_testing_step.side_effect = Exception
            with self.assertRaises(FedbiomedTrainingPlanError):
                tp.testing_routine(metric=MetricTypes.ACCURACY,
                                   metric_args={},
                                   history_monitor=history_monitor,
                                   before_train=True)

            # If testing_step returns none
            patch_testing_step.side_effect = None
            patch_testing_step.return_value = None
            with self.assertRaises(FedbiomedTrainingPlanError):
                tp.testing_routine(metric=MetricTypes.ACCURACY,
                                   metric_args={},
                                   history_monitor=history_monitor,
                                   before_train=True)

    def test_torch_training_plan_11_logging_progress_computation(self):
        """Test the logging facility during training

        ## Test two scenarios
        ### Scenario 1
        - batch size = 5
        - total num samples = 15 (5*3 = batch_size * num_batches)
        - num_updates = 3

        The expected behaviour is that the first iteration should report a progress of 5/15 (33%),
        while the second iteration should report a progress of 10/15 (66%). Last iteration should report
        15/15 (100%). Only one epoch should be completed.

        ### Scenario 2
        - batch size = 3
        - total num samples = 5
        - num_updates = 3

        The expected behaviour is that the first update should report:
            - epoch 1
            - progress of 3/9 (33%),
        the second update should report:
            - epoch 2
            - progress of 6/9 (66%),
        the third and final update should report:
            - epoch 2
            - progress of 9/9 (100%)
        """

        def conduct_logging_test(num_samples, batch_size, num_updates):
            tp = TorchTrainingPlan()
            with patch.object(tp, 'init_model', new=lambda _: MagicMock(spec=torch.nn.Module)), \
                    patch.object(tp, 'init_optimizer', new=lambda _: MagicMock(spec=torch.optim.Adam)):
                tp.post_init({}, TestTorchnn.FakeTrainingArgs())
            tp._dry_run = False
            tp._log_interval = 1
            tp._training_args['num_updates'] = num_updates
            tp.training_data_loader = MagicMock(spec=torch.utils.data.DataLoader)

            mocked_loss_result = MagicMock()
            mocked_loss_result.item.return_value = 0.
            tp.training_step = lambda x, y: mocked_loss_result

            custom_dataset = self.CustomDataset()
            x_train = torch.Tensor(custom_dataset.X_train)
            y_train = torch.Tensor(custom_dataset.Y_train)
            fake_data = {'modality1': x_train, 'modality2': x_train}
            fake_target = (y_train, y_train)
            tp.training_data_loader.__iter__.return_value = itertools.cycle([(fake_data, fake_target)])
            tp.training_data_loader.__len__.return_value = int(np.ceil(num_samples // batch_size))
            tp.training_data_loader.batch_size = batch_size
            tp.training_data_loader.dataset = MagicMock()
            tp.training_data_loader.dataset.__len__.return_value = num_samples

            tp._dp_controller = FakeDPController()

            with self.assertLogs('fedbiomed', logging.DEBUG) as captured:
                tp.training_routine()
                training_progress_messages = [x for x in captured.output if re.search('Train Epoch: ', x)]
                self.assertEqual(len(training_progress_messages), num_updates)  # Double-check correct number of train iters
                for i, logging_message in enumerate(training_progress_messages):
                    logged_num_processed_samples = int(logging_message.split('[')[1].split('/')[0])
                    logged_total_num_samples = int(logging_message.split('/')[1].split()[0])
                    logged_percent_progress = float(logging_message.split('(')[1].split('%')[0])
                    self.assertEqual(logged_num_processed_samples, (i+1)*batch_size)
                    self.assertEqual(logged_total_num_samples, batch_size*num_updates)
                    self.assertEqual(logged_percent_progress, round(100*(i+1)/num_updates))

        num_samples = 15
        batch_size = 5
        num_updates = 3
        conduct_logging_test(num_samples, batch_size, num_updates)
        num_samples = 5
        batch_size = 3
        num_updates = 3
        conduct_logging_test(num_samples, batch_size, num_updates)

    def test_torch_training_plan_12_num_updates(self):
        """Test that num_updates parameter is respected correctly.

        In the following test, we make sure that no matter the dataset size, nor the batch size, we always perform the
        number of updates requested by the researcher. Remember each update corresponds to one optimizer step, i.e.
        one batch.
        """
        tp = TorchTrainingPlan()
        tp._model = MagicMock()
        tp._set_device = MagicMock()
        tp._batch_maxnum = 0
        tp._optimizer = MagicMock()
        tp._optimizer.step = MagicMock()
        tp.training_step = MagicMock(return_value=Variable(torch.Tensor([0]), requires_grad=True))
        tp._log_interval = 1000  # essentially disable logging
        tp._dry_run = False
        tp._training_args = {}
        tp._dp_controller = FakeDPController()

        def setup_tp(tp, num_samples, batch_size, num_updates):
            """Utility function to prepare the TrainingPlan test"""
            tp._optimizer.step.reset_mock()
            tp.training_data_loader = MagicMock(spec=torch.utils.data.DataLoader)
            num_batches = num_samples // batch_size
            tp.training_data_loader.__iter__.return_value = itertools.cycle(
                num_batches*[(
                    batch_size*[MagicMock(spec=torch.Tensor)],
                    batch_size*[MagicMock(spec=torch.Tensor)]
                 )],
            )
            tp.training_data_loader.__len__.return_value = num_batches
            tp.training_data_loader.dataset = MagicMock()
            tp.training_data_loader.dataset.__len__ = num_samples
            tp._training_args['num_updates'] = num_updates
            tp._training_args['epochs'] = None
            tp._training_args['batch_maxnum'] = None
            return tp

        # Case where we do 1 single epoch with 1 batch
        tp = setup_tp(tp, num_samples=5, batch_size=5, num_updates=1)
        tp.training_routine(None, None)
        self.assertEqual(tp._optimizer.step.call_count, 1)

        # Case where researcher asks for less updates than would be needed to complete even the first epoch
        tp = setup_tp(tp, num_samples=15, batch_size=5, num_updates=2)
        tp.training_routine(None, None)
        self.assertEqual(tp._optimizer.step.call_count, 2)

        # Case where researcher asks for a num_updates that is not a multiple of the num batches per epoch
        tp = setup_tp(tp, num_samples=15, batch_size=5, num_updates=7)
        tp.training_routine(None, None)
        self.assertEqual(tp._optimizer.step.call_count, 7)

        # Case where researcher asks for a num_updates that is a multiple of the num batches per epoch
        tp = setup_tp(tp, num_samples=15, batch_size=5, num_updates=9)
        tp.training_routine(None, None)
        self.assertEqual(tp._optimizer.step.call_count, 9)

        # Case where researcher also set batch_maxnum. In this case we still respect the num_updates, therefore
        # more epochs (each one with only batch_maxnum iterations_ will be performed)
        tp = setup_tp(tp, num_samples=45, batch_size=5, num_updates=3)
        tp._batch_maxnum = 1
        tp.training_routine(None, None)
        self.assertEqual(tp._optimizer.step.call_count, 3)

        # Case where the batch_maxnum is the same as the num_updates
        tp = setup_tp(tp, num_samples=45, batch_size=5, num_updates=3)
        tp._batch_maxnum = 3
        tp.training_routine(None, None)
        self.assertEqual(tp._optimizer.step.call_count, 3)

        tp = setup_tp(tp, num_samples=10, batch_size=5, num_updates=6)
        tp._batch_maxnum = 3
        tp.training_routine(None, None)
        self.assertEqual(tp._optimizer.step.call_count, 6)


class TestSendToDevice(unittest.TestCase):

    def setUp(self) -> None:
        self.patcher = patch.multiple(TorchTrainingPlan, __abstractmethods__=set())
        self.patcher.start()
        self.cuda = torch.device('cuda')
        self.cpu = torch.device('cpu')

    def tearDown(self) -> None:
        self.patcher.stop()

    @patch('torch.Tensor.to')
    def test_send_to_device_01_send_tensor_to_device(self, patch_tensor_to):
        """Test basic case of sending a tensor to cpu and gpu."""
        tp = TorchTrainingPlan()
        t = torch.Tensor([0])
        t = tp.send_to_device(t, self.cpu)
        patch_tensor_to.assert_called_once()
        t = torch.Tensor([0])
        t = tp.send_to_device(t, self.cuda)
        self.assertEqual(patch_tensor_to.call_count, 2)

    def test_send_to_device_02_nested_collections(self):
        """Test case where tensors are contained within nested collections."""
        tp = TorchTrainingPlan()
        t = torch.Tensor([0])
        ll = [t]*3
        d = {'key1': ll, 'key2': t}
        tup = (ll, d, t)
        output = tp.send_to_device(tup, torch.device('cpu'))

        self.assertIsInstance(output[0], type(tup[0]))
        for el in output[0]:
            self.assertIsInstance(el, torch.Tensor)

        self.assertIsInstance(output[1], type(tup[1]))
        for key, val in output[1].items():
            self.assertIsInstance(val, type(d[key]))
            for el in val:
                self.assertIsInstance(el, torch.Tensor)

        self.assertIsInstance(output[2], torch.Tensor)

        with patch('torch.Tensor.to') as p:
            _ = tp.send_to_device(tup, torch.device('cuda'))
            self.assertEqual(p.call_count, 8)

    def test_send_to_device_03_unsupported_parameters(self):
        """Ensure that the function correctly raises errors with wrong parameters."""
        tp = TorchTrainingPlan()
        with self.assertRaises(FedbiomedTrainingPlanError):
            tp.send_to_device("unsupported variable type", self.cpu)


class TestTorchNNTrainingRoutineDataloaderTypes(unittest.TestCase):
    """Test training routine when data loaders return different data types.

    Dataloaders in Fed-BioMed should always return a tuple (data, target). In the base case, the `data` and `target`
    are torch Tensors. However, they could also be lists, tuples or dicts. While the use is responsible for handling
    these data types correctly in the `training_step` routine, we must make sure that the training routine as a whole
    runs correctly.
    """

    @staticmethod
    def iterate_once(return_value):
        """Utility create generators that load a data sample only once."""
        yield return_value

    def setUp(self) -> None:
        self.patcher = patch.multiple(TorchTrainingPlan, __abstractmethods__=set())
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()

    @patch('torch.Tensor.backward')
    def test_torch_training_plan_training_routine_loader_types_01_tensors(self, patch_tensor_backward):
        tp = TorchTrainingPlan()
        with patch.object(tp, 'init_model', new=lambda _: MagicMock(spec=torch.nn.Module)), \
             patch.object(tp, 'init_optimizer', new=lambda _: MagicMock(spec=torch.optim.Adam)):
            tp.post_init({}, TestTorchnn.FakeTrainingArgs())
        tp._dry_run = False
        tp.training_data_loader = MagicMock(spec=DataLoader(MagicMock(spec=Dataset)), batch_size=1)
        gen_data = TestTorchNNTrainingRoutineDataloaderTypes.iterate_once(
            (torch.Tensor([0]), torch.Tensor([1])))
        tp.training_data_loader.__iter__.return_value = gen_data
        tp.training_data_loader.__len__.return_value = 1
        tp.training_step = MagicMock(return_value=torch.Tensor([0.]))

        class FakeDPController:
            def before_training(self, model, optimizer, loader):
                return tp._model, tp._optimizer, tp.training_data_loader
        tp._dp_controller = FakeDPController()

        tp.training_routine()
        tp.training_step.assert_called_once_with(torch.Tensor([0]), torch.Tensor([1]))
        patch_tensor_backward.assert_called_once()

    @patch('torch.Tensor.backward')
    def test_torch_training_plan_training_routine_loader_types_02_tuples(self, patch_tensor_backward):
        tp = TorchTrainingPlan()
        with patch.object(tp, 'init_model', new=lambda _: MagicMock(spec=torch.nn.Module)), \
                patch.object(tp, 'init_optimizer', new=lambda _: MagicMock(spec=torch.optim.Adam)):
            tp.post_init({}, TestTorchnn.FakeTrainingArgs())
        tp._dry_run = False
        tp.training_data_loader = MagicMock(spec=DataLoader(MagicMock(spec=Dataset)), batch_size=1)
        gen_data = TestTorchNNTrainingRoutineDataloaderTypes.iterate_once(
            ((torch.Tensor([0]), torch.Tensor([1])), torch.Tensor([2])))
        tp.training_data_loader.__iter__.return_value = gen_data
        tp.training_data_loader.__len__.return_value = 1
        tp.training_data_loader.batch_size = 1
        tp.training_step = MagicMock(return_value=torch.Tensor([0.]))

        class FakeDPController:
            def before_training(self, model, optimizer, loader):
                return tp._model, tp._optimizer, tp.training_data_loader
        tp._dp_controller = FakeDPController()

        tp.training_routine()
        tp.training_step.assert_called_once_with((torch.Tensor([0]), torch.Tensor([1])), torch.Tensor([2]))
        patch_tensor_backward.assert_called_once()

    @patch('torch.Tensor.backward')
    def test_torch_training_plan_training_routine_loader_types_03_dicts(self, patch_tensor_backward):
        tp = TorchTrainingPlan()
        with patch.object(tp, 'init_model', new=lambda _: MagicMock(spec=torch.nn.Module)), \
             patch.object(tp, 'init_optimizer', new=lambda _: MagicMock(spec=torch.optim.Adam)):
            tp.post_init({}, TestTorchnn.FakeTrainingArgs())
        tp._dry_run = False
        tp.training_data_loader = MagicMock(spec=DataLoader(MagicMock(spec=Dataset)), batch_size=1)
        gen_data = TestTorchNNTrainingRoutineDataloaderTypes.iterate_once(
            ({'key': torch.Tensor([0])}, {'key': torch.Tensor([1])}))
        tp.training_data_loader.__iter__.return_value = gen_data
        tp.training_data_loader.__len__.return_value = 1
        tp.training_data_loader.batch_size = 1
        tp.training_step = MagicMock(return_value=torch.Tensor([0.]))

        class FakeDPController:
            def before_training(self, model, optimizer, loader):
                return tp._model, tp._optimizer, tp.training_data_loader
        tp._dp_controller = FakeDPController()

        tp.training_routine()
        tp.training_step.assert_called_once_with({'key': torch.Tensor([0])}, {'key': torch.Tensor([1])})
        patch_tensor_backward.assert_called_once()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
