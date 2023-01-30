import copy
import itertools
import types
import unittest
import os
import logging
import re

import torch
import torch.nn as nn
from torch.autograd import Variable

from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from torch.nn import Module
from torch.optim.lr_scheduler import LambdaLR
from testsupport.base_fake_training_plan import BaseFakeTrainingPlan
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.training_plans import TorchTrainingPlan, BaseTrainingPlan
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
            return {"dry_run": True, "epochs": 1, "batch_size": 10, "log_interval": 10}

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
    def test_torch_training_plan_02_post_init(self, conf_optimizer_model, conf_deps):

        conf_optimizer_model.return_value = None
        conf_deps.return_value = None

        tp = TorchTrainingPlan()
        tp._model = Module()
        tp.post_init({}, TestTorchnn.FakeTrainingArgs())

        self.assertEqual(tp._log_interval, 10)
        self.assertEqual(tp._epochs, 1)
        self.assertEqual(tp._dry_run, True)

        conf_optimizer_model.assert_called_once()
        conf_deps.assert_called_once()

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
    def test_torch_nn_03_testing_routine(self,
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

    def test_torch_nn_04_logging_progress_computation(self):
        """Test logging bug #313

        Create a DataLoader within a TrainingPlan with the following characteristics:
        - batch size = 5
        - total num samples = 15 (5*3 = batch_size * num_batches)
        - therefore, 3 batches will be processed

        The expected behaviour is that the first iteration should report a progress of 5/15 (33%),
        while the second iteration should report a progress of 10/15 (66%). Last iteration should report
        15/15 (100%). Only one epoch should be completed.
        """
        tp = TorchTrainingPlan()
        tp._optimizer = MagicMock(sepc=torch.optim.SGD)
        tp._model = torch.nn.Module()
        num_batches = 3
        batch_size = 5
        mock_dataset = MagicMock(spec=Dataset)
        
        tp.training_data_loader = MagicMock(spec=DataLoader(mock_dataset), batch_size=batch_size)
        tp._training_args = {'batch_size': batch_size,
                             'optimizer_args': {},
                             'epochs': 1,
                             'log_interval': 1,
                             'batch_maxnum': None,
                             'num_updates': None}
        mocked_loss_result = MagicMock(spec=torch.Tensor, return_value=torch.Tensor([0.]))
        mocked_loss_result.item.return_value = 0.
        tp.training_step = lambda x, y: mocked_loss_result

        custom_dataset = self.CustomDataset()
        x_train = torch.Tensor(custom_dataset.X_train[:batch_size])
        y_train = torch.Tensor(custom_dataset.Y_train[:batch_size])
        
        dataset_size = num_batches * batch_size
        fake_data = {'modality1': x_train, 'modality2': x_train}
        fake_target = (y_train, y_train)
        tp.training_data_loader.__iter__.return_value = num_batches*[(fake_data, fake_target)]
        tp.training_data_loader.__len__.return_value = num_batches
        tp.training_data_loader.dataset.__len__.return_value = dataset_size
        tp._num_updates = num_batches

        tp._dp_controller = FakeDPController()

        with self.assertLogs('fedbiomed', logging.DEBUG) as captured:

            num_samples_observed = tp.training_routine()
            self.assertEqual(num_samples_observed, num_batches * batch_size)
            training_progress_messages = [x for x in captured.output if re.search('Train Epoch: 1', x)]
            self.assertEqual(len(training_progress_messages), num_batches)  # Double-check correct number of train iters
            for i, logging_message in enumerate(training_progress_messages):
                logged_num_processed_samples = int(logging_message.split('Samples')[1].split('/')[0])
                logged_total_num_samples = int(logging_message.split('Samples')[1].split('/')[1].split()[0])
                logged_percent_progress = float(logging_message.split('(')[1].split('%')[0])
                self.assertEqual(logged_num_processed_samples, min((i+1)*batch_size, dataset_size))
                self.assertEqual(logged_total_num_samples, dataset_size)
                self.assertEqual(logged_percent_progress, round(100*(i+1)/num_batches))

    def test_torchnn_05_num_updates(self):
        """Test that num_updates parameter is respected correctly.
e
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
        
        tp._dp_controller = FakeDPController()

        def setup_tp(tp, num_samples, batch_size, num_updates):
            """Utility function to prepare the TrainingPlan test"""
            tp._optimizer.step.reset_mock()
            num_batches_per_epoch = num_samples // batch_size
            tp.training_data_loader = MagicMock(spec=DataLoader(MagicMock(spec=Dataset)),
                                                dataset=[1,2],
                                                batch_size=batch_size)
        
            tp.training_data_loader.__iter__.return_value = list(itertools.repeat(
                (MagicMock(spec=torch.Tensor), MagicMock(spec=torch.Tensor)), num_batches_per_epoch))
            tp.training_data_loader.__len__.return_value = num_batches_per_epoch
            tp._training_args = {'batch_size': batch_size,
                                 'batch_maxnum': None,
                                 'num_updates': num_updates,
                                 'log_interval': 10,
                                 'dry_run': False,
                                 'epochs': None}
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

    def test_torch_nn_06_compute_corrected_loss(self):
        """test_torch_nn_06_compute_corrected_loss: 
        checks:
            that fedavg and scaffold are equivalent if correction states are set to 0
        """
        def set_training_plan(model, aggregator_name:str, loss_value: float = .0):
            """Configure a TorchTrainingPlan with a given model.
            
            Args:
                model: a torch model
                aggregator_name: name of the aggregator method
                loss_value: value that is returned by mocked `training_Step` method
            """
            tp = TorchTrainingPlan()
            tp._set_device = MagicMock()

            tp._model = copy.deepcopy(model)
            tp._log_interval = 1
            tp.training_data_loader = MagicMock()
            tp._log_interval = 1000  # essentially disable logging
            tp._dry_run = False
            
            tp.aggregator_name = aggregator_name
            if aggregator_name == 'scaffold':
                for name, param in tp._model.named_parameters():
                    tp.correction_state[name] = torch.zeros_like(param)

            def training_step(instance, data, target):
                return torch.sum(instance.model().forward(data['modality1']))

            tp.training_step = types.MethodType(training_step, tp)

            custom_dataset = self.CustomDataset()
            x_train = torch.Tensor(custom_dataset.X_train)
            y_train = torch.Tensor(custom_dataset.Y_train)
            num_batches = 1
            batch_size = 5
            dataset_size = num_batches * batch_size
            fake_data = {'modality1': x_train}
            fake_target = (y_train, y_train)
            tp.training_data_loader.__iter__.return_value = num_batches*[(fake_data, fake_target)]
            tp.training_data_loader.__len__.return_value = num_batches
            tp.training_data_loader.batch_size = batch_size
            tp.training_data_loader.dataset.__len__.return_value = dataset_size
            tp._num_updates = num_batches
            tp._training_args = {'batch_size': batch_size}
            
            tp._optimizer_args = {"lr" : 1e-3}
            tp._optimizer = torch.optim.Adam(tp._model.parameters(), **tp._optimizer_args)
            tp._dp_controller = FakeDPController()
            tp._training_args = {'batch_size': batch_size,
                                 'optimizer_args': tp._optimizer_args,
                                 'epochs': 1,
                                 'log_interval': 10,
                                 'batch_maxnum': None,
                                 'dry_run': False,
                                 'num_updates': None}
            return tp
        
        model = torch.nn.Linear(3, 1)
        tp_fedavg = set_training_plan(model, "fedavg", .1)
        tp_fedavg.training_routine(None, None)
        
        tp_scaffold = set_training_plan(model, "scaffold", .1)
        
        tp_scaffold.training_routine(None, None)
        
        # test that model trained with scaffold is equivalent to model trained with fedavg
        for (name, layer_fedavg), (name, layer_scaffold) in zip(tp_fedavg._model.state_dict().items(),
                                                                tp_scaffold._model.state_dict().items()):
            self.assertTrue(torch.isclose(layer_fedavg, layer_scaffold).all())

    def test_torch_nn_07_get_learning_rate(self):
        """test_torch_nn_08_get_learning_rate: test we retrieve the appropriate 
        learning rate
        """
        # first test wih basic optimizer (eg without learning rate scheduler)
        tp = TorchTrainingPlan()
        tp._model = torch.nn.Linear(2, 3)
        lr = .1
        dataset = torch.Tensor([[1, 2], [1, 1], [2, 2]])
        target = torch.Tensor([1, 2, 2])
        tp._optimizer = SGD(tp._model.parameters(), lr=lr)
        
        lr_extracted = tp.get_learning_rate()
        self.assertListEqual(lr_extracted, [lr])
        
        # last test using a pytorch scheduler
        scheduler = LambdaLR(tp._optimizer, lambda e: 2*e)
        # this pytorch scheduler increase earning rate by twice its previous value
        for e, (x,y) in enumerate(zip(dataset, target)):
            # training a simple model in pytorch fashion
            # `e` represents epoch
            out = tp._model.forward(x)
            tp._optimizer.zero_grad()
            loss = torch.mean(out) - y
            loss.backward()
            tp._optimizer.step()
            scheduler.step()
            
            # checks
            lr_extracted = tp.get_learning_rate()
            self.assertListEqual(lr_extracted, [lr * 2 * (e+1)])


class TestSendToDevice(unittest.TestCase):

    def setUp(self) -> None:
        self.patcher = patch.multiple(TorchTrainingPlan, __abstractmethods__=set())
        self.patcher.start()
        self.cuda = torch.device('cuda')
        self.cpu = torch.device('cpu')

    def tearDown(self) -> None:
        self.patcher.stop()

    @patch('torch.Tensor.to')
    def test_send_tensor_to_device(self, patch_tensor_to):
        """Test basic case of sending a tensor to cpu and gpu."""
        tp = TorchTrainingPlan()
        t = torch.Tensor([0])
        t = tp.send_to_device(t, self.cpu)
        patch_tensor_to.assert_called_once()
        t = torch.Tensor([0])
        t = tp.send_to_device(t, self.cuda)
        self.assertEqual(patch_tensor_to.call_count, 2)

    def test_nested_collections(self):
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

    def test_unsupported_parameters(self):
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
    def test_data_loader_returns_tensors(self, patch_tensor_backward):
        batch_size = 1
        tp = TorchTrainingPlan()
        tp._model = torch.nn.Module()
        tp._optimizer = MagicMock(spec=torch.optim.Adam)
        tp._training_args = {'batch_size': batch_size, 'epochs': None, 'batch_maxnum': None,
                             'num_updates': 1, 'log_interval': 100, 'dry_run': False}

        tp.training_data_loader = MagicMock(spec=DataLoader(MagicMock(spec=Dataset)), batch_size=2, dataset=[1, 2])
        gen_load_data_as_tuples = TestTorchNNTrainingRoutineDataloaderTypes.iterate_once(
            (torch.Tensor([0]), torch.Tensor([1])))
        tp.training_data_loader.__len__.return_value = 2
        tp.training_data_loader.__iter__.return_value = gen_load_data_as_tuples

        tp.training_step = MagicMock(return_value=torch.Tensor([0.]))

        class FakeDPController:
            def before_training(self, model, optimizer, loader):
                return tp._model, tp._optimizer, tp.training_data_loader
        tp._dp_controller = FakeDPController()

        num_samples_observed = tp.training_routine()
        self.assertEqual(num_samples_observed, tp._training_args["num_updates"] * batch_size)
        tp.training_step.assert_called_once_with(torch.Tensor([0]), torch.Tensor([1]))
        patch_tensor_backward.assert_called_once()

    @patch('torch.Tensor.backward')
    def test_data_loader_returns_tuples(self, patch_tensor_backward):
        batch_size = 1
        tp = TorchTrainingPlan()
        tp._model = torch.nn.Module()
        tp._optimizer = MagicMock(spec=torch.optim.Adam)
        tp._training_args = {'batch_size': batch_size, 'epochs': None, 'batch_maxnum': None,
                             'num_updates': 1, 'log_interval': 100, 'dry_run': False}

        mock_dataset = MagicMock(spec=Dataset())
        tp.training_data_loader = MagicMock(spec=DataLoader(mock_dataset), batch_size=3)
        gen_load_data_as_tuples = TestTorchNNTrainingRoutineDataloaderTypes.iterate_once(
            ((torch.Tensor([0]), torch.Tensor([1])), torch.Tensor([2])))

        tp.training_data_loader.__len__.return_value = 3
        tp.training_data_loader.__iter__.return_value = gen_load_data_as_tuples
        tp.training_data_loader.dataset.__len__.return_value = 1
        tp.training_step = MagicMock(return_value=torch.Tensor([0.]))

        class FakeDPController:
            def before_training(self, model, optimizer, loader):
                return tp._model, tp._optimizer, tp.training_data_loader
        tp._dp_controller = FakeDPController()

        num_samples_observed = tp.training_routine()
        self.assertEqual(num_samples_observed, tp._training_args["num_updates"] * batch_size)
        tp.training_step.assert_called_once_with((torch.Tensor([0]), torch.Tensor([1])), torch.Tensor([2]))
        patch_tensor_backward.assert_called_once()

    @patch('torch.Tensor.backward')
    def test_data_loader_returns_dicts(self, patch_tensor_backward):
        batch_size = 1
        tp = TorchTrainingPlan()
        tp._model = torch.nn.Module()
        tp._optimizer = MagicMock(spec=torch.optim.Adam)
        tp._training_args = {'batch_size': batch_size, 'epochs': None, 'batch_maxnum': None,
                             'num_updates': 1, 'log_interval': 100, 'dry_run': False}

        # Set training data loader
        mock_dataset = MagicMock(spec=Dataset())
        tp.training_data_loader = MagicMock( spec=DataLoader(mock_dataset),
                                             batch_size=batch_size,
                                             dataset=[1,2]
                                            )
        gen_load_data_as_tuples = TestTorchNNTrainingRoutineDataloaderTypes.iterate_once(
                                                ({'key': torch.Tensor([0])}, {'key': torch.Tensor([1])})
                                    )
        tp.training_data_loader.__len__.return_value = 1
        tp.training_data_loader.__iter__.return_value = gen_load_data_as_tuples

        class FakeDPController:
            def before_training(self, model, optimizer, loader):
                return tp._model, tp._optimizer, tp.training_data_loader
        tp._dp_controller = FakeDPController()

        tp.training_step = MagicMock(return_value=torch.Tensor([0.]))
        num_samples_observed = tp.training_routine()
        self.assertEqual(num_samples_observed, tp._training_args["num_updates"] * batch_size)
        tp.training_step.assert_called_once_with({'key': torch.Tensor([0])}, {'key': torch.Tensor([1])})
        patch_tensor_backward.assert_called_once()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
