import copy
import itertools
import unittest
import os
import logging
import re

import torch
import torch.nn as nn

from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, Dataset

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


class TrainingPlanWithTestingStep(TorchTrainingPlan):

    def __init__(self):
        super(TrainingPlanWithTestingStep, self).__init__()

    def testing_step(self, data, target): # noqa
        return {'Metric': 12}


class TestTorchnn(unittest.TestCase):
    """
    Test the Torchnn class
    """

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
        self.TrainingPlan = TrainingPlan
        self.params = {'one': 1, '2': 'two'}
        self.tmpdir = '.'

    # after the tests
    def tearDown(self):
        pass

    #
    # TODO : add tests for checking the training payload
    #

    def test_save_load_model(self):

        tp1 = self.TrainingPlan()
        self.assertIsNotNone(tp1.test_method)
        self.assertTrue(tp1.test_method())

        modulename = 'tmp_model'
        codefile = self.tmpdir + os.path.sep + modulename + '.py'
        try:
            os.remove(codefile)
        except FileNotFoundError:
            pass

        tp1.save_code(codefile)
        self.assertTrue(os.path.isfile(codefile))

        # would expect commented lines to be necessary
        #
        # sys.path.insert(0, self.tmpdir)
        # exec('import ' + modulename, globals())
        exec('import ' + modulename)
        # sys.path.pop(0)
        TrainingPlan2 = eval(modulename + '.' + self.TrainingPlan.__name__)
        tp2 = TrainingPlan2()

        self.assertIsNotNone(tp2.test_method)
        self.assertTrue(tp2.test_method())

        os.remove(codefile)

    def test_save_load_params(self):
        tp1 = TrainingPlan()
        tp1._model = torch.nn.Module()
        paramfile = self.tmpdir + '/tmp_params.pt'
        try:
            os.remove(paramfile)
        except FileNotFoundError:
            pass

        # save/load from/to variable
        tp1.save(paramfile, self.params)
        self.assertTrue(os.path.isfile(paramfile))
        params2 = tp1.load(paramfile, True)

        self.assertTrue(type(params2) is dict)
        self.assertEqual(self.params, params2)

        # save/load from/to object params
        tp1.save(paramfile)
        tp2 = TrainingPlan()
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
        - batch size = 3
        - total num samples = 5
        - therefore, 2 batches will be processed

        The expected behaviour is that the first iteration should report a progress of 3/5 (60%),
        while the second iteration should report a progress of 5/5 (100%).
        """
        tp = TorchTrainingPlan()
        tp._optimizer = MagicMock()
        tp._model = torch.nn.Module()
        tp._log_interval = 1
        tp.training_data_loader = MagicMock()

        mocked_loss_result = MagicMock()
        mocked_loss_result.item.return_value = 0.
        tp.training_step = lambda x, y: mocked_loss_result

        custom_dataset = self.CustomDataset()
        x_train = torch.Tensor(custom_dataset.X_train)
        y_train = torch.Tensor(custom_dataset.Y_train)
        num_batches = 3
        batch_size = 5
        dataset_size = num_batches * batch_size
        fake_data = {'modality1': x_train, 'modality2': x_train}
        fake_target = (y_train, y_train)
        tp.training_data_loader.__iter__.return_value = num_batches*[(fake_data, fake_target)]
        tp.training_data_loader.__len__.return_value = num_batches
        tp.training_data_loader.batch_size = batch_size
        tp.training_data_loader.dataset.__len__.return_value = dataset_size
        tp._num_updates = num_batches

        with self.assertLogs('fedbiomed', logging.DEBUG) as captured:
            tp.training_routine()
            training_progress_messages = [x for x in captured.output if re.search('Train Epoch: 1', x)]
            self.assertEqual(len(training_progress_messages), num_batches)  # Double-check correct number of train iters
            for i, logging_message in enumerate(training_progress_messages):
                logged_num_processed_samples = int(logging_message.split('[')[1].split('/')[0])
                logged_total_num_samples = int(logging_message.split('/')[1].split()[0])
                logged_percent_progress = float(logging_message.split('(')[1].split('%')[0])
                self.assertEqual(logged_num_processed_samples, min((i+1)*batch_size, dataset_size))
                self.assertEqual(logged_total_num_samples, dataset_size)
                self.assertEqual(logged_percent_progress, round(100*(i+1)/num_batches))

        # TODO: do the same for `num_updates`
        
    def test_torchnn_05_num_updates(self):
        """Test that num_updates parameter is respected correctly.

        In the following test, we make sure that no matter the dataset size, nor the batch size, we always perform the
        number of updates requested by the researcher. Remember each update corresponds to one optimizer step, i.e.
        one batch.
        """
        tp = TrainingPlan()
        tp._model = MagicMock()
        tp._set_device = MagicMock()
        tp._batch_maxnum = 0
        tp._optimizer = MagicMock()
        tp._optimizer.step = MagicMock()
        tp.training_step = MagicMock(return_value=MagicMock())
        tp._log_interval = 1000  # essentially disable logging
        tp._dry_run = False

        def setup_tp(tp, num_samples, batch_size, num_updates):
            """Utility function to prepare the TrainingPlan test"""
            tp._optimizer.step.reset_mock()
            num_batches_per_epoch = num_samples // batch_size
            tp.training_data_loader = list(itertools.repeat(
                (MagicMock(spec=torch.Tensor), MagicMock(spec=torch.Tensor)), num_batches_per_epoch))
            tp._num_updates = num_updates
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


    def test_torch_nn_06_compute_corrected_loss(self):
        
        def set_training_plan(model, aggregator_name:str, loss_value: float = .0):
            
            tp = TrainingPlan()
            tp._set_device = MagicMock()
            tp._batch_maxnum = 0
            
            tp._model = copy.deepcopy(model)
            tp._log_interval = 1
            tp.training_data_loader = MagicMock()
            tp._log_interval = 1000  # essentially disable logging
            tp._dry_run = False
            
            tp.aggregator_name = aggregator_name
            mocked_loss_result = MagicMock()
            mocked_loss_result.item.return_value = loss_value
            tp.training_step = lambda x, y: torch.mean(y[0])

            custom_dataset = self.CustomDataset()
            x_train = torch.Tensor(custom_dataset.X_train)
            y_train = torch.Tensor(custom_dataset.Y_train)
            num_batches = 10
            batch_size = 5
            dataset_size = num_batches * batch_size
            fake_data = {'modality1': x_train}
            fake_target = (y_train, y_train)
            tp.training_data_loader.__iter__.return_value = num_batches*[(fake_data, fake_target)]
            tp.training_data_loader.__len__.return_value = num_batches
            tp.training_data_loader.batch_size = batch_size
            tp.training_data_loader.dataset.__len__.return_value = dataset_size
            tp._num_updates = num_batches
            
            tp._optimizer_args = {"lr" : 1e-3}
            tp._optimizer = torch.optim.Adam(tp._model.parameters(), **tp._optimizer_args)
            return tp
        
        model = torch.nn.Linear(10, 3)
        tp_fedavg = set_training_plan(model, "fedavg", .1)
        tp_fedavg.training_routine(None, None)

        tp_scaffold = set_training_plan(model, "scaffold", .1)
        
        tp_scaffold.training_routine(None, None)
        
        for (name, layer_fedavg), (name, layer_scaffold) in zip(tp_fedavg._model.state_dict().items(),
                                                                tp_scaffold._model.state_dict().items()):
            self.assertTrue(torch.isclose(layer_fedavg, layer_scaffold).all())
        for (name, layer_fedavg), (name, layer_scaffold) in zip(model.state_dict().items(),
                                                                tp_scaffold._model.state_dict().items()):
            self.assertTrue(torch.isclose(layer_fedavg, layer_scaffold).all())
        correction_state = copy.deepcopy(model)
        
        for p in correction_state.parameters():
            p.data.fill_(1)
        tp_scaffold = set_training_plan(correction_state, "scaffold")
        tp_scaffold.correction_state = correction_state.state_dict()
        print(model.state_dict())
        tp_fedavg.training_routine(None, None)
        
        print(tp_scaffold._model.state_dict())
        
        # for (name, layer_fedavg), (name, layer_scaffold) in zip(tp_fedavg._model.state_dict().items(),
        #                                                         tp_scaffold._model.state_dict().items()):
        #     self.assertTrue(torch.isclose(layer_fedavg, layer_scaffold).all())
        
    def test_torch_nn_07_compute_corrected_loss_2(self):
        tp = TrainingPlan()
        model = torch.nn.Linear(10, 3)
        
        correction_state = copy.deepcopy(model)
    
        for p in correction_state.parameters():
            p.data.fill_(1)
        
        # test for Scaffold
        tp._model = correction_state
        tp.correction_state = correction_state.state_dict()
        tp.aggregator_name = 'scaffold'
        loss = torch.tensor(0.)
        val = tp.compute_corrected_loss(loss)
        self.assertEqual(val, loss - (10 + 1) * 3)
        
        # test for fedavg
        tp.aggregator_name = 'fedavg'
        val = tp.compute_corrected_loss(loss)
        self.assertEqual(loss, val)
        
        # TODO: test fedprox


class TestSendToDevice(unittest.TestCase):
    def setUp(self) -> None:
        self.cuda = torch.device('cuda')
        self.cpu = torch.device('cpu')

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

    @patch('torch.Tensor.backward')
    def test_data_loader_returns_tensors(self, patch_tensor_backward):
        tp = TorchTrainingPlan()
        tp._model = torch.nn.Module()
        tp._optimizer = MagicMock(spec=torch.optim.Adam)

        tp.training_data_loader = MagicMock(spec=torch.utils.data.Dataset, batch_size=2)
        gen_load_data_as_tuples = TestTorchNNTrainingRoutineDataloaderTypes.iterate_once(
            (torch.Tensor([0]), torch.Tensor([1])))
        tp.training_data_loader.__getitem__ = lambda _, idx: next(gen_load_data_as_tuples)
        tp.training_step = MagicMock(return_value=torch.Tensor([0.]))
        tp.training_routine()
        tp.training_step.assert_called_once_with(torch.Tensor([0]), torch.Tensor([1]))
        patch_tensor_backward.assert_called_once()

    @patch('torch.Tensor.backward')
    def test_data_loader_returns_tuples(self, patch_tensor_backward):
        tp = TorchTrainingPlan()
        tp._model = torch.nn.Module()
        tp._optimizer = MagicMock(spec=torch.optim.Adam)
        tp.training_data_loader = MagicMock(spec=torch.utils.data.Dataset, batch_size=3)
        gen_load_data_as_tuples = TestTorchNNTrainingRoutineDataloaderTypes.iterate_once(
            ((torch.Tensor([0]), torch.Tensor([1])), torch.Tensor([2])))
        tp.training_data_loader.__getitem__ = lambda _, idx: next(gen_load_data_as_tuples)
        tp.training_step = MagicMock(return_value=torch.Tensor([0.]))
        tp.training_routine()
        tp.training_step.assert_called_once_with((torch.Tensor([0]), torch.Tensor([1])), torch.Tensor([2]))
        patch_tensor_backward.assert_called_once()

    @patch('torch.Tensor.backward')
    def test_data_loader_returns_dicts(self, patch_tensor_backward):
        tp = TorchTrainingPlan()
        tp._model = torch.nn.Module()
        tp._optimizer = MagicMock(spec=torch.optim.Adam)
        tp.training_data_loader = MagicMock(spec=torch.utils.data.Dataset, batch_size=1)
        gen_load_data_as_tuples = TestTorchNNTrainingRoutineDataloaderTypes.iterate_once(
            ({'key': torch.Tensor([0])}, {'key': torch.Tensor([1])}))
        tp.training_data_loader.__getitem__ = lambda _, idx: next(gen_load_data_as_tuples)
        tp.training_step = MagicMock(return_value=torch.Tensor([0.]))
        tp.training_routine()
        tp.training_step.assert_called_once_with({'key': torch.Tensor([0])}, {'key': torch.Tensor([1])})
        patch_tensor_backward.assert_called_once()








if __name__ == '__main__':  # pragma: no cover
    unittest.main()
