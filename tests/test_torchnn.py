import unittest
import os

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
        tp2.load(paramfile)
        self.assertTrue(type(params2) is dict)

        sd1 = tp1.state_dict()
        sd2 = tp2.state_dict()

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

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
