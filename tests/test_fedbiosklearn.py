import os
import tempfile
import unittest
import numpy as np

from unittest.mock import MagicMock, patch
from sklearn.linear_model import SGDRegressor

from fedbiomed.common.training_plans import SGDSkLearnModel
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.metrics import MetricTypes


class TrainingPlan(SGDSkLearnModel):

    def __init__(self, model_args):
        super(TrainingPlan, self).__init__(model_args)


class TrainingPlanTestingStep(SGDSkLearnModel):

    def __init__(self, model_args):
        super(TrainingPlanTestingStep, self).__init__(model_args)

    def testing_step(self, data, target): # noqa
        return {'Metric': 12}


class TestFedbiosklearn(unittest.TestCase):

    def setUp(self):
        self.reg_model = SGDRegressor(max_iter=1000, tol=1e-3)  # seems unused

    def tearDown(self):
        pass

    def test_init(self):
        kw = {'toto': 'le', 'lelo': 'la', 'max_iter': 7000, 'tol': 0.3456, 'n_features': 10, 'model': 'SGDRegressor'}
        fbsk = SGDSkLearnModel(kw)
        m = fbsk.get_model()
        p = m.get_params()
        self.assertEqual(p['max_iter'], 7000)
        self.assertEqual(p['tol'], 0.3456)
        self.assertTrue(np.allclose(m.coef_, np.zeros(10)))
        self.assertIsNone(p.get('lelo'))
        self.assertIsNone(p.get('toto'))
        self.assertIsNone(p.get('model'))

    def test_save_and_load(self):
        randomfile = tempfile.NamedTemporaryFile()

        skm = SGDSkLearnModel({'max_iter': 1000, 'tol': 1e-3, 'n_features': 5, 'model': 'SGDRegressor'})
        skm.save(randomfile.name)

        self.assertTrue(os.path.exists(randomfile.name) and os.path.getsize(randomfile.name) > 0)

        m = skm.load(randomfile.name)

        self.assertEqual(m.max_iter, 1000)
        self.assertEqual(m.tol, 0.001)

    def test_fedbiosklearn_02_testing_routine_regression(self):
        """ Testing `testing_routine` of SKLearnModel training plan"""

        history_monitor = MagicMock()
        history_monitor.add_scalar = MagicMock(return_value=None)

        # Dataset
        test_x = np.array([[1, 1], [1, 1], [1, 1]])
        test_y = np.array([1, 1, 1])

        model_args = {'model': 'SGDRegressor', 'n_features': 2}
        tp = TrainingPlan(model_args=model_args)

        # Test testing routine without setting testing_data_loader
        with self.assertRaises(FedbiomedTrainingPlanError):
            tp.testing_routine(metric=None,
                               metric_args={},
                               history_monitor=history_monitor,
                               before_train=True)

        # Test when metric is None should `ACCURACY` by default ---------------------------------------------------
        tp.set_data_loaders(train_data_loader=(test_x, test_y), test_data_loader=(test_x, test_y))
        tp.testing_routine(metric=None,
                           metric_args={},
                           history_monitor=history_monitor,
                           before_train=True)

        history_monitor.add_scalar.assert_called_once_with(metric={'ACCURACY': 0.0},
                                                           iteration=1,
                                                           epoch=None,
                                                           test=True,
                                                           test_on_local_updates=False,
                                                           test_on_global_updates=True,
                                                           total_samples=3,
                                                           batch_samples=3,
                                                           num_batches=1)
        history_monitor.add_scalar.reset_mock()

        # Test with mean square error ----------------------------------------------------------------------------
        tp.testing_routine(metric=MetricTypes.MEAN_SQUARE_ERROR,
                           metric_args={},
                           history_monitor=history_monitor,
                           before_train=True)
        history_monitor.add_scalar.assert_called_once_with(metric={'MEAN_SQUARE_ERROR': 1.0},
                                                           iteration=1,
                                                           epoch=None,
                                                           test=True,
                                                           test_on_local_updates=False,
                                                           test_on_global_updates=True,
                                                           total_samples=3,
                                                           batch_samples=3,
                                                           num_batches=1)
        history_monitor.add_scalar.reset_mock()

        # Test if predict raises error
        with patch.object(SGDRegressor, 'predict') as patch_predict:
            patch_predict.side_effect = Exception
            with self.assertRaises(FedbiomedTrainingPlanError):
                tp.testing_routine(metric=MetricTypes.MEAN_SQUARE_ERROR,
                                   metric_args={},
                                   history_monitor=history_monitor,
                                   before_train=True)

    def test_fedbiosklearn_03_testing_routine_classification(self):
        """ Test testing_routine when the model is classification model"""
        history_monitor = MagicMock()
        history_monitor.add_scalar = MagicMock(return_value=None)

        # Dataset
        test_x = np.array([[1, 1], [2, 2], [1, 1]])
        test_y = np.array([2, 2, 2])

        model_args = {'model': 'SGDClassifier', 'n_features': 2, 'n_classes': 2}
        tp = TrainingPlan(model_args=model_args)
        tp.set_data_loaders(train_data_loader=(test_x, test_y), test_data_loader=(test_x, test_y))

        # Test with metric `ACCURACY` ----------------------------------------------------------------------------
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
                                                           total_samples=3,
                                                           batch_samples=3,
                                                           num_batches=1)

        # check if `classes_` attribute of classifier has been created
        self.assertTrue(hasattr(tp.model, 'classes_'))
        self.assertTrue(np.array_equal(tp.model.classes_, np.array([2])))
        history_monitor.add_scalar.reset_mock()

        # Test with regression metric
        tp.testing_routine(metric=MetricTypes.MEAN_SQUARE_ERROR,
                           metric_args={},
                           history_monitor=history_monitor,
                           before_train=True)

        history_monitor.add_scalar.assert_called_once_with(metric={'MEAN_SQUARE_ERROR': 0.0},
                                                           iteration=1,
                                                           epoch=None,
                                                           test=True,
                                                           test_on_local_updates=False,
                                                           test_on_global_updates=True,
                                                           total_samples=3,
                                                           batch_samples=3,
                                                           num_batches=1)
        history_monitor.add_scalar.reset_mock()

        # Testing with `testing_step` defined by user -------------------------------------------------------
        model_args = {'model': 'SGDClassifier', 'n_features': 2, 'n_classes': 2}
        tp = TrainingPlanTestingStep(model_args=model_args)
        tp.set_data_loaders(train_data_loader=(test_x, test_y), test_data_loader=(test_x, test_y))
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
                                                           total_samples=3,
                                                           batch_samples=3,
                                                           num_batches=1)


        # If testing step raises an exception ---------------------------------------------------------------
        with patch.object(TrainingPlanTestingStep, 'testing_step') as patch_testing_step:
            patch_testing_step.side_effect = Exception
            with self.assertRaises(FedbiomedTrainingPlanError):
                tp.testing_routine(metric=MetricTypes.ACCURACY,
                                   metric_args={},
                                   history_monitor=history_monitor,
                                   before_train=True)
            # Test if metric value returns None
            patch_testing_step.side_effect = None
            patch_testing_step.return_value = None
            with self.assertRaises(FedbiomedTrainingPlanError):
                tp.testing_routine(metric=MetricTypes.ACCURACY,
                                   metric_args={},
                                   history_monitor=history_monitor,
                                   before_train=True)


        # Test `testig_step` raises an error if test_ratio equals 0
        with self.assertRaises(FedbiomedTrainingPlanError):
            tp.set_data_loaders(train_data_loader=(test_x, test_y), test_data_loader=None)
            tp.testing_routine(metric=MetricTypes.ACCURACY,
                               metric_args={},
                               history_monitor=history_monitor,
                               before_train=False)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
