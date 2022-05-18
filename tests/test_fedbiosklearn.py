import os
import tempfile
import unittest
import numpy as np
from copy import deepcopy

from unittest.mock import MagicMock, patch
from sklearn.linear_model import SGDRegressor

from fedbiomed.common.training_plans import SGDSkLearnModel
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.training_plans import FedPerceptron, FedSGDRegressor, FedSGDClassifier


class TrainingPlan(SGDSkLearnModel):

    def __init__(self, model_args):
        super(TrainingPlan, self).__init__(model_args)


class TrainingPlanTestingStep(SGDSkLearnModel):

    def __init__(self, model_args):
        super(TrainingPlanTestingStep, self).__init__(model_args)

    def testing_step(self, data, target): # noqa
        return {'Metric': 12}

class TestSklearnModelsCorrectlyInheritFunctionalities(unittest.TestCase):
    implemented_models = [FedPerceptron, FedSGDRegressor, FedSGDClassifier]
    model_args = {
        FedPerceptron: {'max_iter': 4242, 'alpha': 0.999, 'n_classes': 2, 'n_features': 2, 'key_not_in_model': None},
        FedSGDRegressor: {'max_iter': 4242, 'alpha': 0.999, 'n_features': 2, 'key_not_in_model': None},
        FedSGDClassifier: {'max_iter': 4242, 'alpha': 0.999, 'n_classes': 2, 'n_features': 2, 'key_not_in_model': None},
    }
    expected_params_list = {
        FedPerceptron: ['intercept_', 'coef_'],
        FedSGDRegressor: ['intercept_', 'coef_'],
        FedSGDClassifier: ['intercept_', 'coef_']
    }

    def setUp(self):
        # create subclasses for each implemented class, and append TrainingPlan
        # to the name, e.g. FedPerceptronTrainingPlan
        # We do this to replicate the intended use where the researcher inherits from our class,
        # e.g. FedPerceptron, to create their own Training Plan
        self.subclass_types = dict()
        self.training_plans = list()
        for sklearn_model_type in TestSklearnModelsCorrectlyInheritFunctionalities.implemented_models:
            new_subclass_type = type(sklearn_model_type.__name__ + 'TrainingPlan',
                                     (sklearn_model_type,),
                                     {'parent_type': sklearn_model_type})
            self.subclass_types[sklearn_model_type] = new_subclass_type
            self.training_plans.append(new_subclass_type(TestSklearnModelsCorrectlyInheritFunctionalities.model_args[sklearn_model_type]))

    def tearDown(self):
        pass

    def test_model_args(self):
        for training_plan in self.training_plans:
            # ensure that the models args passed by the researcher are correctly stored in the class
            self.assertDictEqual(training_plan.model_args,
                                 TestSklearnModelsCorrectlyInheritFunctionalities.model_args[training_plan.parent_type])
            for key, val in training_plan.model.get_params().items():
                # ensure that the model args passed by the researcher are correctly passed to the sklearn model
                if key in TestSklearnModelsCorrectlyInheritFunctionalities.model_args[training_plan.parent_type]:
                    self.assertEqual(val, TestSklearnModelsCorrectlyInheritFunctionalities.model_args[training_plan.parent_type][key])
            # ensure that invalid keys from researcher's model args are not passed to the sklearn model
            self.assertNotIn('key_not_in_model', training_plan.model.get_params())

    def test_save_and_load(self):
        for training_plan in self.training_plans:
            randomfile = tempfile.NamedTemporaryFile()
            training_plan.save(randomfile.name)
            orig_params = deepcopy(training_plan.model.get_params())

            # ensure file has been created and has size > 0
            self.assertTrue(os.path.exists(randomfile.name) and os.path.getsize(randomfile.name) > 0)

            new_tp = self.subclass_types[training_plan.parent_type](
                    model_args={'n_classes': 2, 'n_features': 1})  # empty model_args does not work, sadly

            m = new_tp.load(randomfile.name)
            # ensure output of load is the same as original parameters
            self.assertDictEqual(m.get_params(), orig_params)
            # ensure that the newly loaded model has the same params as the original model
            self.assertDictEqual(training_plan.model.get_params(), new_tp.model.get_params())

    def test_fedbiosklearn_02_testing_routine_regression(self):
        """ Testing `testing_routine` of SKLearnModel training plan"""

        history_monitor = MagicMock()
        history_monitor.add_scalar = MagicMock(return_value=None)

        # Dataset
        test_x = np.array([[1, 1], [1, 1], [1, 1], [1,1]])
        test_y = np.array([1, 0, 1, 0])

        for training_plan in self.training_plans:
            # Test testing routine without setting testing_data_loader
            with self.assertRaises(FedbiomedTrainingPlanError):
                training_plan.testing_routine(metric=None,
                                   metric_args={},
                                   history_monitor=history_monitor,
                                   before_train=True)

            # Test when metric is None should `ACCURACY` by default ---------------------------------------------------
            training_plan.set_data_loaders(train_data_loader=(test_x, test_y), test_data_loader=(test_x, test_y))
            if not training_plan._is_regression:
                training_plan.testing_routine(metric=None,
                                   metric_args={},
                                   history_monitor=history_monitor,
                                   before_train=True)

                history_monitor.add_scalar.assert_called_once_with(metric={'ACCURACY': 0.5},
                                                                   iteration=1,
                                                                   epoch=None,
                                                                   test=True,
                                                                   test_on_local_updates=False,
                                                                   test_on_global_updates=True,
                                                                   total_samples=4,
                                                                   batch_samples=4,
                                                                   num_batches=1)
                history_monitor.add_scalar.reset_mock()

            # Test with mean square error ----------------------------------------------------------------------------
            training_plan.testing_routine(metric=MetricTypes.MEAN_SQUARE_ERROR,
                               metric_args={},
                               history_monitor=history_monitor,
                               before_train=True)
            history_monitor.add_scalar.assert_called_once_with(metric={'MEAN_SQUARE_ERROR': 0.5},
                                                               iteration=1,
                                                               epoch=None,
                                                               test=True,
                                                               test_on_local_updates=False,
                                                               test_on_global_updates=True,
                                                               total_samples=4,
                                                               batch_samples=4,
                                                               num_batches=1)
            history_monitor.add_scalar.reset_mock()

        # Test if predict raises error
#        with patch.object(SGDRegressor, 'predict') as patch_predict:
#            patch_predict.side_effect = Exception
#            with self.assertRaises(FedbiomedTrainingPlanError):
#                tp.testing_routine(metric=MetricTypes.MEAN_SQUARE_ERROR,
#                                   metric_args={},
#                                   history_monitor=history_monitor,
#                                   before_train=True)

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
