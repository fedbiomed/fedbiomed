"""Testing of the whole class hierarchy of Sklearn training plans.

The strategy is to try to copy the expected researcher behaviour by dynamically creating subclasses of all the
support models in the _sklearn_models.py file.

The class `TestSklearnTrainingPlansCommonFunctionalities` tests behaviours that should be common to all models,
while other classes test specific behaviours.

To add a new sklearn model for testing, you should include its name in the `implemented_models` attribute of
`TestSklearnTrainingPlansCommonFunctionalities`, and possibly in other more specialized classes (or implement your own
specialized class).
"""

import os
import tempfile
import unittest
import logging
import numpy as np
from copy import deepcopy
from unittest.mock import MagicMock, patch

from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.training_plans import FedPerceptron, FedSGDRegressor, FedSGDClassifier


class Custom:
    def testing_step(mydata, mytarget):
        return {'Metric': 42.0}


class FakeTrainingArgs:

    def pure_training_arguments(self):
        return {"num_updates": 1}


class TestSklearnTrainingPlansCommonFunctionalities(unittest.TestCase):
    """Class that tests the generic functionalities of a sklearn training plan.

    Attributes:
        implemented_models: (list) the names of the classes being tested (from the _sklean_models module)
        model_args: (dict) a map between each class type and the corresponding dict of model_args to be used for
            initialization
        expected_params_list: (dict) a map between each class type and the expected value of the params_list attribute,
            i.e. the model parameters for aggregation
        training_plans: (list) instances of subclasses of each class in `implemented_models`
        subclass_types: (list) types of the instances in `training_plans`
    """
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
        """Prepare testing environment.

        Populates the attribute `training_plans`, which is a list containing an instance of a subclass for each
        implemented sklearn wrapper from the _sklearn_models.py file. This is made to copy the intended usage
        whereby the researcher would subclass the models implemented in _sklearn_models.py
        """
        self.subclass_types = dict()
        self.training_plans = list()
        for sklearn_model_type in TestSklearnTrainingPlansCommonFunctionalities.implemented_models:
            new_subclass_type = type(sklearn_model_type.__name__ + 'TrainingPlan',
                                     (sklearn_model_type,),
                                     {'parent_type': sklearn_model_type})
            self.subclass_types[sklearn_model_type] = new_subclass_type
            m = new_subclass_type()
            m.post_init(TestSklearnTrainingPlansCommonFunctionalities.model_args[sklearn_model_type],
                        FakeTrainingArgs())
            self.training_plans.append(m)

        logging.disable('CRITICAL')  # prevent flood of messages about missing datasets

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_model_args(self):
        for training_plan in self.training_plans:
            # ensure that the models args passed by the researcher are correctly stored in the class
            self.assertDictEqual(training_plan._model_args,
                                 TestSklearnTrainingPlansCommonFunctionalities.model_args[training_plan.parent_type])
            for key, val in training_plan.model().get_params().items():
                # ensure that the model args passed by the researcher are correctly passed to the sklearn model
                if key in TestSklearnTrainingPlansCommonFunctionalities.model_args[training_plan.parent_type]:
                    self.assertEqual(val, TestSklearnTrainingPlansCommonFunctionalities.model_args[training_plan.parent_type][key])
            # ensure that invalid keys from researcher's model args are not passed to the sklearn model
            self.assertNotIn('key_not_in_model', training_plan.model().get_params())

            # --------- Check that param_list is correctly populated
            # check that param_list is a list
            self.assertIsInstance(training_plan._param_list, list)
            # check that param_list is not empty
            self.assertTrue(training_plan._param_list)
            # check that param_list is a list of str
            for param in training_plan._param_list:
                self.assertIsInstance(param, str)

    def test_save_and_load(self):
        for training_plan in self.training_plans:
            randomfile = tempfile.NamedTemporaryFile()
            training_plan.save(randomfile.name)
            orig_params = deepcopy(training_plan.model().get_params())

            # ensure file has been created and has size > 0
            self.assertTrue(os.path.exists(randomfile.name) and os.path.getsize(randomfile.name) > 0)

            new_tp = self.subclass_types[training_plan.parent_type]()
            new_tp.post_init({'n_classes': 2, 'n_features': 1}, FakeTrainingArgs())

            m = new_tp.load(randomfile.name)
            # ensure output of load is the same as original parameters
            self.assertDictEqual(m.get_params(), orig_params)
            # ensure that the newly loaded model has the same params as the original model
            self.assertDictEqual(training_plan.model().get_params(), new_tp.model().get_params())

    def test_exceptions_are_correctly_converted(self):
        # Dataset
        test_x = np.array([[1, 1], [1, 1], [1, 1], [1,1]])
        test_y = np.array([1, 0, 1, 0])
        for training_plan in self.training_plans:
            training_plan.set_data_loaders(train_data_loader=(test_x, test_y), test_data_loader=(test_x, test_y))

            with patch.object(training_plan.model(), 'predict') as patch_predict:
                patch_predict.side_effect = Exception
                with self.assertRaises(FedbiomedTrainingPlanError):
                    training_plan.testing_routine(metric=MetricTypes.MEAN_SQUARE_ERROR,
                                                  metric_args={},
                                                  history_monitor=None,
                                                  before_train=True)

            training_plan.testing_step = lambda data, target: {'MyMetric': 0.}
            with patch.object(training_plan, 'testing_step') as patch_testing_step:
                patch_testing_step.side_effect = Exception
                with self.assertRaises(FedbiomedTrainingPlanError):
                    training_plan.testing_routine(metric=MetricTypes.ACCURACY,
                                                  metric_args={},
                                                  history_monitor=None,
                                                  before_train=True)

                # Ensure FedbiomedTrainingPlanError is raised when metric returns None
                patch_testing_step.side_effect = None
                with self.assertRaises(FedbiomedTrainingPlanError):
                    training_plan.testing_routine(metric=MetricTypes.ACCURACY,
                                                  metric_args={},
                                                  history_monitor=None,
                                                  before_train=True)

    def test_custom_testing_step(self):
        history_monitor = MagicMock()
        history_monitor.add_scalar = MagicMock(return_value=None)

        # Dataset
        test_x = np.array([[1, 1], [1, 1], [1, 1], [1,1]])
        test_y = np.array([1, 0, 1, 0])

        for training_plan in self.training_plans:
            # case where the researcher defines a custom testing step
            training_plan.testing_step = Custom.testing_step
            training_plan.set_data_loaders(train_data_loader=(test_x, test_y), test_data_loader=(test_x, test_y))

            # call testing routine again
            training_plan.testing_routine(metric=None,  # ignored when custom testing_step is defined
                                          metric_args={},
                                          history_monitor=history_monitor,
                                          before_train=True)

            # ensure that the history monitor was called with the correct parameters
            # note that this requires knowing the actual value of the mean square error metric
            # for this specific case. For now it works but we may have to relax the constraint
            # to something like assert_called_once() in the future if the models default to
            # different values.
            history_monitor.add_scalar.assert_called_once_with(metric={'Metric': 42.0},
                                                               iteration=1,
                                                               epoch=None,
                                                               test=True,
                                                               test_on_local_updates=False,
                                                               test_on_global_updates=True,
                                                               total_samples=4,
                                                               batch_samples=4,
                                                               num_batches=1)
            history_monitor.add_scalar.reset_mock()


class TestSklearnTrainingPlansRegression(unittest.TestCase):
    implemented_models = [FedSGDRegressor]
    model_args = {
        FedSGDRegressor: {'max_iter': 4242, 'alpha': 0.999, 'n_features': 2, 'key_not_in_model': None},
    }
    expected_params_list = {
        FedSGDRegressor: ['intercept_', 'coef_'],
    }

    def setUp(self):
        # create subclasses for each implemented class, and append TrainingPlan
        # to the name, e.g. FedPerceptronTrainingPlan
        # We do this to replicate the intended use where the researcher inherits from our class,
        # e.g. FedPerceptron, to create their own Training Plan
        self.subclass_types = dict()
        self.training_plans = list()
        for sklearn_model_type in TestSklearnTrainingPlansRegression.implemented_models:
            new_subclass_type = type(sklearn_model_type.__name__ + 'TrainingPlan',
                                     (sklearn_model_type,),
                                     {'parent_type': sklearn_model_type})
            self.subclass_types[sklearn_model_type] = new_subclass_type
            m = new_subclass_type()
            m.post_init(TestSklearnTrainingPlansRegression.model_args[sklearn_model_type], FakeTrainingArgs())
            self.training_plans.append(m)

        logging.disable('CRITICAL')  # prevent flood of messages about missing datasets

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_regression_parameters(self):
        for training_plan in self.training_plans:
            self.assertTrue(training_plan._is_regression)
            self.assertFalse(training_plan._is_classification)
            self.assertFalse(training_plan._is_clustering)

    def test_regression_testing_routine(self):
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

            # set data loader and call testing routine
            training_plan.set_data_loaders(train_data_loader=(test_x, test_y), test_data_loader=(test_x, test_y))
            training_plan.testing_routine(metric=MetricTypes.MEAN_SQUARE_ERROR,
                                          metric_args={},
                                          history_monitor=history_monitor,
                                          before_train=True)

            # ensure that the history monitor was called with the correct parameters
            # note that this requires knowing the actual value of the mean square error metric
            # for this specific case. For now it works but we may have to relax the constraint
            # to something like assert_called_once() in the future if the models default to
            # different values.
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


class TestSklearnTrainingPlansClassification(unittest.TestCase):
    implemented_models = [FedPerceptron, FedSGDClassifier]
    model_args = {
        FedPerceptron: {'max_iter': 4242, 'alpha': 0.999, 'n_classes': 2, 'n_features': 2, 'key_not_in_model': None},
        FedSGDClassifier: {'max_iter': 4242, 'alpha': 0.999, 'n_classes': 2, 'n_features': 2, 'key_not_in_model': None},
    }
    expected_params_list = {
        FedPerceptron: ['intercept_', 'coef_'],
        FedSGDClassifier: ['intercept_', 'coef_']
    }

    def setUp(self):
        # create subclasses for each implemented class, and append TrainingPlan
        # to the name, e.g. FedPerceptronTrainingPlan
        # We do this to replicate the intended use where the researcher inherits from our class,
        # e.g. FedPerceptron, to create their own Training Plan
        self.subclass_types = dict()
        self.training_plans = list()
        for sklearn_model_type in TestSklearnTrainingPlansClassification.implemented_models:
            new_subclass_type = type(sklearn_model_type.__name__ + 'TrainingPlan',
                                     (sklearn_model_type,),
                                     {'parent_type': sklearn_model_type})
            self.subclass_types[sklearn_model_type] = new_subclass_type
            m = new_subclass_type()
            m.post_init(TestSklearnTrainingPlansClassification.model_args[sklearn_model_type], FakeTrainingArgs())
            self.training_plans.append(m)

        logging.disable('CRITICAL')  # prevent flood of messages about missing datasets

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_classification_parameters(self):
        for training_plan in self.training_plans:
            self.assertFalse(training_plan._is_regression)
            self.assertTrue(training_plan._is_classification)
            self.assertFalse(training_plan._is_clustering)

    def test_classification_testing_routine(self):
        """ Testing `testing_routine` of SKLearnModel training plan"""
        history_monitor = MagicMock()
        history_monitor.add_scalar = MagicMock(return_value=None)

        # Dataset
        test_x = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        test_y = np.array([1, 0, 1, 0])

        for training_plan in self.training_plans:
            # Test testing routine without setting testing_data_loader
            with self.assertRaises(FedbiomedTrainingPlanError):
                training_plan.testing_routine(metric=None,
                                              metric_args={},
                                              history_monitor=history_monitor,
                                              before_train=True)

            training_plan.set_data_loaders(train_data_loader=(test_x, test_y), test_data_loader=(test_x, test_y))

            training_plan.testing_routine(metric=MetricTypes.ACCURACY,
                                          metric_args={},
                                          history_monitor=history_monitor,
                                          before_train=True)

            # ensure that the history monitor was called with the correct parameters
            # note that this requires knowing the actual value of the mean square error metric
            # for this specific case. For now it works but we may have to relax the constraint
            # to something like assert_called_once() in the future if the models default to
            # different values.
            history_monitor.add_scalar.assert_called_once_with(metric={'ACCURACY': 0.5},
                                                               iteration=1,
                                                               epoch=None,
                                                               test=True,
                                                               test_on_local_updates=False,
                                                               test_on_global_updates=True,
                                                               total_samples=4,
                                                               batch_samples=4,
                                                               num_batches=1)

            # check if `classes_` attribute of classifier has been created
            self.assertTrue(hasattr(training_plan.model(), 'classes_'), msg=training_plan.parent_type.__name__ + ' does not automatically create the classes_ attribute')
            self.assertTrue(np.array_equal(training_plan.model().classes_, np.array([0, 1])))

            history_monitor.add_scalar.reset_mock()

            # Ensure that ACCURACY is used when metric=None
            training_plan.testing_routine(metric=None,
                                          metric_args={},
                                          history_monitor=history_monitor,
                                          before_train=True)

            # ensure that the history monitor was called with the correct parameters
            # note that this requires knowing the actual value of the mean square error metric
            # for this specific case. For now it works but we may have to relax the constraint
            # to something like assert_called_once() in the future if the models default to
            # different values.
            history_monitor.add_scalar.assert_called_once_with(metric={'ACCURACY': 0.5},
                                                               iteration=1,
                                                               epoch=None,
                                                               test=True,
                                                               test_on_local_updates=False,
                                                               test_on_global_updates=True,
                                                               total_samples=4,
                                                               batch_samples=4,
                                                               num_batches=1)

            # check if `classes_` attribute of classifier has been created
            self.assertTrue(hasattr(training_plan.model(), 'classes_'), msg=training_plan.parent_type.__name__ + ' does not automatically create the classes_ attribute')
            self.assertTrue(np.array_equal(training_plan.model().classes_, np.array([0, 1])))

            history_monitor.add_scalar.reset_mock()



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
