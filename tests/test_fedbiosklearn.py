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
from unittest.mock import MagicMock, patch, mock_open

import fedbiomed.node.history_monitor
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.training_plans import SKLearnTrainingPlan, FedPerceptron, FedSGDRegressor, FedSGDClassifier
from fedbiomed.common.training_plans._sklearn_models import SKLearnTrainingPlanPartialFit
from sklearn.linear_model import SGDClassifier

class Custom:
    def testing_step(mydata, mytarget):
        return {'Metric': 42.0}


class FakeTrainingArgs:

    def pure_training_arguments(self):
        return {"epochs": 1,
                "batch_maxnum": 2}


class TestSklearnTrainingPlanBasicInheritance(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.abstract_methods_patcher = patch.multiple(SKLearnTrainingPlan, __abstractmethods__=set())
        self.abstract_methods_patcher.start()
        SKLearnTrainingPlan._model_cls = FedPerceptron._model_cls  # just for testing

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)
        self.abstract_methods_patcher.stop()

    def test_sklearntrainingplanbasicinheritance_01_dataloaders(self):
        X = np.array([])
        loader = NPDataLoader(dataset=X, target=X)

        training_plan = SKLearnTrainingPlan()
        training_plan.set_data_loaders(loader, loader)

        with self.assertRaises(FedbiomedTrainingPlanError):
            training_plan.set_data_loaders('wrong-type', 'wrong-type')

    def test_sklearntrainingplanbasicinheritance_02_training_testing_routine(self):
        training_plan = SKLearnTrainingPlan()
        # Model is not of the correct type
        with patch.object(training_plan, '_model'):
            with self.assertRaises(FedbiomedTrainingPlanError):
                training_plan.training_routine()

        X = np.array([])
        loader = NPDataLoader(dataset=X, target=X)
        training_plan.set_data_loaders(loader, loader)
        training_plan.training_routine()  # assert this works without failure

        # Data loader is not of the correct type
        with patch.object(training_plan, 'training_data_loader'):
            with self.assertRaises(FedbiomedTrainingPlanError):
                training_plan.training_routine()

        # The training routine raises some error (here ValueError for example)
        with patch.object(training_plan, '_training_routine', side_effect=ValueError):
            with self.assertRaises(FedbiomedTrainingPlanError):
                training_plan.training_routine()

        # Node requests GPU, but sklearn does not support it
        logging.disable(logging.NOTSET)  # Temporarily re-enable logging to capture warning output
        with self.assertLogs('fedbiomed', logging.DEBUG) as captured:
            training_plan.training_routine(node_args={'gpu_only': True})
            self.assertIn('Node would like to force GPU usage, but sklearn training '
                          'plan does not support it. Training on CPU.',
                          captured.output[0])
        logging.disable(logging.CRITICAL)

        # testing_routine for classification tasks should create classes on the fly if they don't exist
        with patch.object(training_plan, '_classes_from_concatenated_train_test', return_value=np.array([0, 1])), \
                patch('fedbiomed.common.training_plans.BaseTrainingPlan.testing_routine', return_value=None):
            training_plan._is_classification = True  # testing fixture for classification
            training_plan.testing_routine(metric=None, metric_args={}, history_monitor=None, before_train=True)
            self.assertTrue(hasattr(training_plan.model(), 'classes_'))
            self.assertListEqual([x for x in training_plan.model().classes_], [0, 1])

        # Inferring the classes works correctly
        X = np.array([0, 1, 2, 3, 0, 1, 2])
        loader = NPDataLoader(dataset=X, target=X)
        training_plan.set_data_loaders(loader, loader)
        classes = training_plan._classes_from_concatenated_train_test()
        self.assertListEqual(
            [x for x in classes],
            [x for x in np.unique(X)]
        )

    def test_sklearntrainingplanbasicinheritance_03_save_load(self):
        training_plan = SKLearnTrainingPlan()
        saved_params = []

        def mocked_joblib_dump(obj, *args, **kwargs):
            saved_params.append(obj)

        # Base case where params are not provided to save function
        with patch('fedbiomed.common.training_plans._sklearn_training_plan.joblib.dump',
                   side_effect=mocked_joblib_dump), \
                patch('builtins.open', mock_open()):
            training_plan.save('filename')
            self.assertEqual(saved_params[-1], training_plan.model())

        # Params passed to save function as dict
        with patch('fedbiomed.common.training_plans._sklearn_training_plan.joblib.dump',
                   side_effect=mocked_joblib_dump), \
                patch('builtins.open', mock_open()):
            training_plan.save('filename', params={'coef_': 0.42, 'intercept_': 0.42})
            self.assertEqual(saved_params[-1].coef_, 0.42)
            self.assertEqual(saved_params[-1].intercept_, 0.42)

        # Params passed as dict with 'model_params' field
        with patch('fedbiomed.common.training_plans._sklearn_training_plan.joblib.dump',
                   side_effect=mocked_joblib_dump), \
                patch('builtins.open', mock_open()):
            training_plan.save('filename', params={'model_params': {'coef_': 0.42, 'intercept_': 0.42}})
            self.assertEqual(saved_params[-1].coef_, 0.42)
            self.assertEqual(saved_params[-1].intercept_, 0.42)

        # Saved object is not the correct type
        with patch('fedbiomed.common.training_plans._sklearn_training_plan.joblib.load',
                   return_value=FedSGDRegressor._model_cls()), \
                patch('builtins.open', mock_open()):
            with self.assertRaises(FedbiomedTrainingPlanError):
                training_plan.load('filename')

        # Option to retrieve model parameters instead of full model from load function
        with patch.object(training_plan, '_param_list', ['coef_', 'intercept_']), \
                patch.object(training_plan._model, 'coef_', 0.42), \
                patch.object(training_plan._model, 'intercept_', 0.42), \
                patch('fedbiomed.common.training_plans._sklearn_training_plan.joblib.load',
                      return_value=training_plan._model), \
                patch('builtins.open', mock_open()):
            params = training_plan.load('filename', to_params=True)
            self.assertDictEqual(params, {'model_params': {'coef_': 0.42, 'intercept_': 0.42}})
            params = training_plan.after_training_params()
            self.assertDictEqual(params, {'coef_': 0.42, 'intercept_': 0.42})


class TestSklearnTrainingPlanPartialFit(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.abstract_methods_patcher = patch.multiple(SKLearnTrainingPlanPartialFit, __abstractmethods__=set())
        self.abstract_methods_patcher.start()
        SKLearnTrainingPlanPartialFit._model_cls = FedPerceptron._model_cls  # just for testing

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)
        self.abstract_methods_patcher.stop()

    def test_sklearntrainingplanpartialfit_01_losses(self):
        training_plan = SKLearnTrainingPlanPartialFit()
        logging.disable(logging.NOTSET)  # Temporarily re-enable logging to capture warning output
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            losses = training_plan._parse_sample_losses(['loss: 1.0',
                                                         'no loss',
                                                         'loss: \n 4.2',
                                                         'loss: inf',
                                                         'loss: nan',
                                                         'loss: over 9000'])
            self.assertEqual(len(captured.output), 1)
            self.assertIn('over 9000', captured.output[0])
            self.assertIn('Value error during monitoring', captured.output[0])
        self.assertListEqual(losses[:-1], [1.0, 4.2, np.inf])
        self.assertTrue(np.isnan(losses[-1]))

        # test a pretty common loss output
        losses = training_plan._parse_sample_losses([
            '-- Epoch 1',
            'Norm: 0.00, NNZs: 0, Bias: 0.000000, T: 1, Avg. loss: 0.000000',
            'Total training time: 0.00 seconds.'
        ])
        self.assertEqual(losses[0], 0.)
        logging.disable(logging.CRITICAL)

    def test_sklearntrainingplanpartialfit_02_training_routine(self):
        training_plan = SKLearnTrainingPlanPartialFit()
        test_x = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        test_y = np.array([1, 0, 1, 0])
        train_data_loader = test_data_loader = NPDataLoader(dataset=test_x, target=test_y, batch_size=1)
        training_plan.set_data_loaders(train_data_loader=train_data_loader, test_data_loader=test_data_loader)
        training_plan._training_args['epochs'] = 1
        training_plan._training_args['num_updates'] = None
        training_plan._training_args['batch_size'] = train_data_loader.batch_size()

        # First scenario: assert that number of training iterations is correct
        training_plan._training_args['batch_maxnum'] = None
        with patch.object(training_plan, '_train_over_batch', return_value=0.) as mocked_train:
            num_samples_observed = training_plan._training_routine(history_monitor=None)
            self.assertEqual(mocked_train.call_count, 4)
            self.assertEqual(num_samples_observed,
                             len(test_x) / training_plan._training_args['batch_size'] *
                             training_plan._training_args['epochs'], "Training routine for SkLearnTrainingPlan"
                                                                     " did not return correct number of samples "
                                                                     "observed during the training")

        # Second scenario: assert that history monitor is given the correct reporting values
        training_plan._training_args['batch_maxnum'] = 1
        training_plan._training_args['log_interval'] = 1
        history_monitor = MagicMock(spec=fedbiomed.node.history_monitor.HistoryMonitor)
        with patch.object(training_plan, '_train_over_batch', return_value=0.) as mocked_train:
            training_plan._training_routine(history_monitor=history_monitor)
            self.assertEqual(mocked_train.call_count, 1)
            self.assertEqual(history_monitor.add_scalar.call_count, 1)
            history_monitor.add_scalar.assert_called_with(
                train=True,
                num_batches=1,
                total_samples=1,
                num_samples_trained=1,
                metric={'Loss hinge': 0.},
                iteration=1,
                epoch=1,
                batch_samples=1
            )


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

    def test_sklearntrainingplancommonfunctionalities_01_model_args(self):
        for training_plan in self.training_plans:
            # training plan type
            self.assertEqual(training_plan.type(), TrainingPlans.SkLearnTrainingPlan)
            # ensure that the model args passed by the researcher are correctly stored in the class
            self.assertDictEqual(training_plan._model_args,
                                 TestSklearnTrainingPlansCommonFunctionalities.model_args[training_plan.parent_type])
            for key, val in training_plan.model().get_params().items():
                # ensure that the model args passed by the researcher are correctly passed to the sklearn model
                if key in TestSklearnTrainingPlansCommonFunctionalities.model_args[training_plan.parent_type]:
                    self.assertEqual(
                        val,
                        TestSklearnTrainingPlansCommonFunctionalities.model_args[training_plan.parent_type][key]
                    )
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

    def test_sklearntrainingplancommonfunctionalities_02_save_and_load(self):
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

    @patch.multiple(SKLearnTrainingPlan, __abstractmethods__=set())
    def test_sklearntrainingplancommonfunctionalities_03_getters(self):
        """Test getter methods of SkLearnTrainingPlan"""
        # Set a model class to be able to build abstract SkLearnTrainingPlan class
        SKLearnTrainingPlan._model_cls = SGDClassifier
        training_plan = SKLearnTrainingPlan()
        training_plan._model_cls = SGDClassifier
        _tr_args = FakeTrainingArgs()
        tr_args = _tr_args.pure_training_arguments()
        m_args = {'n_classes': 2, 'n_features': 1}
        training_plan.post_init(m_args, _tr_args)

        model_args = training_plan.model_args()
        training_args = training_plan.training_args()

        self.assertDictEqual(m_args, model_args)
        self.assertDictEqual(training_args, tr_args)

    def test_sklearntrainingplancommonfunctionalities_03_exceptions_are_correctly_converted(self):
        # Dataset
        test_x = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        test_y = np.array([1, 0, 1, 0])
        train_data_loader = test_data_loader = NPDataLoader(dataset=test_x, target=test_y, batch_size=len(test_x))
        for training_plan in self.training_plans:
            training_plan.set_data_loaders(train_data_loader=train_data_loader, test_data_loader=test_data_loader)

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
                patch_testing_step.return_value = None
                with self.assertRaises(FedbiomedTrainingPlanError):
                    training_plan.testing_routine(metric=MetricTypes.ACCURACY,
                                                  metric_args={},
                                                  history_monitor=None,
                                                  before_train=True)

    def test_sklearntrainingplancommonfunctionalities_04_custom_testing_step(self):
        history_monitor = MagicMock()
        history_monitor.add_scalar = MagicMock(return_value=None)

        # Dataset
        test_x = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        test_y = np.array([1, 0, 1, 0])
        train_data_loader = test_data_loader = NPDataLoader(dataset=test_x, target=test_y, batch_size=len(test_x))

        for training_plan in self.training_plans:
            # case where the researcher defines a custom testing step
            training_plan.testing_step = Custom.testing_step
            training_plan.set_data_loaders(train_data_loader=train_data_loader, test_data_loader=test_data_loader)

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

    def test_sklearntrainingplancommonfunctionalities_05_train_over_batch(self):
        for training_plan in self.training_plans:
            inputs = np.array([[0., 0.]])  # one batch of a 2-feature array
            target = np.array([[0.]])
            loss = training_plan._train_over_batch(inputs, target, report=True)
            # Assert loss values are within reasonable ranges
            # Since different models handle loss differently, we cannot assert that loss == 0.
            # Instead, we check it lies within [-1., 1.]
            self.assertGreaterEqual(loss, -1.,
                                    f"{training_plan.__class__.__name__} unexpected loss value")
            self.assertLessEqual(loss, 1.,
                                 f"{training_plan.__class__.__name__} unexpected loss value")
            # Test that coefs are not updated.
            # Cannot test intercept because classes are internally converted to [-1, 1], and therefore intercept_
            # is updated even after a single iteration
            self.assertTrue(np.all(training_plan._model.coef_ == 0),
                            f"{training_plan.__class__.__name__} incorrectly computed non-zero gradients for coef_.")
            self.assertEqual(training_plan._model.n_iter_, 1)

            # When report is False, expected return value is NaN
            loss = training_plan._train_over_batch(inputs, target, report=False)
            self.assertTrue(np.isnan(loss),
                            f"{training_plan.__class__.__name__} loss should be NaN")
            self.assertTrue(np.all(training_plan._model.coef_ == 0),
                            f"{training_plan.__class__.__name__} incorrectly computed non-zero gradients for coef_.")
            self.assertEqual(training_plan._model.n_iter_, 1)  # n_iter_ == 1 always after calling _train_over_batch


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

    def test_skleanregression_01_parameters(self):
        for training_plan in self.training_plans:
            self.assertFalse(training_plan._is_classification)
            self.assertTrue(training_plan._is_regression)
            # Parameters all initialized to 0.
            for key in training_plan._param_list:
                self.assertTrue(np.all(getattr(training_plan._model, key) == 0.))

    def test_sklearnregression_02_testing_routine(self):
        """ Testing `testing_routine` of SKLearnModel training plan"""
        history_monitor = MagicMock()
        history_monitor.add_scalar = MagicMock(return_value=None)

        # Dataset
        test_x = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        test_y = np.array([1, 0, 1, 0])
        train_data_loader = test_data_loader = NPDataLoader(dataset=test_x, target=test_y, batch_size=len(test_x))

        for training_plan in self.training_plans:
            # Test testing routine without setting testing_data_loader
            with self.assertRaises(FedbiomedTrainingPlanError):
                training_plan.testing_routine(metric=None,
                                              metric_args={},
                                              history_monitor=history_monitor,
                                              before_train=True)

            # set data loader and call testing routine
            training_plan.set_data_loaders(train_data_loader=train_data_loader, test_data_loader=test_data_loader)
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

    def test_sklearnclassification_01_parameters(self):
        for training_plan in self.training_plans:
            self.assertTrue(training_plan._is_classification)
            # Parameters all initialized to 0.
            for key in training_plan._param_list:
                self.assertTrue(np.all(getattr(training_plan._model, key) == 0.))
                self.assertEqual(getattr(training_plan._model, key).shape[0], 1)
            # Test that classes values are integers in the range [0, n_classes)
            for i in np.arange(2):
                self.assertEqual(training_plan._model.classes_[i], i)

        # Multiclass
        for sklearn_model_type, new_subclass_type in self.subclass_types.items():
            training_plan = new_subclass_type()
            multiclass_model_args = {
                **TestSklearnTrainingPlansClassification.model_args[sklearn_model_type],
                'n_classes': 3
            }
            training_plan.post_init(multiclass_model_args, FakeTrainingArgs())
            # Parameters all initialized to 0.
            for key in training_plan._param_list:
                self.assertTrue(np.all(getattr(training_plan._model, key) == 0.),
                                f"{training_plan.__class__.__name__} Multiclass did not initialize all parms to 0.")
                self.assertEqual(getattr(training_plan._model, key).shape[0], 3,
                                 f"{training_plan.__class__.__name__} Multiclass wrong shape for {key}")
            # Test that classes values are integers in the range [0, n_classes)
            for i in np.arange(3):
                self.assertEqual(training_plan._model.classes_[i], i,
                                 f"{training_plan.__class__.__name__} Multiclass wrong values for classes")

    def test_sklearnclassification_02_testing_routine(self):
        """ Testing `testing_routine` of SKLearnModel training plan"""
        history_monitor = MagicMock()
        history_monitor.add_scalar = MagicMock(return_value=None)

        # Dataset
        test_x = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        test_y = np.array([1, 0, 1, 0])
        train_data_loader = test_data_loader = NPDataLoader(dataset=test_x, target=test_y, batch_size=len(test_x))

        for training_plan in self.training_plans:
            # Test testing routine without setting testing_data_loader
            with self.assertRaises(FedbiomedTrainingPlanError):
                training_plan.testing_routine(metric=None,
                                              metric_args={},
                                              history_monitor=history_monitor,
                                              before_train=True)

            training_plan.set_data_loaders(train_data_loader=train_data_loader, test_data_loader=test_data_loader)

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
            self.assertTrue(hasattr(training_plan.model(), 'classes_'),
                            msg=training_plan.parent_type.__name__ + ' does not automatically create the classes_ attribute')
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
            self.assertTrue(
                hasattr(training_plan.model(), 'classes_'),
                msg=training_plan.parent_type.__name__ + ' does not automatically create the classes_ attribute')
            self.assertTrue(np.array_equal(training_plan.model().classes_, np.array([0, 1])))

            history_monitor.add_scalar.reset_mock()

    def test_sklearnclassification_03_losses(self):
        for training_plan in self.training_plans:
            batch_losses_stdout = [
                ['loss: 1.0'],
                ['loss: 0.0'],
            ]
            loss = training_plan._parse_batch_loss(batch_losses_stdout, None, None)
            self.assertEqual(loss, 0.5)

            batch_losses_stdout.append(['loss: inf'])
            loss = training_plan._parse_batch_loss(batch_losses_stdout, None, None)
            self.assertEqual(loss, np.inf)

            batch_losses_stdout.append(['loss: nan'])
            loss = training_plan._parse_batch_loss(batch_losses_stdout, None, None)
            self.assertTrue(np.isnan(loss))

            with patch.object(training_plan, '_model_args', {'n_classes': 3}), \
                    patch.object(training_plan._model, 'classes_', np.array([0, 1, 2])):
                batch_losses_stdout = [
                    ['loss: 1.0', 'loss: 0.0', 'loss: 2.0'],
                    ['loss: 0.0', 'loss: 1.0', 'epoch', 'loss: 0.0'],
                ]
                target = np.array([[0], [2]])
                loss = training_plan._parse_batch_loss(batch_losses_stdout, None, target)
                # batch-average losses for each class are: [0.5, 0.5, 1.0]
                # since we should have guessed once the first class, and once the last class, the final loss
                # is the mean of 0.5 and 1.0, i.e. it should be 0.75
                self.assertEqual(loss, 0.75)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

# Test init params
