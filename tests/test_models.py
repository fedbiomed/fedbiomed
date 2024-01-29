import copy
import logging
import unittest
import urllib.request
from unittest.mock import MagicMock, create_autospec, mock_open, patch

import numpy as np
import torch
from declearn.optimizer import Optimizer
from fedbiomed.common.optimizers.declearn import MomentumModule
from declearn.model.sklearn import NumpyVector
from declearn.model.torch import TorchVector
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor

from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.models import SkLearnModel, TorchModel
from fedbiomed.common.models._sklearn import SKLEARN_MODELS


class TestDocumentationLinks(unittest.TestCase):
    skip_internet_test: bool
    baselinks = (
        'https://scikit-learn.org/',
        'https://gitlab.inria.fr/',
        'http://www.plantuml.com/',
        'https://arxiv.org/',
    )
    links = (
        'https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html',
        'https://gitlab.inria.fr/magnet/declearn/declearn2/-/tree/r2.1',
        'http://www.plantuml.com/plantuml/dsvg/xLRDJjmm4BxxAUR82fPbWOe2guYsKYyhSQ6SgghosXDYuTYMFIbjdxxE3r7MIac3UkH4i6Vy_SpCsZU1kAUgrApvOEofK1hX8BSUkIZ0syf88506riV7NnQCNGLUkXXojmcLosYpgl-0YybAACT9cGSmLc80Mn7O7BZMSDikNSTqOSkoCafmGdZGTiSrb75F0pUoYLe6XqBbIe2mtgCWPGqG-f9jTjdc_l3axEFxRBEAtmC2Hz3kdDUhkqpLg_iH4JlNzfaV8MZCwMeo3IJcog047Y3YYmvuF7RPXmoN8x3rZr6wCef0Mz5B7WXwyTmOTBg-FCcIX4HVMhlAoThanwvusqNhlgjgvpsN2Wr130OgL80T9r4qIASd5zaaiwF77lQAEwT_fTK2iZrAO7FEJJNFJbr27tl-eh4r-SwbjY1FYWgm1i4wKgNwZHu2eGFs3-27wvJv7CPjuCLUq6kAWKPsRS1pGW_RhWt28fczN9czqTF8lQc7myVTQRslKRljKYBSgDxhTbA0Ft1btkPbwjotUNcRbqY_krm-TPrA1RRNw9CA-2o6DUcNvzd_u9bUU9C7zhrpNxCPq1lCGAWj5BCuJVSh7C9iuQk3CQjXknW8eA9_koHJF50nplnWlRfTD0WVpZg4vh_FxxBR5ch_X57pGA8c7jY43MFuKoudhvYqWdL3fI-tfFbVsKYzxQkxl_XprxATLz69br_40nMQWWRqFz1_rvunjlnQA2dHV5jc340YSL54zMXa-o8U_72y58i_7NfLeg5h5iWwTXDNgrB_0G00',
        'https://arxiv.org/abs/1711.05101',
    )
    skip_links = []

    def setUp(self) -> None:
        # test internet connection by reaching google website
        google_url = 'http://www.google.com'
        try:
            url_res = urllib.request.urlopen(google_url)
        except urllib.error.URLError:
            self.skip_internet_test = True
            return
        if url_res.code != 200:
            self.skip_internet_test = True
            return
        else:
            self.skip_internet_test = False

        for baselink in self.baselinks:
            try:
                url_res = urllib.request.urlopen(baselink)
            except urllib.error.URLError:
                skip = True
            else:
                if url_res.code != 200:
                    skip = True
                else:
                    skip = False
            self.skip_links.append(skip)

    def tearDown(self) -> None:
        pass

    def test_testdocumentationlinks_01(self):
        if self.skip_internet_test:
            self.skipTest("no internet connection: skipping test_testdocumentationlinks_01")
        for skip_links, link in zip(self.skip_links, self.links):
            if skip_links:
                print(f"Skipping URL test for {link} because base link of web site does not answer")
            else:
                url_res = urllib.request.urlopen(link)
                self.assertEqual(url_res.code, 200, f"cannot reach url link {link} pointed in documentation")


class TestSkLearnModelBuilder(unittest.TestCase):
    def setUp(self):
        self.implemented_models =  (
            SGDClassifier,
            SGDRegressor
        )

    def test_sklearnbuilder_1_test_sklearn_builder(self):
        for sk_model in self.implemented_models:
            model = SkLearnModel(sk_model)
            self.assertIsInstance(model._instance, SKLEARN_MODELS[sk_model.__name__])
            self.assertTrue(SKLEARN_MODELS.get(sk_model.__name__, False))

    def test_sklearnbuilder_2_test_sklearn_methods(self):
        # check that methods in implemented model also belong to the builder
        for model in self.implemented_models:
            _fbm_models = SKLEARN_MODELS[model.__name__]
            model_wrapper = SkLearnModel(model)
            for method in dir(_fbm_models):
                self.assertTrue(hasattr(model_wrapper, str(method),))

    def test_sklearnbuilder_3_test_sklearn_builder_error(self):
        for sk_model in self.implemented_models:
            model = SkLearnModel(sk_model)
            with self.assertRaises(FedbiomedModelError):
                val = model.this_method_does_not_exist()

    def test_sklearnbuilder_4_test_sklearn_deepcopy(self):
        for sk_model in self.implemented_models:
            model = SkLearnModel(sk_model)
            copied_model = copy.deepcopy(model)

            # check that copied_model is a deepcopy of model
            self.assertIsInstance(model, SkLearnModel)
            self.assertIsInstance(copied_model, SkLearnModel)
            self.assertNotEqual(id(model), id(copied_model), "error, deep copy failed, objects share same reference")
            # check that all attributes have different references

            for attribute, copied_attribute in zip(model._instance.__dict__, copied_model._instance.__dict__):
                self.assertNotEqual(id(getattr(model._instance,attribute)), id(getattr(copied_model._instance, copied_attribute)),
                                    f"deep copy failed, attribute {attribute} {copied_attribute} have shared refrences!")
            

            # check that model parameters are not the same
            model.set_init_params({'n_classes': 2, 'n_features': 4})
            copied_model.set_init_params({'n_classes': 2, 'n_features': 4})
            for layer_name in model.param_list:
                
                self.assertNotEqual(id(getattr(model.model, layer_name)), id(getattr(copied_model.model, layer_name)))
            
            new_weights = {layer: np.random.normal(size=getattr(model.model, layer).shape) for layer in model.param_list}
            model.set_weights(new_weights)
            for layer_name in model.param_list:
                self.assertFalse(np.array_equal(getattr(model.model, layer_name), getattr(copied_model.model, layer_name)))


class TestSkLearnModel(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.sgdclass_model = SkLearnModel(SGDClassifier)
        self.sgdregressor_model = SkLearnModel(SGDRegressor)
        self.models = (SGDClassifier, SGDRegressor)

        self.declearn_optim = Optimizer(lrate=.01, modules=[MomentumModule(.1)])
        
        # create dummy data
        data_2d = np.array([[1, 2, 3, 1, 2, 3],
                            [1, 2, 0, 1, 2, 3],
                            [1, 2, 3, 1, 2, 3],
                            [1, 2, 3, 1, 2, 3],
                            [1, 0, 3, 1, 2, 3],
                            [1, 2, 3, 2, 2, 3],
                            [1, 0, 1, 1, 2, 0],
                            [1, 0, 3, 1, 2, 3],
                            [1, 2, 3, 1, 0, 0],
                            [0, 2, 2, 1, 2, 3],
                            [1, 2, 0, 1, 0, 3]])
        data_1d = np.array([1, 2, 3, 1, 2, 3, 1, 2, 2, 1, 3]).reshape(-1, 1)
        self.data_collection = (data_1d, data_2d)
        self.targets = np.array([[1], [2], [0], [1], [0], [1], [1], [2], [0], [1], [0]])
        self._n_classes = 3  # number of classes in the data_collection
        
    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_sklearnmodel_01_init_failures(self):
        class InvalidModel:
            pass
        invalid_models = (
            torch.nn.Linear(4, 1),
            InvalidModel
        )

        for invalid_model in invalid_models:
            with self.assertRaises(FedbiomedModelError):
                model = SkLearnModel(invalid_model)

    def test_sklearnmodel_02_method_export(self):
        """Test that 'SklearnModel.export' works - in one specific case."""
        saved_params = []
        def mocked_joblib_dump(obj, *_):
            saved_params.append(obj)
        coefs = {
            'coef_': np.array([[0.42]]),
            'intercept_': np.array([0.42]),
        }
        self.sgdclass_model.set_weights(coefs)
        with (
            patch('joblib.dump', side_effect=mocked_joblib_dump),
            patch('builtins.open', mock_open())
        ):
            self.sgdclass_model.export('filename')
            self.assertEqual(saved_params[-1].coef_, coefs["coef_"])
            self.assertEqual(saved_params[-1].intercept_, coefs["intercept_"])

    def test_sklearnmodel_03_method_reload(self):
        """Test that 'SklearnModel.reload' works - in one specific case."""
        self.sgdclass_model.set_init_params({'n_classes':3, 'n_features':5})
        coefs = {
            'coef_': np.array([[0.42]]),  # true shape would be (3, 5)
            'intercept_': np.array([0.42]),  # true shape would be (3,)
        }
        self.sgdclass_model.model.coef_ = coefs["coef_"]
        self.sgdclass_model.model.intercept_ = coefs["intercept_"]
        with (
            patch('joblib.load', return_value=self.sgdclass_model.model),
            patch('builtins.open', mock_open())
        ):
            self.sgdclass_model.reload('filename')
            self.assertDictEqual(self.sgdclass_model.get_weights(), coefs)

    def test_sklearnmodel_04_set_init_params(self):
        # self.assertEqual(training_plan._model.n_iter_, 1)
        # test several values for `model_args`
        model_args_iterator = (
            {'n_classes': 2, 'n_features': 1},
            {'n_classes': 2, 'n_features': 2},
            {'n_classes': 3, 'n_features': 1},
            {'n_classes': 3, 'n_features': 3}
        )

        for model_args in model_args_iterator:
            self.sgdclass_model.set_init_params(model_args)
            self.assertListEqual(sorted(self.sgdclass_model.param_list), sorted(['coef_', 'intercept_']))

    def test_sklearnmodel_05_set_init_params_failures(self):
        for model in self.models:
            model = SkLearnModel(model)
            with self.assertRaises(FedbiomedModelError):
                model.init_training()

    def test_sklearnmodel_06_sklearn_training_01_plain_sklearn(self):
        # FIXME: this is an more an integration test, but I feel it is quite useful
        # to test the correct execution of the whole training process
        # Goal fo the test: checking that plain sklearn model has been updated when trained 
        # using `Model` interface
        _n_classes = 3

        data_2d = np.array([[1, 2, 3, 1, 2, 3],
                            [1, 2, 0, 1, 2, 3],
                            [1, 2, 3, 1, 2, 3],
                            [1, 2, 3, 1, 2, 3],
                            [1, 0, 3, 1, 2, 3],
                            [1, 2, 3, 2, 2, 3],
                            [1, 0, 1, 1, 2, 0],
                            [1, 0, 3, 1, 2, 3],
                            [1, 2, 3, 1, 0, 0],
                            [0, 2, 2, 1, 2, 3],
                            [1, 2, 0, 1, 0, 3]])
        data_1d = np.array([1, 2, 3, 1, 2, 3, 1, 2, 2, 1, 3]).reshape(-1, 1)
        targets = np.array([[1], [2], [0], [1], [0], [1], [1], [2], [0], [1], [0]])

        for data in (data_1d, data_2d):
            for model in self.models:
                # disable learning rate evolution, penality
                model = SkLearnModel(model)
                model.set_init_params(model_args={'n_classes': _n_classes, 'n_features': data.shape[1]})
                model.init_training()
                init_model = copy.deepcopy(model)
                #for idx in range(n_values):
                model.train(data, targets)
                grads = model.get_gradients()
                model.apply_updates(grads)

                # checks
                self.assertEqual(model.model.n_iter_, 1, "BaseEstimator n_iter_ attribute should always be reset to 1")
                for layer in model.param_list:
                    self.assertFalse(np.array_equal(getattr(model.model, layer), getattr(init_model.model, layer),
                                                    "model has not been updated during training"))

    def test_sklearnmodel_06_sklearn_training_02_plain_sklearn_grad_descent(self):
        #  checks plain sklearn is effectivly doing a gradient descent
        data = np.array([[1, 1, 1, 1,],
                         [1, 0,0, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 0]])
        
        targets = np.array([[1], [0], [1], [1]])
        random_seed = 1234
        learning_rate = .1234
        for model in self.models:
            model = SkLearnModel(model)
            model.set_params(random_state=random_seed,
                                eta0=learning_rate,
                                penalty=None,
                                learning_rate='constant')

            model.set_init_params({'n_features': 4, 'n_classes': 2})
            for _ in range(2):
                init_model= copy.deepcopy(model)
                model.init_training()
                model.train(data, targets)
                grads = model.get_gradients() # get_gradients returns  - learning_rate * grads
                model.apply_updates(grads)

            # checks that model updates(t+1) = update(t) - learning_rate * grads
            for layer in model.param_list:
                self.assertTrue(np.all(np.isclose(getattr(model.model, layer),
                                           getattr(init_model.model, layer) + grads[layer])))
        

    def test_sklearnmodel_06_sklearn_training_03_declearn_optimizer(self):

        n_iter = 10 # number of iterations
        
        for data in self.data_collection:
            for model in self.models:
                model = SkLearnModel(model)

                model.disable_internal_optimizer()
                model.set_init_params(model_args={'n_classes': self._n_classes, 'n_features': data.shape[1]})
                model.init_training()
                init_model_weights = model.get_weights()
                for _ in range(n_iter):
                    model.train(data, self.targets)
                grads = NumpyVector(model.get_gradients())
                updts = self.declearn_optim.compute_updates_from_gradients(model, grads)
                model.apply_updates(updts.coefs)

                # checks
                self.assertEqual(model.model.n_iter_, 1, "BaseEstimator n_iter_ attribute should always be reset to 1")
                self.assertEqual(model.model.eta0, 1)
                for layer in model.param_list:
                    self.assertFalse(np.array_equal(getattr(model.model, layer), init_model_weights[layer]),
                                                    "model has not been updated during training")

    def test_sklearnmodel_07_train_failures(self):
        inputs = np.array([[1, 2], [1, 1],[0, 1]])
        target = np.array([[0], [2], [1]])
        for skmodel in self.models:
            model = SkLearnModel(skmodel)
            with self.assertRaises(FedbiomedModelError):
                # raise exception because model has not been initialized
                model.train(inputs, target)

    def test_sklearnmodel_08_get_weights(self):
        inputs = np.array([[1, 2], [1, 1],[0, 1]])
        target = np.array([0, 2, 1])
        for skmodel in self.models:
            model = SkLearnModel(skmodel)

            model.set_init_params(model_args={'n_classes': 3, 'n_features': 2})
            initial_model = copy.deepcopy(model)
            init_weights = initial_model.get_weights()

            model.model.partial_fit(inputs, target)
            model._instance.model = MagicMock(spec=BaseEstimator)

            for key in init_weights:
                # making sure updated model's weights are different than the initial ones
                setattr(model.model, key, 10 + init_weights[key])

            # Check that the weights-getter works properly.
            weights = model.get_weights()
            self.assertEqual(weights.keys(), init_weights.keys())
            for key, val in weights.items():
                init_val = init_weights[key]
                self.assertFalse(np.any(np.isclose(val, init_val)))

    def test_sklearnmodel_09_get_weights_failures(self):

        for skmodel in self.models:
            model = SkLearnModel(skmodel)
            with self.assertRaises(FedbiomedModelError):
                # should raise exception regarding missing `param_list`
                model.get_weights()
            model.set_init_params(model_args={'n_classes': 2, 'n_features': 3})
            model.param_list.append('wrong-attribute')
            with self.assertRaises(FedbiomedModelError):
                # should raise exception complaining about non reachable model layer
                model.get_weights()

    def test_sklearn_model_10_flatten_and_unflatten(self):
        """Tests flatten and unflatten methods of Sklearn methods"""

        inputs = np.array([[1, 2], [1, 1], [0, 1]])
        target = np.array([0, 2, 1])
        for model in self.models:
            model = SkLearnModel(model)
            model.set_init_params(model_args={'n_classes': 3, 'n_features': 2})
            model.model.partial_fit(inputs, target)
            coef = model.model.coef_.astype(float).flatten()
            intercepts = model.model.intercept_.astype(float).flatten()
            flatten = model.flatten()

            self.assertListEqual(flatten, [*intercepts, *coef])

            unflatten = model.unflatten(flatten)
            self.assertListEqual(unflatten["coef_"].tolist(), model.model.coef_.tolist())
            self.assertListEqual(unflatten["intercept_"].tolist(), model.model.intercept_.tolist())

            with self.assertRaises(FedbiomedModelError):
                model.unflatten({"un-sported-type": "oopps"})

            with self.assertRaises(FedbiomedModelError):
                model.unflatten(["not-float-list"])

    def test_sklearnmodel_11_set_weights(self):
        """Test that 'SkLearnModel.set_weights' works properly."""
        for skmodel in self.models:
            # Instantiate a model, initialize it and create random weights.
            model = SkLearnModel(skmodel)
            model.set_init_params(model_args={'n_classes': 3, 'n_features': 2})
            weights = {
                key: np.random.normal(size=wgt.shape).astype(wgt.dtype)
                for key, wgt in model.get_weights().items()
            }
            # Test that weights assignment works.
            model.set_weights(weights)
            current = model.get_weights()
            self.assertEqual(current.keys(), weights.keys())
            self.assertTrue(
                all(np.all(weights[key] == current[key]) for key in weights)
            )


class TestSklearnClassification(unittest.TestCase):
    implemented_models = [SGDClassifier]  # store here implemented model
    model_args = {
        SGDClassifier: {'max_iter': 4242, 'alpha': 0.999, 'n_classes': 2, 'n_features': 2, 'key_not_in_model': None},
    }
    expected_params_list = {
        SGDClassifier: ['intercept_', 'coef_'],

    }

    def setUp(self):

        logging.disable('CRITICAL')  # prevent flood of messages about missing datasets

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_model_sklearnclassification_01_parameters(self):
        # TODO: add testing additional parameters (set_learning_rate/get_learning_rate)

        for model in self.implemented_models:
            # binary classification
            sk_model = SkLearnModel(model)
            self.assertTrue(sk_model.is_classification)
            sk_model.set_init_params(model_args={'n_classes':2, 'n_features': 5})
            # Parameters all initialized to 0.
            for key in sk_model.param_list:
                self.assertTrue(np.all(getattr(sk_model.model, key) == 0.))
                self.assertEqual(getattr(sk_model.model, key).shape[0], 1)
            # Test that classes values are integers in the range [0, n_classes)
            for i in np.arange(2):
                self.assertEqual(sk_model.model.classes_[i], i)

            # Multiclass (class=3):

            multiclass_model_args = {
                **TestSklearnClassification.model_args[model],
                'n_classes': 3
            }
            sk_model = SkLearnModel(model)
            sk_model.set_init_params(multiclass_model_args)
            # Parameters all initialized to 0.
            for key in sk_model.param_list:
                self.assertTrue(np.all(getattr(sk_model.model, key) == 0.),
                                f"{model.__class__.__name__} Multiclass did not initialize all parms to 0.")
                self.assertEqual(getattr(sk_model.model, key).shape[0], 3,
                                 f"{model.__class__.__name__} Multiclass wrong shape for {key}")
            # Test that classes values are integers in the range [0, n_classes)
            for i in np.arange(3):
                self.assertEqual(sk_model.model.classes_[i], i,
                                 f"{model.__class__.__name__} Multiclass wrong values for classes")


    def test_model_sklearnclassification_03_losses(self):
        def fake_context_manager(value):
            # mimics context_manager
            class MockContextManager:
                return_value = None
                def __init__(self, *args, **kwargs):
                    pass
                def __enter__(self):
                    return next(value)
                def __exit__(self, type, value, traceback):
                    pass
            return MockContextManager()

        for model in self.implemented_models:

            sk_model = SkLearnModel(model)
            sk_model.model_args = {'n_classes': 3}
            sk_model.model.classes_ = np.array([0, 1, 2])
            sk_model.set_init_params({'n_classes': 3, 'n_features': 2})

            sk_model.init_training()
            actual_losses_stdout = ['loss: 1.0', 'loss: 0.0', 'loss: 2.0']


            iterator = iter(actual_losses_stdout)
            inputs = np.array([[1, 2], [1, 1],[0, 1]])
            target = np.array([[0], [2], [1]])

            # in this test, we will make sure that collected stdout is the same caught
            # by the `capture_stdout` context manager
            context_manager_patcher = patch('fedbiomed.common.models._sklearn.capture_stdout',
                                            return_value=fake_context_manager(iterator))

            collected_losses_stdout = []
            context_manager_patcher.start()
            sk_model.train(inputs, target, collected_losses_stdout)
            context_manager_patcher.stop()

            self.assertListEqual(collected_losses_stdout, actual_losses_stdout)

    def test_model_sklearnclassification_04_disable_internal_optimizer(self):

        model = SkLearnModel(SGDClassifier)
        # action
        model.disable_internal_optimizer()

        # checks

        self.assertEqual(model._null_optim_params['eta0'], model.model.eta0)
        self.assertEqual(model._null_optim_params['learning_rate'], model.model.learning_rate)


class TestSkLearnRegressorModel(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self) -> None:
        pass

    def test_model_sklearnregressor_01_disable_intenral_optimizer(self):
        model = SkLearnModel(SGDRegressor)

        model.disable_internal_optimizer()

        # checks
        self.assertEqual(model._null_optim_params['eta0'], model.model.eta0)
        self.assertEqual(model._null_optim_params['learning_rate'], model.model.learning_rate)


class TestTorchModel(unittest.TestCase):
    def setUp(self):
        self.torch_model = torch.nn.Linear(4, 1)
        self.model = TorchModel(self.torch_model)
        self.torch_optim = torch.optim.SGD(self.model.model.parameters(), lr=.01, momentum=.1)
        self.declearn_optim = Optimizer(lrate=.01, modules=[MomentumModule(.1)])

        self.data = torch.randn(8, 1,  4, requires_grad=True)
        self.targets = torch.tensor([1,0,2,1,0,2,1,1])#.type(torch.LongTensor)


    def tearDown(self) -> None:
        pass


    def fake_training_step(self, data, targets):
        output = self.model.model.forward(data)

        output = torch.squeeze(output, dim=1)

        loss   = torch.nn.functional.nll_loss(torch.squeeze(output), targets)
        return loss

    def test_torchmodel_01_get_gradients_method(self):
        # case where no gradients have been found: model has not been trained
        self.assertDictEqual({}, self.model.get_gradients(), "get_gradients should return an empty dict since model hasnot been trained")

        # case model has been trained with pytorch optimizer
        self.torch_optim.zero_grad()
        loss = self.fake_training_step(self.data, self.targets)
        loss.backward()
        self.torch_optim.step()

        grads = self.model.get_weights()
        for layer_name, values in grads.items():
            self.assertTrue(torch.all(values))

    def test_torchmodel_02_get_weights(self):
        # test case where model_wweitghs is retunred as a dict
        model_weights = self.model.get_weights()
        for (layer, wrapped_model_weight) in model_weights.items():
            self.assertTrue(torch.all(torch.isclose(wrapped_model_weight, self.torch_model.get_parameter(layer))))

    def test_torchmodel_03_set_weights(self):
        """Test that 'TorchModel.set_weights' works properly."""
        # Create random weights that are suitable for the model.
        weights = {
            key: torch.randn(size=wgt.shape, dtype=wgt.dtype)
            for key, wgt in self.model.get_weights().items()
        }
        # Test that weights assignment works properly.
        self.model.set_weights(weights)
        current = self.model.get_weights()
        self.assertEqual(current.keys(), weights.keys())
        self.assertTrue(
            all(torch.all(weights[key] == current[key]) for key in weights)
        )

    def test_torchmodel_04_apply_updates_1(self):
        init_weights = copy.deepcopy(self.model.get_weights())

        updates = torch.nn.Linear(4, 1).state_dict()

        self.model.apply_updates(updates)
        updated_weights = self.model.get_weights()

        # checks
        for (layer, w), (_, updated_w) in zip(init_weights.items(), updated_weights.items()):
            self.assertFalse(torch.all(torch.isclose(w, updated_w)))
            self.assertTrue(torch.all(torch.isclose(updated_w, self.model.model.get_parameter(layer))))

    def test_torchmodel_05_apply_updates_3_failures(self):
        # check that error is raised when passing incorrect type
        incorrect_types = (
            "incorrect usage",
            True,
            1234,
            [1, 2, 3],
            set((1,2,3))
        )
        for incorrect_type in incorrect_types:
            with self.assertRaises(FedbiomedModelError):
                self.model.apply_updates(incorrect_type)

    def test_torchmodel_06_predict(self):
        data = torch.randn(1, 1,  4, requires_grad=True)

        tested_prediction = self.model.predict(data)

        ground_truth_prediction = self.model.model(data)

        self.assertIsInstance(tested_prediction, np.ndarray)
        self.assertListEqual(tested_prediction.tolist(), ground_truth_prediction.tolist())

    def test_torchmodel_07_add_corrections_to_gradients(self):
        self.torch_optim.zero_grad()
        loss = self.fake_training_step(self.data, self.targets)

        loss.backward()
        # self.torch_optim.step()
        correction_values = (
            torch.randn(1,   4, requires_grad=True),
            torch.randn(1, requires_grad=True)
        )
        corrections = {
            layer_name: val
            for (layer_name, _), val
            in zip(self.model.model.named_parameters(), correction_values)
        }

        # zeroes gradients model
        self.model.model.zero_grad()
        # action
        self.model.add_corrections_to_gradients(corrections)
        # checks
        for (layer_name, param), val in zip(self.model.model.named_parameters(), correction_values):
            self.assertTrue(torch.all(torch.isclose(param.grad, val)))

    def test_torchmodel_08_training(self):
        self.model.init_training()

        #  before training, check values contained in `init_training` are the same as in model
        weights = self.model.get_weights()
        for key, wgt in weights.items():
            self.assertTrue(key in self.model.init_params)
            self.assertTrue(torch.all(wgt == self.model.init_params[key]))

        # mimic training by updating model weights
        # 1. training through torch optimizer
        self.torch_optim.zero_grad()
        loss = self.fake_training_step(self.data, self.targets)

        loss.backward()
        self.torch_optim.step()
        torch_update_weights = self.model.get_weights()
        # checks
        for key, wgt in torch_update_weights.items():
            self.assertTrue(key in self.model.init_params)
            self.assertFalse(torch.all(wgt == self.model.init_params[key]))

        # 2. training through declearn optimizer
        self.model.init_training()
        #  before training, check values contained in `init_training` are the same as in model
        weights = self.model.get_weights()
        for key, wgt in weights.items():
            self.assertTrue(key in self.model.init_params)
            self.assertTrue(torch.all(wgt == self.model.init_params[key]))

        self.model.model.zero_grad()
        loss = self.fake_training_step(self.data, self.targets)

        loss.backward()
        grads = TorchVector(self.model.get_weights())
        updts = self.declearn_optim.compute_updates_from_gradients(self.model, grads)
        self.model.apply_updates(updts.coefs)

        declearn_optimized_model_weights = self.model.get_weights()
        # checks
        for key, wgt in declearn_optimized_model_weights.items():
            self.assertTrue(key in self.model.init_params)
            self.assertFalse(torch.all(wgt == self.model.init_params[key]))

    def test_torch_model_9_flatten_and_unflatten(self):
        """Tests flatten and unflatten methods of Sklearn methods"""

        # Flatten model parameters
        flatten = self.model.flatten()

        weights = self.model.get_weights()

        w = weights["weight"].flatten().tolist()
        b = weights["bias"].flatten().tolist()
        self.assertListEqual(flatten, [*w, *b])

        unflatten = dict(self.model.unflatten(flatten))
        self.assertListEqual(unflatten["weight"].tolist(), weights["weight"].tolist())
        self.assertListEqual(unflatten["bias"].tolist(), weights["bias"].tolist())

        # Test invalid argument types
        with self.assertRaises(FedbiomedModelError):
            self.model.unflatten({"un-sported-type": "oopps"})

        with self.assertRaises(FedbiomedModelError):
            self.model.unflatten(["not-float-list"])

    def test_torchmodel_09_export(self):
        """Test that 'TorchModel.export' works properly."""
        with patch("torch.save") as save_patch:
            self.model.export("filename")
        save_patch.assert_called_once()
        call_args = save_patch.call_args
        self.assertEqual(call_args.args[1], "filename")
        t1 = self.torch_model.state_dict()
        t2 = call_args.args[0]
        self.assertTrue(all([torch.allclose(t1[key], t2[key]) for key in list(t1.keys()) + list(t2.keys())]))

    def test_torchmodel_10_reload(self):
        """Test that 'TorchModel.reload' works properly."""
        d1 = self.model.model.state_dict()
        with patch("torch.load", return_value=d1) as load_patch:
            self.model.reload("filename")
        load_patch.assert_called_once_with("filename")
        d2 = self.model.model.state_dict()
        self.assertTrue(all([torch.allclose(d1[key], d2[key]) for key in list(d1.keys()) + list(d2.keys())]))

    def test_torchmodel_11_reload_fails(self):
        """Test that 'TorchModel.reload' fails with a non-torch dump object."""
        with patch("torch.load", return_value=MagicMock()) as load_patch:
            with self.assertRaises(FedbiomedModelError):
                self.model.reload("filename")
        load_patch.assert_called_once_with("filename")
        self.assertIs(self.model.model, self.torch_model)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
