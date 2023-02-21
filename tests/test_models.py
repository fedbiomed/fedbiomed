import copy
import unittest
import urllib.request
import logging
from unittest.mock import MagicMock, patch, mock_open
from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.logger import logger
from fedbiomed.common.models.model import (BaseSkLearnModel, SGDClassiferSKLearnModel,
                                           SGDRegressorSKLearnModel, SkLearnModel,
                                           capture_stdout, Models, TorchModel
)

import torch
from sklearn.linear_model import SGDClassifier, SGDRegressor
from declearn.optimizer import Optimizer
from declearn.optimizer.modules import MomentumModule
from declearn.model.torch import TorchVector
import numpy as np


class TestDocumentationLinks(unittest.TestCase):
    skip_internet_test: bool
    def setUp(self) -> None:
        # test internet connection by reaching google website
        google_url = 'http://www.google.com'
        try:
            url_res = urllib.request.urlopen(google_url)
        except urllib.error.URLError as err:
            self.skip_internet_test = True
            return
        if url_res.code != 200:
            self.skip_internet_test = True
        else:
            self.skip_internet_test = False
    
    def tearDown(self) -> None:
        pass
    
    def test_testdocumentationlinks_01(self):
        links = (
            'https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html',
        )
        if self.skip_internet_test:
            self.skipTest("no internet connection: skipping test_testdocumentationlinks_01")
        for link in links:
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
            self.assertIsInstance(model._instance, Models[sk_model.__name__])
            self.assertTrue(Models.get(sk_model.__name__, False))

    def test_sklearnbuilder_2_test_sklearn_methods(self):
        # check that methods in implemented model also belongs to the 
        for model in self.implemented_models:
            _fbm_models = Models[model.__name__]
            model_wrapper = SkLearnModel(model)
            for method in dir(_fbm_models):
                self.assertTrue(hasattr(model_wrapper, str(method),))
            
    def test_sklearnbuilder_3_test_sklearn_builder_error(self):
        for sk_model in self.implemented_models:
            model = SkLearnModel(sk_model)
            with self.assertRaises(FedbiomedModelError):
                val = model.this_method_does_not_exist()


class TestSkLearnModel(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.sgdclass_model = SkLearnModel(SGDClassifier)
        self.sgdregressor_model = SkLearnModel(SGDRegressor)
        self.models = (self.sgdclass_model, self.sgdregressor_model)

        self.n_features = (1, 10)  # possible number of features
        self.n_classes = (2, 5)  # possible number of classes (for classification)

        
    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)
    
    def test_sklearnmodel_method_01_save(self):
        saved_params = []
        def mocked_joblib_dump(obj, *args, **kwargs):
            saved_params.append(obj)
        
        self.sgdclass_model.set_weights({'coef_': 0.42, 'intercept_': 0.42})
        with patch('fedbiomed.common.models.model.joblib.dump',
                   side_effect=mocked_joblib_dump), \
                patch('builtins.open', mock_open()):
            self.sgdclass_model.save('filename')
            self.assertEqual(saved_params[-1].coef_, 0.42)
            self.assertEqual(saved_params[-1].intercept_, 0.42)
            
    def test_sklearnmodel_method_02_load(self):
        self.sgdclass_model.set_init_params({'n_classes':3, 'n_features':5})
        with (
            # patch.object(self.sgdclass_model, 'param_list', ['coef_', 'intercept_']), 
                patch.object(self.sgdclass_model.model, 'coef_', 0.42), 
                patch.object(self.sgdclass_model.model, 'intercept_',  0.42), 
                patch('fedbiomed.common.models.model.joblib.load',
                    return_value=self.sgdclass_model.model), 
                patch('builtins.open', mock_open())
                ):
            self.sgdclass_model.load('filename')
            self.assertDictEqual(self.sgdclass_model.get_weights(), {'coef_': 0.42, 'intercept_': 0.42})

    def test_sklearnmodel_03_set_init_params(self):
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
            
        self.assertEqual(self.sgdclass_model.batch_size, 0)
            
    def test_sklearmmodel_04_plain_sklearn_training(self):
        n_values = 100
        for model in self.models:
            for _n_features in self.n_features:
                
                data = np.random.randn(_n_features, n_values)
                
                if model._is_classification:
                    for _n_classes in self.n_classes:
                        targets = np.random.randint(0, _n_classes, n_values)
                else:
                    targets = np.random.randn(n_values)
                
                model.set_init_params(model_args={'n_classes': _n_classes, 'n_features': _n_features})
                model.init_training()
                for idx in range(n_values):
                    print("Target", targets, targets[idx])
                    model.train(data[idx:idx+1], targets[idx])
                grads = model.get_gradients()
                model.apply_updates(grads)
                
                # checks
                self.assertEqual(self.model.n_iter, n_values)
                self.assertEqual(model.batch_size, n_values)
                
    def test_sklearnmodel_05_sklearn_training_through_declearn_optimizer(self):
        pass
        
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
            # dual classes
            sk_model = SkLearnModel(model)
            self.assertTrue(sk_model._is_classification)
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
                    return value
                def __exit__(self, type, value, traceback):
                    pass
            return MockContextManager()
        
        for model in self.implemented_models:
            
            sk_model = SkLearnModel(model)
            sk_model.model_args = {'n_classes': 3}
            sk_model.model.classes_ = np.array([0, 1, 2])
            sk_model.set_init_params({'n_classes': 3, 'n_features': 2})

            sk_model.init_training()
            actual_losses_stdout = [
                    ['loss: 1.0', 'loss: 0.0', 'loss: 2.0'],
                    ['loss: 0.0', 'loss: 1.0', 'epoch', 'loss: 0.0'],
            ]
            
            inputs = np.array([[1, 2], [1, 1],[0, 1]])
            target = np.array([0, 2, 1])
            
            
            context_manager_patcher = patch('fedbiomed.common.models.model.capture_stdout',
                                            return_value=fake_context_manager(actual_losses_stdout))
            
            collected_losses_stdout = []
            context_manager_patcher.start()
            sk_model.train(inputs, target, collected_losses_stdout)
            context_manager_patcher.stop()

            
            self.assertListEqual(collected_losses_stdout, [actual_losses_stdout])



class TestSkLearnToolboxClasses(unittest.TestCase):
    pass

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
        print("ouput",output)
        loss   = torch.nn.functional.nll_loss(torch.squeeze(output), targets)
        return loss

    def test_torchmodel_1_get_gradients_method(self):
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
        
        torch_vector_grads = self.model.get_weights(return_type=TorchVector)
        for layer_name, values in torch_vector_grads.coefs.items():
            self.assertTrue(torch.all(values))
            self.assertTrue(torch.all(torch.isclose(values, grads[layer_name])))
            
        
    def test_torchmodel_2_get_gradients_method_failures(self):
        
        # test incorrect usage
        incorrect_return_types = (
            "incorrect usage",
            True,
            1234
        )
        for incorrect_return_type in incorrect_return_types:
            with self.assertRaises(FedbiomedModelError):
                self.model.get_gradients(return_type=incorrect_return_type)

    def test_torchmodel_3_get_weights(self):
        # test case where model_wweitghs is retunred as a dict
        model_weights = self.model.get_weights()
        
        
        for (layer, wrapped_model_weight) in model_weights.items():
            self.assertTrue(torch.all(torch.isclose(wrapped_model_weight, self.torch_model.get_parameter(layer))))
        # test case where model weigths is returned as a TorchVector
        torchvector_model_weights = self.model.get_weights(TorchVector)
        
        for (layer, wrapped_model_weight) in torchvector_model_weights.coefs.items():
            self.assertTrue(torch.all(torch.isclose(wrapped_model_weight, self.torch_model.get_parameter(layer))))

    def test_torchmodel_4_get_weights_failures(self):
        # test incorrect usage
        incorrect_return_types = (
            "incorrect usage",
            True,
            1234
        )
        
        for incorrect_return_type in incorrect_return_types:
            with self.assertRaises(FedbiomedModelError):
                self.model.get_weights(return_type=incorrect_return_type)
                
    def test_torchmodel_5_apply_updates_1(self):
        init_weights = copy.deepcopy(self.model.get_weights())
        
        updates = torch.nn.Linear(4, 1).state_dict()
        
        self.model.apply_updates(updates)
        updated_weights = self.model.get_weights()

        # checks
        for (layer, w), (_, updated_w) in zip(init_weights.items(), updated_weights.items()):
            self.assertFalse(torch.all(torch.isclose(w, updated_w)))
            self.assertTrue(torch.all(torch.isclose(updated_w, self.model.model.get_parameter(layer))))

    def test_torchmodel_5_apply_updates_2(self):
        init_weights = copy.deepcopy(self.model.get_weights(return_type=TorchVector))
        
        updates = torch.nn.Linear(4, 1).state_dict()
        
        self.model.apply_updates(TorchVector(updates))
        updated_weights = self.model.get_weights(return_type=TorchVector)

        # checks
        for (layer, w), (_, updated_w) in zip(init_weights.coefs.items(), updated_weights.coefs.items()):
            self.assertFalse(torch.all(torch.isclose(w, updated_w)))
            self.assertTrue(torch.all(torch.isclose(updated_w, self.model.model.get_parameter(layer))))
    
    def test_torchmodel_5_apply_updates_3_failures(self):
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
    
    def test_torchmodel_6_predict(self):
        data = torch.randn(1, 1,  4, requires_grad=True)
        
        tested_prediction = self.model.predict(data)
        
        ground_truth_prediction = self.model.model(data)
        
        self.assertIsInstance(tested_prediction, np.ndarray)
        self.assertListEqual(tested_prediction.tolist(), ground_truth_prediction.tolist())
        
    def test_torchmodel_7_training(self):
        self.model.init_training()
        
        #  before training, check values contained in `init_training` are the same as in model
        weights = self.model.get_weights()

        for (layer, w), ( init_w) in zip(weights.items(), self.model.init_params):
            self.assertTrue(torch.all(torch.isclose(w, init_w)))
        
        # mimic training by updating model weights
        # 1. training through torch optimizer
        self.torch_optim.zero_grad()
        loss = self.fake_training_step(self.data, self.targets)

        loss.backward()
        self.torch_optim.step()
        torch_update_weights = self.model.get_weights()
        # checks
        for (layer, w), ( init_w) in zip(torch_update_weights.items(), self.model.init_params):
            self.assertFalse(torch.all(torch.isclose(w, init_w)))

        # 2. training through declearn optimizer
        self.model.init_training()
        #  before training, check values contained in `init_training` are the same as in model
        weights = self.model.get_weights()

        for (layer, w), ( init_w) in zip(weights.items(), self.model.init_params):
            self.assertTrue(torch.all(torch.isclose(w, init_w)))
            
        self.model.model.zero_grad()
        loss = self.fake_training_step(self.data, self.targets)

        loss.backward()
        grads = self.model.get_weights(return_type=TorchVector)
        self.declearn_optim.apply_gradients(self.model, grads)
        
        declearn_optimized_model_weights = self.model.get_weights()
        # checks
        
        for (layer, w), w_init in zip(declearn_optimized_model_weights.items(), self.model.init_params):
            self.assertFalse(torch.all(torch.isclose(w, w_init)))

if __name__ == '__main__':  # pragma: no cover
    unittest.main()

# Test init params