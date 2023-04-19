import copy
import random
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch, Mock
from fedbiomed.common.exceptions import FedbiomedOptimizerError

import numpy as np
import torch.nn as nn
import torch
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
from declearn.optimizer import Optimizer as DecOptimizer
from declearn.optimizer.modules import (
        ScaffoldServerModule, ScaffoldClientModule, GaussianNoiseModule,
        YogiMomentumModule, L2Clipping, AdaGradModule, YogiModule)
from declearn.optimizer.regularizers import FedProxRegularizer, LassoRegularizer, RidgeRegularizer
from declearn.model.torch import TorchVector
from declearn.model.sklearn import NumpyVector

from fedbiomed.common.optimizers.generic_optimizers import NativeSkLearnOptimizer, NativeTorchOptimizer, DeclearnOptimizer, OptimizerBuilder
from fedbiomed.common.models import SkLearnModel, Model, TorchModel, BaseSkLearnModel
from fedbiomed.common.optimizers.optimizer import Optimizer as FedOptimizer


class TestDeclearnOptimizer(unittest.TestCase):

    def setUp(self) -> None:
        self._optim_wrapper = DeclearnOptimizer

        self._torch_model = nn.Linear(4,2)
        self._zero_model = copy.deepcopy(self._torch_model)
        # setting all coefficients of `zero_model` to 0
        for p in self._zero_model.parameters():
            p.data.fill_(0)
        self._sklearn_model_wrappers = (SkLearnModel(SGDClassifier),
                                        SkLearnModel(SGDRegressor))

        self._torch_model_wrappers = (TorchModel(self._torch_model),)
        self._torch_zero_model_wrappers = (TorchModel(self._zero_model),)
        
        self.modules = [
            ScaffoldServerModule(),
            GaussianNoiseModule(),
            YogiMomentumModule(),
            L2Clipping(),
            AdaGradModule(),
            YogiModule()]
        self.regularizers = [FedProxRegularizer(), LassoRegularizer(), RidgeRegularizer()]

    def tearDown(self) -> None:
        return super().tearDown()

    #@patch.multiple(BaseOptimizer, __abstractmethods__=set())  # disable base abstract to trigger errors if method(s) are not implemented
    def test_declearnoptimizer_01_init_invalid_model_arguments(self):

        correct_optimizers = (MagicMock(spec=FedOptimizer),
                              DecOptimizer(lrate=.1)
                                )
        incorrect_type_models = (None,
                                 True,
                                 np.array([1, 2, 3]),
                                 MagicMock(spec=BaseEstimator),
                                 nn.Module())

        for model in incorrect_type_models:
            for optimizer in correct_optimizers:
                with self.assertRaises(FedbiomedOptimizerError):
                    self._optim_wrapper(model, optimizer)

    def test_declearnoptimizer_02_init_invalid_optimizer_arguments(self):
        incorrect_optimizers = (None,
                                nn.Module(),
                                torch.optim.SGD(self._torch_model.parameters(), .01))
        correct_models = (
            MagicMock(spec=Model),
            MagicMock(spec=SkLearnModel)
        )

        for model in correct_models:
            for optim in incorrect_optimizers:

                with self.assertRaises(FedbiomedOptimizerError):
                    self._optim_wrapper(model, optim)


    def test_declearnoptimizer_03_step_method_1_TorchOptimizer(self):

        optim = FedOptimizer(lr=1)

        # initilise optimizer wrappers
        initialized_torch_optim_wrappers = (
            DeclearnOptimizer(copy.deepcopy(model), optim) for model in self._torch_zero_model_wrappers
            )

        fake_retrieved_grads = [
            {name: param for (name, param) in model.model.state_dict().items() }
            for model in self._torch_zero_model_wrappers
        ]

        fake_retrieved_grads = [TorchVector(grads) + 1 for grads in fake_retrieved_grads]
        # operation: do a SGD step with all gradients equal 1 and learning rate equals 1
        for torch_optim_wrapper, zero_model, grads in zip(initialized_torch_optim_wrappers,
                                                          self._torch_zero_model_wrappers,
                                                          fake_retrieved_grads):
            with patch.object(TorchModel, 'get_gradients') as get_gradients_patch:
                get_gradients_patch.return_value = grads.coefs

                torch_optim_wrapper.step()
                get_gradients_patch.assert_called_once()

            for (l, val), (l_ref, val_ref) in zip(zero_model.get_weights().items(), torch_optim_wrapper._model.get_weights().items()):

                self.assertTrue(torch.isclose(val - 1, val_ref).all())


    def test_declearnoptimizer_03_step_method_2_SklearnOptimizer(self):
        optim = FedOptimizer(lr=1)
        num_features = 4
        num_classes = 2

        # zero sklearn model weights
        for model in self._sklearn_model_wrappers:
            model.set_init_params({'n_features': num_features, 'n_classes': num_classes})
            model.model.eta0 = .1  # set learning rate to make sure it is different from 0
        initialized_sklearn_optim = (DeclearnOptimizer(copy.deepcopy(model), optim) for model in self._sklearn_model_wrappers)

        fake_retrieved_grads = [
            copy.deepcopy(model.get_weights()) for model in self._sklearn_model_wrappers
        ]
        fake_retrieved_grads = [NumpyVector(grads) + 1 for grads in fake_retrieved_grads]
        # operation: do a SGD step with all gradients equal 1 and learning rate equals 1
        for sklearn_optim_wrapper, zero_model, grads in zip(initialized_sklearn_optim,
                                                            self._sklearn_model_wrappers,
                                                            fake_retrieved_grads):
            with patch.object(BaseSkLearnModel, 'get_gradients') as get_gradients_patch:
                get_gradients_patch.return_value = grads.coefs
                sklearn_optim_wrapper.step()
                get_gradients_patch.assert_called()

            for (l, val), (l_ref, val_ref) in zip(zero_model.get_weights().items(),
                                                  sklearn_optim_wrapper._model.get_weights().items()):
                self.assertTrue(np.all(val - 1 == val_ref))  # NOTA: all val values are equal 0

    def test_declearnoptimizer_04_get_learning_rate(self):
        learning_rate = .12345
        optim = FedOptimizer(lr=learning_rate)

        # for torch specific optimizer wrapper

        for model in self._torch_model_wrappers:
            optim_wrapper = DeclearnOptimizer(model, optim)
            retrieved_lr = optim_wrapper.get_learning_rate()

            self.assertEqual(learning_rate, retrieved_lr[0])

        # for sckit-learn specific optimizer wrapper
        for model in self._sklearn_model_wrappers:
            optim_wrapper = DeclearnOptimizer(model, optim)
            retrieved_lr = optim_wrapper.get_learning_rate()

            self.assertEqual(learning_rate, retrieved_lr[0])

    def test_declearnoptimizer_05_aux_variables(self):
        learning_rate = .12345
        
        
        for model_wrappers in (self._torch_model_wrappers, self._sklearn_model_wrappers):
            for model in model_wrappers:
                optim = FedOptimizer(lr=learning_rate, modules = [ScaffoldServerModule()])
                optim_wrapper = DeclearnOptimizer(model, optim)
                empty_aux = {}
                optim_wrapper.set_aux(empty_aux)

                self.assertDictEqual(optim_wrapper.get_aux(), {})
                aux = { 'scaffold': 
                    {
                        'node-1': {'state': 1.},
                        'node-2': {'state': 2.},
                        'node-3': {'state': 3.}
                    }
                }

                optim_wrapper.set_aux(aux)
                
                collected_aux_vars = optim_wrapper.get_aux()
                expected_aux_vars = {
                    'scaffold':
                        {
                            'node-1': {'delta': -1.},
                            'node-2': {'delta': 0.},
                            'node-3': {'delta': 1.}
                        }
                }

                self.assertDictEqual(collected_aux_vars, expected_aux_vars)

                optim = FedOptimizer(lr=learning_rate, modules = [ScaffoldClientModule()])
                optim_wrapper = DeclearnOptimizer(model, optim)
                aux = {'scaffold': {'delta': 1.}}
                optim_wrapper.set_aux(aux)
                expected_aux = {'scaffold': {'state': 0.}}
                collected_aux = optim_wrapper.get_aux()
                self.assertDictEqual(expected_aux, collected_aux)
        
    def test_declearnoptimizer_06_states(self):
        def check_state(state: Dict[str, Any], learning_rate: float, w_decay: float, modules: List, regs: List, model):
            self.assertEqual(state['config']['lrate'], learning_rate)
            self.assertEqual(state['config']['w_decay'], w_decay)
            self.assertListEqual(state['config']['regularizers'], [(reg.name, reg.get_config()) for reg in regs])
            self.assertListEqual(state['config']['modules'], [(mod.name , mod.get_config()) for mod in modules])
            new_optim_wrapper = DeclearnOptimizer.load_state(model, state)
            self.assertDictEqual(new_optim_wrapper.save_state(), state)
            self.assertIsInstance(new_optim_wrapper.optimizer, FedOptimizer)

        learning_rate = .12345
        w_decay = .54321
        
        optim = FedOptimizer(lr=learning_rate, decay=w_decay)
        
        for model_wrappers in (self._torch_model_wrappers, self._sklearn_model_wrappers):
            for model in model_wrappers:
                optim_wrapper = DeclearnOptimizer(model, optim)
                state = optim_wrapper.save_state()
                
                check_state(state, learning_rate, w_decay, [], [], model)
        
        nb_tests = 10  # number of time the following test will be executed
        
        for model_wrappers in (self._torch_model_wrappers, self._sklearn_model_wrappers):
            for model in model_wrappers:
                for _ in range(nb_tests):
                    # test DeclearnOptimizer with random modules and regularizers 
                    selected_modules = random.sample(self.modules, random.randint(0, len(self.modules)))
                    selected_reg = random.sample(self.regularizers, random.randint(0, len(self.regularizers)))

                    optim = FedOptimizer(lr=learning_rate,
                                         decay=w_decay, 
                                         modules=selected_modules,
                                         regularizers=selected_reg)
                    optim_wrapper = DeclearnOptimizer(model, optim)
                    state = optim_wrapper.save_state()
                    
                    check_state(state, learning_rate, w_decay, selected_modules, selected_reg, model)

    def test_declearnoptimizer_05_declearn_optimizers_1_sklearnModel(self):
        # TODO: test here several declearn optimizers, regardless of the framework used
        nb_tests = 10  # number of time the following test will be executed
        pass
    
    def test_declearnoptimizer_06_declearn_scaffold_1_sklearnModel(self):
        # FIXME: this test is more a funcitonal test and should belong to trainingplan tests
        # test with one server and one node on a SklearnModel
        
        # this test was made to simulate a training with node and researcher with optimizer sending auxiliary variables
        # in addition to model parameters
        self.data = np.array([[1, 1, 1, 1,],
                              [1, 0, 1, 0],
                              [1, 1, 1, 1]])
        
        self.targets = np.array([[1], [0], [1], [1]])
        
        num_features = 4
        num_classes = 2
        grad_patch_value = 3
        researcher_lr, node_lr = .03, .5

        for model in self._sklearn_model_wrappers:
            researcher_optim = FedOptimizer(lr=researcher_lr, modules=[ScaffoldServerModule()])
            node_optim = FedOptimizer(lr=node_lr, modules=[ScaffoldClientModule()])
            # step 1: initializes and zeroes node model weights 
            # zero sklearn model weights
            model.set_init_params({'n_features': num_features, 'n_classes': num_classes})

            # step 2: sends auxiliary variables from researcher to nodes
            researcher_sklearn_optim_wrapper = DeclearnOptimizer(model, researcher_optim)
            node_model = copy.deepcopy(model)
            node_sklearn_optim_wrapper = DeclearnOptimizer(node_model, node_optim)
            aux_var = researcher_sklearn_optim_wrapper.get_aux()
            
            self.assertDictEqual(aux_var, {})  # test that aux_var are 0 for the first round

            for _ in range(3):  # simulates 3 rounds
                node_sklearn_optim_wrapper.set_aux(aux_var)
                node_model.set_weights(model.get_weights())  # simulate sending researcher model weights to node model weights 

                fake_retrieved_grads = copy.deepcopy(model.get_weights()) 
                fake_retrieved_grads = NumpyVector(fake_retrieved_grads) + grad_patch_value

                # step 3: performs (or simulates) a training on node side
                with node_sklearn_optim_wrapper.optimizer_processing():

                    with patch.object(BaseSkLearnModel, 'get_gradients') as get_gradients_patch:
                        get_gradients_patch.return_value = fake_retrieved_grads.coefs

                        # training process
                        node_sklearn_optim_wrapper.init_training()
                        model.train(self.data, self.targets, stdout=[])
                        node_sklearn_optim_wrapper.step()
                        # NOTA: due to patching `get_gradients` and since model weights are set to `0` at the begining, weights_t+1 equal weights_t - grad_patch_value * learning_rate
                
                # step 4: sends aux_var and node model params from node to researcher
                aux_var = node_sklearn_optim_wrapper.get_aux()
                aux_var = {'scaffold': {'node-1': aux_var['scaffold']}}
                
                model.set_weights(node_model.get_weights())  # simulate sending back node model weights to researcher model
                researcher_sklearn_optim_wrapper.set_aux(aux_var)

                # step 5: performs an update on researcher side
                researcher_sklearn_optim_wrapper.init_training()
                researcher_sklearn_optim_wrapper.step()

                # step 6: retrieves researcher model parameters and aux_var

                aux_var = researcher_sklearn_optim_wrapper.get_aux()
                aux_var = {k: v.get('node-1') for k, v in aux_var.items() if v.get('node-1')}

            final_aux_var = node_sklearn_optim_wrapper.get_aux()
            # according to scaffold module in declearn, delta should be equal to gradients  (ie fake_retrieved_grads)
            for (l, state), (_, val) in zip(final_aux_var['scaffold']['state'].coefs.items(), fake_retrieved_grads.coefs.items()):
                # check state value sent from node to researcher
                self.assertTrue(np.all(state == val))

    def test_declearnoptimizer_06_declearn_scaffold_2_torchModel(self):
        grad_patch_value = 3
        researcher_lr, node_lr = .03, .5
        data = torch.Tensor([[1,1,1,1],
                             [1,0,0,1]])
        targets = torch.Tensor([[1, 1]])

        loss_func = torch.nn.MSELoss()
        for model in self._torch_zero_model_wrappers:
            researcher_optim = FedOptimizer(lr=researcher_lr, modules=[ScaffoldServerModule()])
            node_optim = FedOptimizer(lr=node_lr, modules=[ScaffoldClientModule()])
             # step 2: sends auxiliary variables from researcher to nodes
            researcher_torch_optim_wrapper = DeclearnOptimizer(model, researcher_optim)
            node_model = copy.deepcopy(model)
            node_torch_optim_wrapper = DeclearnOptimizer(node_model, node_optim)
            aux_var = researcher_torch_optim_wrapper.get_aux()
            
            self.assertDictEqual(aux_var, {})  # test that aux_var are 0 for the first round
            
            for _ in range(3):  # simulates 3 rounds
                
                node_torch_optim_wrapper.set_aux(aux_var)
                node_model.set_weights(model.get_weights())  # simulate sending researcher model weights to node model weights 

                fake_retrieved_grads = copy.deepcopy(model.get_weights()) 
                fake_retrieved_grads = TorchVector(fake_retrieved_grads) + grad_patch_value

                # step 3: performs (or simulates) a training on node side
                node_model_params = node_model.get_weights()
                with patch.object(TorchModel, 'get_gradients') as get_gradients_patch:
                    get_gradients_patch.return_value = fake_retrieved_grads.coefs

                    # training process
                    node_torch_optim_wrapper.init_training()
                    node_torch_optim_wrapper.zero_grad()
                    output = model.model.forward(data)
                    loss = loss_func(output, targets)
                    loss.backward()
                    node_torch_optim_wrapper.step()
                    # NOTA: due to patching `get_gradients` and since model weights are set to `0` at the begining, weights_t+1 equal weights_t - grad_patch_value * learning_rate
                
                # step 4: sends aux_var and node model params from node to researcher
                aux_var = node_torch_optim_wrapper.get_aux()
                aux_var = {'scaffold': {'node-1': aux_var['scaffold']}}
                print("N MODEL", node_model.get_weights())
                
                model.set_weights(node_model.get_weights())  # simulate sending back node model weights to researcher model
                researcher_torch_optim_wrapper.set_aux(aux_var)

                # step 5: performs an update on researcher side
                researcher_torch_optim_wrapper.init_training()
                researcher_torch_optim_wrapper.zero_grad()
                researcher_torch_optim_wrapper.step()
                
                # step 6: retrieves researcher model parameters and aux_var
                aux_var = researcher_torch_optim_wrapper.get_aux()
                aux_var = {k: v.get('node-1') for k, v in aux_var.items() if v.get('node-1')}
                
                for (l, p_before), (_, p_after) in zip(node_model_params.items(), model.get_weights().items()):
                    self.assertTrue(torch.isclose(node_lr * (p_before - grad_patch_value) , p_after).all())
                    # model weights follows a progression p_t+1 = node_lr(p_t - grad_patch_value) due to SGD repeated several time
                
            final_aux_var = node_torch_optim_wrapper.get_aux()
            for (l, state), (_, val) in zip(final_aux_var['scaffold']['state'].coefs.items(), 
                                            fake_retrieved_grads.coefs.items()):
                self.assertTrue(torch.isclose(state, val).all())
            

            print("F", final_aux_var['scaffold']['state'].coefs)

    def test_declearnoptimizer_07_multiple_scaffold(self):
        pass
        
class TestTorchBasedOptimizer(unittest.TestCase):
    # make sure torch based optimizers does the same action on torch models - regardless of their nature
    def setUp(self):

        self._torch_model = (nn.Linear(4, 2),)
        self._fed_models = (TorchModel(model) for model in self._torch_model)
        self._zero_models = [copy.deepcopy(model) for model in self._torch_model]
        for model in self._zero_models:
            for p in model.parameters():
                p.data.fill_(0)

    def tearDown(self) -> None:
        return super().tearDown()

    def test_torchbasedoptimizer_01_zero_grad(self):
        declearn_optim = FedOptimizer(lr=.1)
        torch_optim_type = torch.optim.SGD

        for model in self._fed_models:
            # initialisation of declearn optimizer wrapper
            declearn_optim_wrapper = DeclearnOptimizer(model, declearn_optim)

            declearn_optim_wrapper.zero_grad()
            grads = declearn_optim_wrapper._model.get_gradients()

            for l, val in grads.items():
                self.assertTrue(torch.isclose(val, torch.zeros(val.shape)).all())

            # initialisation of native torch optimizer wrapper
            torch_optim = torch_optim_type(model.model.parameters(), .1)
            native_torch_optim_wrapper = NativeTorchOptimizer(model, torch_optim)
            native_torch_optim_wrapper.zero_grad()

            grads = native_torch_optim_wrapper._model.get_gradients()
            for l, val in grads.items():
                self.assertTrue(torch.isclose(val, torch.zeros(val.shape)).all())

    def test_torchbasedoptimizer_03_step(self):
        # check that declearn and torch optimizers give the same result
        declearn_optim = FedOptimizer(lr=1)
        torch_optim_type = torch.optim.SGD

        data = torch.Tensor([[1,1,1,1]])
        targets = torch.Tensor([[1, 1]])

        loss_func = torch.nn.MSELoss()

        for model in self._zero_models:
            model = TorchModel(model)
            # initialisation of declearn optimizer wrapper
            declearn_optim_wrapper = DeclearnOptimizer(model, declearn_optim)
            declearn_optim_wrapper.zero_grad()


            output = declearn_optim_wrapper._model.model.forward(data)

            loss = loss_func(output, targets)

            loss.backward()
            declearn_optim_wrapper.step()

            # initialisation of native torch optimizer wrapper
            torch_optim = torch_optim_type(model.model.parameters(), 1)
            native_torch_optim_wrapper = NativeTorchOptimizer(model, torch_optim)
            native_torch_optim_wrapper.zero_grad()

            output = native_torch_optim_wrapper._model.model.forward(data)

            loss = loss_func(output, targets)

            loss.backward()

            native_torch_optim_wrapper.step()

            # checks
            for (l, dec_optim_val), (l, torch_optim_val) in zip(declearn_optim_wrapper._model.get_weights().items(),
                                                                 native_torch_optim_wrapper._model.get_weights().items()):
                self.assertTrue(torch.isclose(dec_optim_val, torch_optim_val).all())

    def test_torchbasedoptimizer_04_invalid_methods(self):
        declearn_optim = FedOptimizer(lr=.1)

        for model in self._fed_models:
            # initialisation of declearn optimizer wrapper
            declearn_optim_wrapper = DeclearnOptimizer(model, declearn_optim)
            with self.assertRaises(FedbiomedOptimizerError):
                with declearn_optim_wrapper.optimizer_processing():
                    pass


class TestSklearnBasedOptimizer(unittest.TestCase):
    def setUp(self):
        self._sklearn_model = (SkLearnModel(SGDClassifier),
                                SkLearnModel(SGDRegressor))
    
    def tearDown(self):
        pass

    def test_sklearnbasedoptimizer_01_get_learning_rate(self):
        pass
    
    def test_sklearnbasedoptimizer_02_step(self):
        pass
   
    def test_sklearnbasedoptimizer_03_optimizer_processing(self):
        self.data = np.array([[1, 1, 1, 1,],
                              [1, 0,0, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 0]])
        
        self.targets = np.array([[1], [0], [1], [1]])
        learning_rate = .12345
        num_features = 4
        num_classes = 2
        for model in self._sklearn_model:
            # 1. test for declearn optimizers
            optim = FedOptimizer(lr=learning_rate) 
            optim_wrapper = DeclearnOptimizer(model, optim)
            model.set_init_params({'n_features': num_features, 'n_classes': num_classes})
            model.model.penality = None # disable penality
            init_optim_hyperparameters = copy.deepcopy(model.get_params())

            with optim_wrapper.optimizer_processing():
                disabled_optim_hyperparameters = model.get_params()
                model.train(self.data, self.targets)
                model_weights_before_step = copy.deepcopy(model.get_weights())
                optim_wrapper.step()
                model_weights_after = model.get_weights()
                for (l, grads), (l, w_before_step), (l, w_after) in zip(model.get_gradients().items(),
                                                                        model_weights_before_step.items(),
                                                                        model_weights_after.items()):
                    # check that only the declearn learning rate is used for training the model
                    self.assertTrue(np.all(np.isclose(w_after, w_before_step - learning_rate * grads)))
                    self.assertNotEqual(disabled_optim_hyperparameters, init_optim_hyperparameters)
                    
            self.assertDictEqual(init_optim_hyperparameters, model.get_params())

    def test_sklearnbasedoptimizer_04_invalid_method(self):
        # test that zero_grad raises error if model is sklearn

        for model in self._sklearn_model:

            optim = FedOptimizer(lr=.12345) 
            optim_wrapper = DeclearnOptimizer(model, optim)
            
            with self.assertRaises(FedbiomedOptimizerError):
                optim_wrapper.zero_grad()
            

class TestNativeTorchOptimizer(unittest.TestCase):
    pass
    # to be completed
class TestNativeSklearnOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self._sklearn_model_wrappers = (SkLearnModel(SGDClassifier),
                                        SkLearnModel(SGDRegressor))
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_nativesklearnoptimizer_01_optimizer_post_processing(self):
        
        for model in self._sklearn_model_wrappers:
            init_model_hyperparameters = model.get_params()
            
            optim_wrapper = NativeSkLearnOptimizer(model)
            with optim_wrapper.optimizer_processing():
                model_hyperparameters = model.get_params()
                self.assertDictEqual(init_model_hyperparameters, model_hyperparameters,
                                     'context manager has changed some model hyperparameters but should not when using NativeSklearnOptimizer')

            self.assertDictEqual(model_hyperparameters, model.get_params())

  
class TestOptimizerBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_models = (nn.Module(),
                             nn.Linear(4, 2),)
        self.sklearn_models = (SkLearnModel(SGDClassifier),
                               SkLearnModel(SGDRegressor),)
        
        
    def tearDown(self) -> None:
        return super().tearDown()
    
    
    def test_01_correct_build_optimizer(self):
        pass
    
    def test_02_get_parent_class(self):
        pass
    
    def test_03_unknowntrainingplan(self):
        for incorrect_training_plan_type in (None, 'unknown_training_plan',):
            for model_collection in (self.torch_models, self.sklearn_models,):
                for model in model_collection:
                    with self.assertRaises(FedbiomedOptimizerError):
                        OptimizerBuilder().build(incorrect_training_plan_type, model, None)


if __name__ == "__main__":
    unittest.main()
