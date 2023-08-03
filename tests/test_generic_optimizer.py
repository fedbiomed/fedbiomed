import copy
import random
from typing import Any, Dict, List, Tuple, Union
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
import declearn
from declearn.optimizer import Optimizer as DecOptimizer
from declearn.optimizer.modules import (
        ScaffoldServerModule, ScaffoldClientModule, GaussianNoiseModule,
        YogiMomentumModule, L2Clipping, AdaGradModule, YogiModule)
from declearn.optimizer.regularizers import FedProxRegularizer, LassoRegularizer, RidgeRegularizer
from declearn.model.torch import TorchVector
from declearn.model.sklearn import NumpyVector

from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.optimizers.generic_optimizers import NativeSkLearnOptimizer, NativeTorchOptimizer, DeclearnOptimizer, OptimizerBuilder
from fedbiomed.common.optimizers.optimizer import Optimizer as FedOptimizer
from fedbiomed.common.models import SkLearnModel, Model, TorchModel, BaseSkLearnModel


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

        # It could be nice to have several models in the tuple `torch_model_wrappers`
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

    def create_random_declearn_optimizer(self,
                                         learning_rate: float,
                                         w_decay: float,
                                         forces_mod_and_reg: bool = False) -> Tuple[DecOptimizer,
                                                                  List[declearn.optimizer.modules.OptiModule],
                                                                  List[declearn.optimizer.regularizers.Regularizer]]:
        """Creates random declearn optimizers, by picking random declearn `OptiModules` optimizers and regularizers,
        from self.modules and self.regularizers

        Args:
            learning_rate: learning rate passed into DecLearn optimizer
            w_decay: weight decay passed into DecLearn optimizer
            forces_mod_and_reg (bool, optional): forces declearn optimizer to have at least one OptiModule and Regularizer.
            Defaults to False.

        Returns:
            tuple of size 3 containing:
            - the declearn optimizer
            - the list of randomized OptiModules
            - the list of randomized Regularizers
        """
        min_modules = 0 if not forces_mod_and_reg else 1
        selected_modules = random.sample(self.modules, random.randint(min_modules, len(self.modules)))
        selected_reg = random.sample(self.regularizers, random.randint(min_modules, len(self.regularizers)))

        optim = FedOptimizer(lr=learning_rate,
                                decay=w_decay, 
                                modules=selected_modules,
                                regularizers=selected_reg)
        return optim, selected_modules, selected_reg

    def serializer(self, 
                   msg: Dict[str, Dict[str, Dict[str, Union[NumpyVector, TorchVector]]]]) -> Dict[str, Dict[str, Dict[str,Dict[str, Any]]]]:
            """Mimics the serialization module: converts nested declearn vector in nested dictionaries"""
            types = (NumpyVector, TorchVector)
            msg = {k: v.coefs if isinstance(v, types) else self.serializer(v) for k, v in msg.items()}
            return msg
        
    def deserializer(self, msg):
        """Mimics the deserialization module: converts model parameters into declearn Vector objects"""
        types = (np.ndarray, torch.Tensor)

        msg =  {k: declearn.model.api.Vector.build(v) if any([isinstance(q, types) for p, q in v.items()]) else self.deserializer(v) for k, v in msg.items()}
        return msg
    
    def perform_sklearn_training_node_side(self, node_id: str,
                                           node_model: SkLearnModel,
                                           node_optim_w: DeclearnOptimizer,
                                           global_model_weights: Dict,
                                           aux_var: Dict,
                                           data: np.ndarray,
                                           target: np.ndarray) -> Dict:
        """Performs a optimizer step on node side for sklearn models, given a dictionary
        of auxiliary variables, when using a Declearn Optimizer. Useful for testing the
        correct setting of auxiliary variables.

        Args:
            node_id: id of the node set for the test
            node_model: node model
            node_optim_w: declearn optimizer warpper for Node
            global_model_weights: global model weights, obtained after aggregation (should be sent from Researcher to Node)
            aux_var: auxiliary  variables, in a dictionary. All elements of the aux_var are human readable.
            data: data used to train the node model.
            target: target data used to train node model

        Returns:
            Dict: auxiliary  variables for node to be sent to researcher.
        """
        if aux_var:
            aux_var = self.deserializer(aux_var)
            aux_var = {k: v[node_id] for k, v in aux_var.items() if v.get(node_id)}
        node_optim_w.set_aux(copy.deepcopy(aux_var))
        node_model.set_weights(global_model_weights)
        
        # performs (or simulates) a training on node side
        with node_optim_w.optimizer_processing():
            # training process
            node_optim_w.init_training()
            node_model.train(data, target, stdout=[])
            node_optim_w.step()

        # collect auxiliary  variables
        aux_var = node_optim_w.get_aux()
        # serialization: convert everything from declearn Vector to dictionaries
        
        aux_var = self.serializer(aux_var)
        aux_var = {'scaffold': {node_id: aux_var['scaffold']}}

        return aux_var
    
    def perform_torch_training_node_side(self, node_id: str,
                                         node_model: TorchModel,
                                         node_optim_w: DeclearnOptimizer,
                                         global_model_weights: Dict,
                                         aux_var: Dict,
                                         data: torch.Tensor,
                                         targets: torch.Tensor,
                                         loss_func: torch.nn) -> Dict:
        """Performs a optimizer step on node side for Torch models, given a dictionary
        of auxiliary variables, when using a Declearn Optimizer. Useful for testing
        the correct setting of auxiliary variables.

        Args:
            node_id: node_id
            node_model: torch node model that will be optimized
            node_optim_w: declearn optimizer wrapper for Node
            global_model_weights: aggregated model parameters (sent form Researcher to Nodes)
            aux_var: auxiliary  variables sent from Researcher to Nodes
            data: data to be used for training the model
            targets: data to be used for training the model
            loss_func: loss function to be used for optimization.

        Returns:
            Dict: auxiliary  variables to be sent back from Node to Researcher
        """

        if aux_var:
            aux_var = self.deserializer(aux_var)
            aux_var = {k: v[node_id] for k, v in aux_var.items() if v.get(node_id)}
        node_optim_w.set_aux(copy.deepcopy(aux_var))
        node_model.set_weights(global_model_weights)
        
        # performs (or simulates) a training on node side
        node_optim_w.init_training()
        node_optim_w.zero_grad()
        output = node_model.model.forward(data)
        loss = loss_func(output, targets)
        loss.backward()
        node_optim_w.step()

        # collect auxiliary  variables
        aux_var = node_optim_w.get_aux()
        # serialization: convert everything from declearn Vector to dictionaries
        
        aux_var = self.serializer(aux_var)
        aux_var = {'scaffold': {node_id: aux_var['scaffold']}}

        return aux_var

    def perform_training_researcher_side(self,
                                         optim_w: DeclearnOptimizer,
                                         global_model_weights: Dict,
                                         aggr_model_weights: Dict,
                                         aux_var: Dict) -> Dict:
        """Performs/simulates an optimizer update on Researcher side, given auxiliary  variables.

        Args:
            optim_w: optimizer wrapper for Researcher
            global_model_weights: dictionary that contains all Nodes models that need to be aggregated
            aggr_model_weights: previous aggregated model weights. Graidents will be computed a follows: 
                aggr_model_weights - global_model_weights
            aux_var: auxialiary variables sent by the Nodes to the Researcher

        Returns:
            Dict: auxiliary variables to be sent back from Researcher to Nodes
        """
        # serialization: convert everything from dictionaries to declearn Vector
        # TODO: use serializer once it has been merged
        aux_var = self.deserializer(aux_var)

        optim_w.set_aux(aux_var)

        # performs an update on researcher side
        global_model_weights = declearn.model.api.Vector.build(global_model_weights)
        updates = declearn.model.api.Vector.build(aggr_model_weights) - global_model_weights
        optim_w.optimizer.step(updates, global_model_weights)
        
        # retrieves researcher model parameters and aux_var
        aux_var = optim_w.get_aux()

        aux_var = self.serializer(aux_var)

        return aux_var

    def compute_delta_from_nodes_state(self,
                                       aux_var: Dict,
                                       global_model_weigths: Dict) -> Dict:
        """Used for testing delta scaffold computation for scaffold tests (auxiliary variable to be sent
        from Node to Researcher).
        
        Args:
            aux_var: auxiliary variables sent from Nodes to Researcher, that should contain `scaffold` entry,
                containing a `state` entry
            global_model_weights: dictionary that contains model global weights. Used only to get the model layers.
        
        Returns:
            Dict: dictionary containing delats for each Node (mapping node-id to its delta)
        """
        node_ids = list(aux_var['scaffold'])
        deltas = {node_id: {} for node_id in node_ids}
        for layer in global_model_weigths:
            
            sum_state = sum(aux_var['scaffold'][k]['state'][layer] for k in aux_var['scaffold'] )
            sum_state /= len(aux_var['scaffold'])
            delta = {}
            for node_id in node_ids:

                delta[layer] = aux_var['scaffold'][node_id]['state'][layer] - sum_state 
                deltas[node_id].update(delta)
        return deltas
    
    # -------- TESTS ------------------------------------
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
                # computation performed to get deltas: node_state -  sum(node_states) / nb_nodes_involved
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
                    optim, selected_modules, selected_reg = self.create_random_declearn_optimizer(learning_rate, w_decay)
                    optim_wrapper = DeclearnOptimizer(model, optim)
                    state = optim_wrapper.save_state()
                    
                    check_state(state, learning_rate, w_decay, selected_modules, selected_reg, model)

    def test_declearnoptimizer_05_declearn_optimizers_1_sklearnModel(self):
        # performs a number of optimization steps with randomly created optimizers
        # FIXME: nothing is being asserted here...
        nb_tests = 10  # number of time the following test will be executed
        learning_rate = .12345
        w_decay = .54321
        
        data = np.array([[1, 1, 1, 1,],
                        [1, 0, 1, 0],
                        [1, 1, 1, 1]])
        
        targets = np.array([[1], [0], [1], [1]])
        optim, modules , regs = self.create_random_declearn_optimizer(learning_rate, w_decay)
        for _ in range(nb_tests):
            for model in self._sklearn_model_wrappers:
                optim_w = DeclearnOptimizer(model, optim)
                model.set_init_params({'n_features': 4, 'n_classes': 2})
                
                with optim_w.optimizer_processing():
                    optim_w.init_training()
                    model.train(data, targets)
                    optim_w.step()

    def test_declearnoptimizer_05_declearn_optimizers_2_torchmodel(self):
        # performs a number of optimization steps with randomly created optimizers
        # FIXME: nothing is being asserted here...
        data = torch.Tensor([[1,1,1,1],
                             [1,0,0,1]])
        targets = torch.Tensor([[1, 1], [0, 1]])
        
        learning_rate = .12345
        w_decay = .54321
        optim, modules , regs = self.create_random_declearn_optimizer(learning_rate, w_decay)
        loss_func = torch.nn.MSELoss()
        nb_tests = 10
        for _ in range(nb_tests):
            for model in self._torch_model_wrappers:
                optim_w = DeclearnOptimizer(model, optim)

                optim_w.init_training()
                optim_w.zero_grad()
                output = model.model.forward(data)
                loss = loss_func(output, targets)
                loss.backward()
                optim_w.step()

    
    def test_declearnoptimizer_06_declearn_scaffold_1_sklearnModel(self):
        # FIXME: this test is more a funcitonal test and should belong to trainingplan tests
        # test with one server and one node on a SklearnModel
        
        # this test was made to simulate a training with node and researcher with optimizer sending auxiliary variables for scaffold
        # in addition to model parameters
        # it tests:
        # - correct interfacing between Nodes auxiiliary variables and Researcher auxiliary variables
        # - correct update of Node weights wrt deltas (auxiliary variable for scaffold sent to Nodes)
        # - correct computation of Nodes states (auxiliary variable sent back to Researcher)
        # - correct properties of Nodes deltas

        data = np.array([[1, 1, 1, 1,],
                              [1, 0, 1, 0],
                              [1, 1, 1, 1]])
        data2 = np.array([[1, 0, 1, 0],
                              [1, 1, 1, 1,],
                              [1, 1, 1, 1]])
        
        targets = np.array([[1], [0], [1], [1]])
        
        num_features = 4
        num_classes = 2
        # node_scenarios is a tuple containing for each test a dictionary specifying:
        # - nodes_id (ie the simulated nodes taking part in the training)
        # - data: dictionry that maps node_id to the dataset used for training
        # - target: dictionary that map node_id to the target dataset used for training
        # here we have one test scenario invoving only one node first, then another one involving 3 nodes
        nodes_scenarios = ({'nodes': ('node-1',),
                            'data': {'node-1': data},
                            'target': {'node-1': targets}},
                           {'nodes': ('node-1', 'node-2', 'node-3',),
                            'data': {'node-1': data,
                                     'node-2': data,
                                     'node-3': data2},
                            'target': {'node-1':targets,
                                       'node-2': targets,
                                       'node-3': targets}},)
        researcher_lr, node_lr = .03, .5
        for nodes_scenario in nodes_scenarios:
            for model in self._sklearn_model_wrappers:
                researcher_optim = FedOptimizer(lr=researcher_lr, modules=[ScaffoldServerModule()])

                # step 1: initializes and zeroes node model weights 
                # zero sklearn model weights
                model.set_init_params({'n_features': num_features, 'n_classes': num_classes})
                node_optims = {node_id: FedOptimizer(lr=node_lr, modules=[ScaffoldClientModule()]) for node_id in nodes_scenario['nodes']}

                # step 2: sends auxiliary variables from researcher to nodes
                researcher_sklearn_optim_wrapper = DeclearnOptimizer(model, researcher_optim)
                node_models = {node_id: copy.deepcopy(model) for node_id in nodes_scenario['nodes']}
                node_sklearn_optim_wrappers = {k: DeclearnOptimizer(node_models[k], node_optims[k]) for k in nodes_scenario['nodes']}

                aux_var = researcher_sklearn_optim_wrapper.get_aux()
                
                self.assertDictEqual(aux_var, {})  # test that aux_var are 0 for the first round

                for _ in range(5):  # simulates 5 rounds

                    # step 3: performs (or simulates) a training on node side
                    aux_var_collections = {}
                    for idx, node_id in enumerate(nodes_scenario['nodes']):

                        node_aux_var = self.perform_sklearn_training_node_side(node_id,
                                                                               node_models[node_id],
                                                                               node_sklearn_optim_wrappers[node_id],
                                                                               model.get_weights(),
                                                                               aux_var,
                                                                               nodes_scenario['data'][node_id], 
                                                                               nodes_scenario['target'][node_id])

                        for k, v in node_models[node_id].get_gradients().items():
                            # ------ CHECKS
                            # check that states equals to the gradients in the very specific setting where we are performing
                            # only one iteration over data (so step = 1 and there is no gradient accumulation)
                            self.assertTrue(np.array_equal(v, node_aux_var['scaffold'][node_id]['state'][k]))
                            # check weights are updated according to delta values
                            if aux_var:
                                self.assertTrue(np.array_equal((aux_var['scaffold'][node_id]['delta'][k] - v) * node_lr, node_models[node_id].get_weights()[k]))

                        if idx == 0:
                            aux_var_collections = {k: {} for k in node_aux_var}
                        for k in node_aux_var:
                            aux_var_collections[k].update(node_aux_var[k])


                    # step 4: perform an update on Researcher side
                    aux_var = self.perform_training_researcher_side(researcher_sklearn_optim_wrapper,
                                                               model.get_weights(), model.get_weights(), aux_var_collections)

                # final checks
                deltas = self.compute_delta_from_nodes_state(aux_var_collections, model.get_weights())

                # final check 1. test that sum(delta_nodeid) = 0
                for  (k, v) in model.get_weights().items():
                    avg = np.zeros_like(v)
                    for node_id in nodes_scenario['nodes']:
                        avg += aux_var['scaffold'][node_id]['delta'][k]
                    self.assertTrue(np.isclose(avg, np.zeros_like(avg)).all())
                    
                # final check 2. test correct computation of deltas
                for node_id in nodes_scenario['nodes']:
                    for  (k, v) in model.get_weights().items():
                        self.assertTrue(np.array_equal(deltas[node_id][k], aux_var['scaffold'][node_id]['delta'][k]))

    def test_declearnoptimizer_06_declearn_scaffold_2_torchModel(self):
        # this test was made to simulate a training with node and researcher with optimizer sending auxiliary variables for scaffold
        # in addition to model parameters
        # it tests:
        # - correct interfacing between Nodes auxiiliary variables and Researcher auxiliary variables
        # - correct update of Node weights wrt deltas (auxiliary variable for scaffold sent to Nodes)
        # - correct computation of Nodes states (auxiliary variable sent back to Researcher)
        # - correct properties of Nodes deltas
        researcher_lr, node_lr = .03, .5
        data = torch.Tensor([[1, 1, 1 ,1],
                             [1, 0, 0, 1], 
                             [1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [1, 0, 0, 1]])
        data2 = torch.Tensor([[1,1,1,1],
                             [1,1,0,1], 
                             [1, 1, 0, 0],
                             [0, 0, 1, 0],
                             [1, 0, 0, 1]])
        targets = torch.Tensor([[1, 1], [1, 0], [1,1], [0,0], [0,1]])

        # node_scenarios is a tuple containing for each test a dictionary specifying:
        # - nodes_id (ie the simulated nodes taking part in the training)
        # - data: dictionry that maps node_id to the dataset used for training
        # - target: dictionary that map node_id to the target dataset used for training
        # here we have one test scenario invoving only one node first, then another one involving 3 nodes
        nodes_scenarios = (
            {
                'nodes': ('node-1',),
                'data': {'node-1': data},
                'target': {'node-1': targets}
            },
            {
                'nodes': ('node-1', 'node-2', 'node-3',),
                'data':{
                    'node-1': data,
                    'node-2': data,
                    'node-3': data2
                },
                
                'target': {
                    'node-1': targets,
                    'node-2': targets, 
                    'node-3': targets
                    }
            }
        )
        loss_func = torch.nn.MSELoss()
        for nodes_scenario in nodes_scenarios:
            for model in self._torch_model_wrappers:
                researcher_optim = FedOptimizer(lr=researcher_lr, modules=[ScaffoldServerModule()])
                node_optims = {node_id: FedOptimizer(lr=node_lr, modules=[ScaffoldClientModule()]) for node_id in nodes_scenario['nodes']}
                # step 2: sends auxiliary variables from researcher to nodes
                researcher_torch_optim_wrapper = DeclearnOptimizer(model, researcher_optim)

                node_models = {node_id: copy.deepcopy(model) for node_id in nodes_scenario['nodes']}
                node_torch_optim_wrappers = {k: DeclearnOptimizer(node_models[k], node_optims[k]) for k in nodes_scenario['nodes']}
                aux_var = researcher_torch_optim_wrapper.get_aux()
                self.assertDictEqual(aux_var, {})  # test that aux_var are 0 for the first round
                
                for _ in range(5):  # simulates 5 rounds

                    # step 3: performs (or simulates) a training on node side
                    aux_var_collections = {}
                    for idx, node_id in enumerate(nodes_scenario['nodes']):
                        copy_model_w = copy.deepcopy(model.get_weights())  # we copy here model weights (for testing only)

                        node_aux_var = self.perform_torch_training_node_side(node_id,
                                                            node_models[node_id],
                                                            node_torch_optim_wrappers[node_id],
                                                            model.get_weights(),
                                                            aux_var,
                                                            nodes_scenario['data'][node_id], 
                                                            nodes_scenario['target'][node_id],
                                                            loss_func)
                        v = node_torch_optim_wrappers[node_id].optimizer._optimizer.modules[0].delta

                        for k, v in node_models[node_id].get_gradients().items():
                            # ------ CHECKS
                            # check that states equals to the griadients in the very specific we are performing
                            # only one iteration over data (so step = 1 and there is no gradient accumulation)
                            self.assertTrue(torch.isclose(v, node_aux_var['scaffold'][node_id]['state'][k]).all())
                            # check weights are updated according to delta values
                            if aux_var:
                                self.assertTrue(torch.isclose(copy_model_w[k] - (v - aux_var['scaffold'][node_id]['delta'][k] ) * node_lr, node_models[node_id].get_weights()[k]).all())

                        if idx == 0:
                            aux_var_collections = {k: {} for k in node_aux_var}
                        for k in node_aux_var:
                            aux_var_collections[k].update(node_aux_var[k])


                    # step 4: perform a last update on Researcher side
                    aux_var = self.perform_training_researcher_side(researcher_torch_optim_wrapper,
                                                               model.get_weights(), model.get_weights(), aux_var_collections)

                # final checks
                deltas = self.compute_delta_from_nodes_state(aux_var_collections, model.get_weights())
                # final check 1. test that sum(delta_nodeid) = 0
                for  (k, v) in model.get_weights().items():
                    zero_vector = torch.zeros(v.shape)
                    avg = copy.deepcopy(zero_vector)
                    for node_id in nodes_scenario['nodes']:
                        avg += aux_var['scaffold'][node_id]['delta'][k]

                    self.assertTrue(torch.isclose(avg, zero_vector, atol=1e-5).all())
                # final check 2. test correct computation of deltas
                for node_id in nodes_scenario['nodes']:
                    for  (k, v) in model.get_weights().items():
                        self.assertTrue(torch.isclose(deltas[node_id][k], aux_var['scaffold'][node_id]['delta'][k]).all())

    def test_declearnoptimizer_07_multiple_scaffold(self):
        # the goal of this test is to check that user will get error if specifying non sensical 
        # Optimizer when passing both `ScaffoldServerModule()`and `ScaffoldClientModule()` modules

        researcher_lr, node_lr = .03, .5
        data = torch.Tensor([[1,1,1,1],
                             [1,0,0,1]])
        targets = torch.Tensor([[1, 1]])

        loss_func = torch.nn.MSELoss()
        for model in self._torch_zero_model_wrappers:
            incorrect_optim = FedOptimizer(lr=researcher_lr, modules=[ScaffoldServerModule(), ScaffoldClientModule()])  # Non sensical!!!
            incorrect_optim_w = DeclearnOptimizer(model, incorrect_optim)
            incorrect_optim_w.init_training()
            incorrect_optim_w.zero_grad()
            output = model.model.forward(data)
            loss = loss_func(output, targets)
            loss.backward()
            incorrect_optim_w.step()
            
            incorrect_state = incorrect_optim_w.get_aux()
            optim = FedOptimizer(lr=node_lr, modules=[ScaffoldClientModule()])

            correct_optim_w = DeclearnOptimizer(model, optim)
            aux_var = {'scaffold': {'node-1': incorrect_state['scaffold']}}
            with self.assertRaises(FedbiomedOptimizerError):
                correct_optim_w.set_aux(aux_var)


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
            # first check that element of model are non zeros
            grads = model.get_gradients()
            for l, val in grads.items():
                self.assertFalse(torch.isclose(val, torch.zeros(val.shape)).all())
            # initialisation of declearn optimizer wrapper
            dec_model = copy.deepcopy(model)
            declearn_optim_wrapper = DeclearnOptimizer(dec_model, declearn_optim)

            declearn_optim_wrapper.zero_grad()
            grads = dec_model.get_gradients()

            for l, val in grads.items():
                self.assertTrue(torch.isclose(val, torch.zeros(val.shape)).all())

            # initialisation of native torch optimizer wrapper
            torch_model = copy.deepcopy(model)
            torch_optim = torch_optim_type(torch_model.model.parameters(), .1)
            native_torch_optim_wrapper = NativeTorchOptimizer(torch_model, torch_optim)
            native_torch_optim_wrapper.zero_grad()

            grads = torch_model.get_gradients()
            for l, val in grads.items():
                self.assertTrue(torch.isclose(val, torch.zeros(val.shape)).all())

    def test_torchbasedoptimizer_02_step(self):
        # check that declearn and torch plain SGD optimization step give the same result
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

    def test_torchbasedoptimizer_03_invalid_methods(self):
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

        self.data = np.array([[1, 1, 1, 1,],
                              [1, 0,0, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 0]])
        
        self.targets = np.array([[1], [0], [1], [1]])
    
    def tearDown(self):
        pass

    def test_sklearnbasedoptimizer_01_step(self):
        # tests that a plain gradient descent performed by native sklearn optimizer and
        # using declearn optimizer gives the same results
        random_seed = 1234
        learning_rate = .1234
        
        for sk_model in self._sklearn_model:
            # native sklearn
            sk_model.set_params(random_state=random_seed,
                                eta0=learning_rate,
                                penalty=None,
                                learning_rate='constant')
            sk_model.set_init_params({'n_features': 4, 'n_classes': 2})
            sk_model_native = copy.deepcopy(sk_model)

            sk_optim_w = NativeSkLearnOptimizer(sk_model_native, None)
            with sk_optim_w.optimizer_processing():
                sk_optim_w.init_training()
                sk_model_native.train(self.data, self.targets)
                sk_optim_w.step()
            
            # sklearn with declearn optimizers
            sk_model_declearn = copy.deepcopy(sk_model)
            self.assertDictEqual(sk_model_declearn.get_params(), sk_model_native.get_params())  # if this is not passing, test will fail
            
            dec_optim_w = DeclearnOptimizer(sk_model_declearn, FedOptimizer(lr=learning_rate))
            with dec_optim_w.optimizer_processing():
                dec_optim_w.init_training()
                sk_model_declearn.train(self.data, self.targets)
                dec_optim_w.step()

            for (k,v_ref), (k, v) in zip(sk_model_native.get_weights().items(), sk_model_declearn.get_weights().items()):
                self.assertTrue(np.all(np.isclose(v, v_ref)))

    def test_sklearnbasedoptimizer_02_optimizer_processing(self):
        
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

    def test_sklearnbasedoptimizer_03_invalid_method(self):
        # test that zero_grad raises error if model is sklearn

        for model in self._sklearn_model:

            optim = FedOptimizer(lr=.12345) 
            optim_wrapper = DeclearnOptimizer(model, optim)
            
            with self.assertRaises(FedbiomedOptimizerError):
                optim_wrapper.zero_grad()


class TestNativeTorchOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_models = (nn.Sequential(
                             nn.Conv2d(1,20,5),
                             nn.ReLU(),
                             nn.Conv2d(20,64,5),
                             nn.ReLU()
                            ),
                             nn.Linear(4,2),)
        self.torch_optimizers = (torch.optim.SGD,
                                 torch.optim.Adam,
                                 torch.optim.Adagrad
                                 )

    def test_nativetorchoptimizer_01_step(self):
        # check that pytorch optimizer are called when calling `NativeTorchOptimizer.step()`
        learning_rate = .12345
        for model in self.torch_models:
            for optim_func in self.torch_optimizers:
                optim = optim_func(model.parameters(), lr=learning_rate)
                optim_w = NativeTorchOptimizer(TorchModel(model), optim)
                
                with patch.object(optim_func, 'step') as step_patch:
                    optim_w.step()
                step_patch.assert_called_once()
    
    def test_nativetorchoptimizer_02_getlearningrate_1(self):
        """test_torch_nn_08_get_learning_rate: test we retrieve the appropriate
        learning rate
        """
        # first test wih basic optimizer (eg without learning rate scheduler)

        model = TorchModel(torch.nn.Linear(2, 3))
        lr = .1
        dataset = torch.Tensor([[1, 2], [1, 1], [2, 2]])
        target = torch.Tensor([1, 2, 2])
        optimizer = NativeTorchOptimizer(model, torch.optim.SGD(model.model.parameters(), lr=lr))
        
        lr_extracted = optimizer.get_learning_rate()
        self.assertDictEqual(lr_extracted, {k: lr for k,_ in model.model.named_parameters()})
        
        # then test using a pytorch scheduler
        scheduler = LambdaLR(optimizer.optimizer, lambda e: 2*e)
        # this pytorch scheduler increase earning rate by twice its previous value
        for e, (x,y) in enumerate(zip(dataset, target)):
            # training a simple model in pytorch fashion
            # `e` represents epoch
            out = model.model.forward(x)
            optimizer.zero_grad()
            loss = torch.mean(out) - y
            loss.backward()
            optimizer.step()
            scheduler.step()

            # checks
            lr_extracted = optimizer.get_learning_rate()
            self.assertDictEqual(lr_extracted, {k: lr * 2 * (e+1) for k,_ in model.model.named_parameters()})

    def test_nativetorchoptimizer_03_getlearningrate_2(self):
        """test_nativetorchoptimizer_03_getlearningrate_2: test the method but with a more complex model,
        that possesses different learning rate per block / layers
        """

        lr_1,  lr_block, lr = .1, .2, .3

        class ComplexModel(nn.Module):
            """a complex model with batchnorms (trainable layers that doesnot requoere learning rates)
            data input example for using such model
            ```
            batch_size = 5
            model = ComplexModel()
            data = torch.rand(batch_size,1, 25,25) 
            model.forward(data)
            ```
            """
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.relu = nn.functional.relu
                self.conv2 = nn.Conv2d(20, 20, 5)
                self.block = nn.Sequential(nn.Linear(20 * 17 *17, 10),
                                        nn.BatchNorm1d(10),
                                        nn.Linear(10, 5))
                self.bn = nn.BatchNorm1d(5)
                self.upsampler = nn.Upsample(scale_factor=2)
                self.classifier = nn.Linear(10, 2)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = x.reshape(-1, 20 * 17 *17)
                x = self.block(x)
                x = self.bn(x)
                x = torch.unsqueeze(x, dim=0)
                x = torch.unsqueeze(x, dim=0)
                x = self.upsampler(x)
                return self.classifier(x)

        model = ComplexModel()

        opt = torch.optim.SGD([
            {'params': model.block.parameters(), 'lr': lr_block},
            {'params': model.conv1.parameters(), 'lr': lr_1},
            {'params': model.conv2.parameters()},
            {'params': model.classifier.parameters()},
            {'params': model.upsampler.parameters()},
            {'params': model.bn.parameters()}
            ], lr=lr)

        t_model = TorchModel(model)
        t_opt = NativeTorchOptimizer(t_model, opt)
        # checks
        lr_extracted = t_opt.get_learning_rate()
        model_layers_names = sorted([k for k, _ in model.named_parameters()])
        self.assertListEqual(sorted(list(lr_extracted.keys())), model_layers_names)

        # check that learning rates are extracted correctly according to the model layer names
        for k, v in model.named_parameters():
            if 'block' in k:
                self.assertEqual(lr_extracted[k], lr_block)
            elif 'conv1' in k:
                self.assertEqual(lr_extracted[k], lr_1)
            else:
                self.assertEqual(lr_extracted[k], lr)

        # check that the extracted learning rates
        self.assertGreaterEqual(len(t_model.model.state_dict()), len(lr_extracted))


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
        
        torch_model = nn.Linear(4, 2)
        self.native_torch_optimizers = (torch.optim.SGD(torch_model.parameters(), lr=.12345),
                                       torch.optim.Adam(torch_model.parameters(), lr=.12345),
                                       torch.optim.Adagrad(torch_model.parameters(), lr=.12345),)
        
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

    def create_random_fedoptimizer(self, lr: float = .12345, w_decay: float = .54321) -> FedOptimizer:
        selected_modules = random.sample(self.modules, random.randint(0, len(self.modules)))
        selected_reg = random.sample(self.regularizers, random.randint(0, len(self.regularizers)))

        optim = FedOptimizer(lr=lr,
                             decay=w_decay, 
                             modules=selected_modules,
                             regularizers=selected_reg)
        return optim

    def test_01_correct_build_optimizer(self):
        optim_builder = OptimizerBuilder()
        nb_tests = 10
        random_declearn_optim = [self.create_random_fedoptimizer() for _ in range(nb_tests)]
        # check that NativeTorchOptimizer and DeclearnOptimizer are correclty built 
        for torch_model in self.torch_models:

            for optim in self.native_torch_optimizers:
                optim_wrapper = optim_builder.build(TrainingPlans.TorchTrainingPlan, TorchModel(torch_model), optim)
                self.assertIsInstance(optim_wrapper, NativeTorchOptimizer)
            for optim in random_declearn_optim:
                optim_wrapper = optim_builder.build(TrainingPlans.TorchTrainingPlan, TorchModel(torch_model), optim)
                self.assertIsInstance(optim_wrapper, DeclearnOptimizer)
        
        # check that NativeSklearnOptimizer and DeclearnOptimizer are correctly built
        for sk_model in self.sklearn_models:
            optim_wrapper = optim_builder.build(TrainingPlans.SkLearnTrainingPlan, sk_model, None)
            self.assertIsInstance(optim_wrapper, NativeSkLearnOptimizer)
            for optim in random_declearn_optim:
                optim_wrapper = optim_builder.build(TrainingPlans.SkLearnTrainingPlan, sk_model, optim)
                self.assertIsInstance(optim_wrapper, DeclearnOptimizer)

    def test_02_get_parent_class(self):
        def check_type(obj, parent_obj):
            """Tests that function `get_parent_class` retruns 
            the appropriate type of the highest parent class (just after `object`)

            Args:
                obj : sub-calss or class of the object from which to guess the parent class type
                parent_obj : highest parent class from which `obj` object has been built

            Raises:
                AssertionError: raised if function `get_parent_class` doesnot return the expected type
            """
            res = optim_builder.get_parent_class(obj)
            self.assertEqual(res, type(parent_obj))

        class A:
            pass
        
        class B(A):
            pass
        
        class C(A):
            pass
        
        class D(B, C):
            pass
        
        class E(C, A):
            pass
        
        objects = [A, B, C, D, E]
        
        optim_builder = OptimizerBuilder()
        # test with `object`
        check_type(object, object)

        # test with `None`
        res = optim_builder.get_parent_class(None)
        self.assertEqual(res, None)
        
        # test  with dummy objects
        for obj in objects:
            check_type(obj, A)
            
        # test with torch optimizers
        for torch_optim in self.native_torch_optimizers:
            res = optim_builder.get_parent_class(torch_optim)
            self.assertEqual(res, torch.optim.Optimizer)
    
    def test_03_buildfailure_unknowntrainingplan(self):
        for incorrect_training_plan_type in (None, 'unknown_training_plan',):
            for model_collection in (self.torch_models, self.sklearn_models,):
                for model in model_collection:
                    with self.assertRaises(FedbiomedOptimizerError):
                        OptimizerBuilder().build(incorrect_training_plan_type, model, None)

    def test_04_buildfailure_trainingplan_model_mismatches(self):
        optim_builer = OptimizerBuilder()
        model = MagicMock(spec=SkLearnModel)

        # for SkLearnTrainingPlan
        for coll in (self.native_torch_optimizers, ("unknown", object,),):
            for item in coll:
        
                with self.assertRaises(FedbiomedOptimizerError):
                    optim_builer.build(TrainingPlans.SkLearnTrainingPlan, model, item)

        # for TorchTrainingPlan
        model = MagicMock(spec=TorchModel)
        for item in ("unknown", object, None,):
            with self.assertRaises(FedbiomedOptimizerError):
                optim_builer.build(TrainingPlans.SkLearnTrainingPlan, model, item)


if __name__ == "__main__":
    unittest.main()
