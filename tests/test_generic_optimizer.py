import copy
import random
from typing import Any, Dict, List, Tuple, Union
import unittest
from unittest.mock import MagicMock, patch

import declearn
import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier, SGDRegressor
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from declearn.optimizer import Optimizer as DecOptimizer
from declearn.model.torch import TorchVector
from declearn.model.sklearn import NumpyVector

from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.optimizers.declearn import (
    AdaGradModule,
    AdamModule,
    FedProxRegularizer,
    LassoRegularizer,
    MomentumModule,
    RidgeRegularizer,
    ScaffoldAuxVar,
    ScaffoldClientModule,
    ScaffoldServerModule,
    YogiModule,
    YogiMomentumModule,
    list_optim_modules,
    list_optim_regularizers,
)
from fedbiomed.common.optimizers.generic_optimizers import (
    DeclearnOptimizer,
    NativeSkLearnOptimizer,
    NativeTorchOptimizer,
    OptimizerBuilder,
)
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
            YogiMomentumModule(),
            AdaGradModule(),
            AdamModule(),
            YogiModule()
            ]
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

    def check_optimizer_states(self,
                               previous_round_optim: DeclearnOptimizer,
                               current_round_optim: DeclearnOptimizer,
                               is_same_optimizer:bool):
        """Checks that states of `previous_round_optim` (that mimicks the optimizer of previous round)is different
        from state of `current_round_optim` (that mimicks optimizer of the current round).

        Args:
            previous_round_optim (DeclearnOptimizer): mimicks optimizer of previous round
            current_round_optim (DeclearnOptimizer): mimicks optimizer of current round
            is_same_optimizer (bool): set `True` to perform an additional test, that checks if optimizers states
                are equal.

        Raises:
            AssertionError: as expected for tests, raised if test fails
        """
        previous_round_optim_state = previous_round_optim.save_state()
        optim_state_before_loading = copy.deepcopy(current_round_optim.save_state())
        current_round_optim.load_state(previous_round_optim_state, load_from_state=True)
        current_round_optim_state = current_round_optim.save_state()

        # checks
        if is_same_optimizer:
            # here we check that optimizer that has been trained is reloaded for the next round accordingly
            # (provided optimizer has not changed from one Round to another)
            self.assertDictEqual(current_round_optim_state, previous_round_optim_state)
        # check that reloaded optimodules have different states than the original ones
        for curr_module_state, before_loading_module_state in zip(
            current_round_optim_state['states']['modules'],
            optim_state_before_loading['states']['modules']
            ):

            self.assertNotEqual(curr_module_state, before_loading_module_state)


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

        optim = FedOptimizer(lr=1.)

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
        optim = FedOptimizer(lr=1.)
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
                # Set up a declearn-backed Scaffold optimizer.
                optim = FedOptimizer(lr=learning_rate, modules = [ScaffoldServerModule()])
                optim_wrapper = DeclearnOptimizer(model, optim)
                # Verify that at first it emits a zero-valued shared state.
                collected_aux_vars = optim_wrapper.get_aux()
                expected_aux_vars = {"scaffold": ScaffoldAuxVar(state=0.0)}
                self.assertDictEqual(collected_aux_vars, expected_aux_vars)
                # Pass it aggregated (i.e. sum of) client state updates.
                # Here, we simulate participation from three nodes.
                aux_1 = ScaffoldAuxVar(delta=1.0, clients={"node-1"})
                aux_2 = ScaffoldAuxVar(delta=2.0, clients={"node-2"})
                aux_3 = ScaffoldAuxVar(delta=3.0, clients={"node-3"})
                aux = {'scaffold': aux_1 + aux_2 + aux_3}
                optim_wrapper.set_aux(aux)
                # Verify that the updated global state matches expectation (average of states).
                collected_aux_vars = optim_wrapper.get_aux()
                expected_aux_vars = {'scaffold': ScaffoldAuxVar(state=2.0)}
                self.assertDictEqual(collected_aux_vars, expected_aux_vars)
                # Process new state updates and verify the global state again.
                # Here, one new node participates, and an old one does not.
                aux_1 = ScaffoldAuxVar(delta=0.0, clients={"node-1"})
                aux_2 = ScaffoldAuxVar(delta=2.0, clients={"node-2"})
                aux_4 = ScaffoldAuxVar(delta=4.0, clients={"node-4"})
                aux = {'scaffold': aux_1 + aux_2 + aux_4}
                optim_wrapper.set_aux(aux)
                # Expected state: average of newest node states:
                # {node-1: 1.0, node-2: 4.0, node-3: 3.0, node-4: 4.0}
                collected_aux_vars = optim_wrapper.get_aux()
                expected_aux_vars = {'scaffold': ScaffoldAuxVar(state=3.0)}
                self.assertDictEqual(collected_aux_vars, expected_aux_vars)

    def test_declearnoptimizer_06_states(self):
        def check_state(state: Dict[str, Any], learning_rate: float, w_decay: float, modules: List, regs: List, model):
            self.assertEqual(state['config']['lrate'], learning_rate)
            self.assertEqual(state['config']['w_decay'], w_decay)
            self.assertListEqual(state['config']['regularizers'], [(reg.name, reg.get_config()) for reg in regs])
            self.assertListEqual(state['config']['modules'], [(mod.name , mod.get_config()) for mod in modules])
            new_optim = FedOptimizer.load_state(state)
            new_optim_wrapper = DeclearnOptimizer(model, new_optim).load_state(state)
            self.assertDictEqual(new_optim_wrapper.save_state(), state)
            self.assertIsInstance(new_optim_wrapper.optimizer, FedOptimizer)

        learning_rate = .12345
        w_decay = .54321

        for model_wrappers in (self._torch_model_wrappers, self._sklearn_model_wrappers):
            for model in model_wrappers:
                optim = FedOptimizer(lr=learning_rate, decay=w_decay)
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

    def test_declearnoptimizer_07_loading_from_previous_state_1_unchanged_optim(self):
        # simulates loading of an optimizer state from previous round into a new defined optimizer
        # in the case when declearn optimizer is left unchanged
        lr = .12345
        previous_round_optim = FedOptimizer(lr=lr,
                                      modules=[AdamModule(), YogiMomentumModule()],
                                      regularizers=[LassoRegularizer()])

        for model_wrappers in (self._torch_model_wrappers, self._sklearn_model_wrappers):
            for model in model_wrappers:
                previous_round_optim_w = DeclearnOptimizer(model, previous_round_optim)
                previous_optim_state = previous_round_optim_w.save_state()
                current_round_optim = copy.deepcopy(previous_round_optim)
                current_round_optim_w = DeclearnOptimizer(model, current_round_optim)
                current_round_optim_w.load_state(previous_optim_state, load_from_state=True)
                current_optim_state = current_round_optim_w.save_state()

                self.assertDictEqual(previous_optim_state, current_optim_state)
                self.assertNotEqual(id(current_round_optim), id(previous_round_optim))  # make sure they are not same object

    def test_declearnoptimizer_07_loading_from_previous_state_1_fully_changed_optim(self):
        # simulates loading of an optimizer state from previous round into a new defined optimizer
        # case where optimizer has been fully redefined
        previous_r_lr = .12345
        current_r_lr = .3456
        previous_round_optim = FedOptimizer(lr=previous_r_lr,
                                      modules=[YogiMomentumModule(), AdaGradModule()],
                                      regularizers=[LassoRegularizer()])

        current_round_optim = FedOptimizer(lr=current_r_lr,
                                          modules=[AdamModule()])


        for model_wrappers in (self._torch_model_wrappers, self._sklearn_model_wrappers):
            for model in model_wrappers:
                previous_round_optim_w = DeclearnOptimizer(model, previous_round_optim)
                current_round_optim_w = DeclearnOptimizer(model, current_round_optim)
                previous_round_optim_state = previous_round_optim_w.save_state()
                current_round_optim_state = current_round_optim_w.save_state()

                current_round_optim_w.load_state(copy.deepcopy(previous_round_optim_state),
                                                 load_from_state=True)

                current_round_optim_state_after_loading = current_round_optim_w.save_state()

                for module_name in current_round_optim_state['states']:
                    prev = previous_round_optim_state['states'][module_name]
                    current = current_round_optim_state['states'][module_name]
                    current_after_loading = current_round_optim_state_after_loading['states'][module_name]
                    if module_name not in ('lrate', 'w_decay'):
                        self.assertNotEqual(prev, current)
                        self.assertEqual(current, current_after_loading)
                self.assertNotEqual(
                    previous_round_optim_state['config']['lrate'],
                    current_round_optim_state_after_loading['config']['lrate']
                )

                self.assertEqual(current_round_optim_state_after_loading['config']['lrate'], current_r_lr)
                self.assertDictEqual(current_round_optim_state_after_loading, current_round_optim_state)
                # these states are equal because model hasnot been trained from previous round to current

    def test_declearnoptimizer_07_loading_from_previous_state_2_unchanged_optim_sklearn(self):
        # simulates training for sklearn models
        # FIXME: this looks more like an integration test
        learning_rate = .12345

        w_decay = .54321
        n_iter = 10
        data = np.array([[1, 1, 1, 1,],
                        [1, 0, 1, 0],
                        [1, 1, 1, 1]])

        targets = np.array([[1], [0], [1], [1]])

        for model in self._sklearn_model_wrappers:
            # TODO: add ScaffoldModule in the list when it will be updated
            previous_round_optim = FedOptimizer(lr=learning_rate,
                                            decay=w_decay,
                                            modules=[AdamModule(), YogiMomentumModule()],
                                            regularizers=[LassoRegularizer()])

            current_round_optim = copy.deepcopy(previous_round_optim)
            previous_round_optim_w = DeclearnOptimizer(model, previous_round_optim)
            for _ in range(n_iter):
                previous_round_optim_w = DeclearnOptimizer(model, previous_round_optim)
                model.set_init_params({'n_features': 4, 'n_classes': 2})

                with previous_round_optim_w.optimizer_processing():
                    previous_round_optim_w.init_training()
                    model.train(data, targets)
                    previous_round_optim_w.step()

            current_round_optim_w = DeclearnOptimizer(model, current_round_optim)
            self.check_optimizer_states(previous_round_optim_w, current_round_optim_w, is_same_optimizer=True)

    def test_declearnoptimizer_07_loading_from_previous_state_3_unchanged_optim_pytorch(self):
        # tests on pytorch models
        learning_rate = .12345

        w_decay = .54321
        n_iter = 10
        data = torch.Tensor([[1,1,1,1],
                             [1,0,0,1]])
        targets = torch.Tensor([[1, 1], [0, 1]])

        loss_func = torch.nn.MSELoss()

        for model in self._torch_model_wrappers:
            previous_round_optim = FedOptimizer(lr=learning_rate,
                                            decay=w_decay,
                                            modules=[AdamModule(), YogiMomentumModule()],
                                            regularizers=[LassoRegularizer()])

            current_round_optim = copy.deepcopy(previous_round_optim)
            previous_round_optim_w = DeclearnOptimizer(model, previous_round_optim)
            for _ in range(n_iter):

                previous_round_optim_w.init_training()
                previous_round_optim_w.zero_grad()
                output = model.model.forward(data)
                loss = loss_func(output, targets)
                loss.backward()
                previous_round_optim_w.step()

            current_round_optim_w = DeclearnOptimizer(model, current_round_optim)
            self.check_optimizer_states(previous_round_optim_w, current_round_optim_w, is_same_optimizer=True)


    def test_declearnoptimizer_07_loading_from_previous_state_4_partially_changed_optimizer_sklearn(self):
        # test that optimizer is loaded accordingly, when partial changes are detected in the modules of
        # the declearn optimizer

        previous_r_lr = .12345
        current_r_lr = .3456
        w_decay = .54321
        n_iter = 10
        data = np.array([[1, 1, 1, 1,],
                        [1, 0, 1, 0],
                        [1, 1, 1, 1]])

        targets = np.array([[1], [0], [1], [1]])


        for model in self._sklearn_model_wrappers:
            common_opi_module = YogiMomentumModule()
            previous_round_optim = FedOptimizer(lr=previous_r_lr,
                                                decay=w_decay,
                                        modules=[YogiMomentumModule(), AdaGradModule()],
                                        regularizers=[LassoRegularizer()])

            current_round_optim = FedOptimizer(lr=current_r_lr,
                                            modules=[YogiMomentumModule(), AdamModule()])
            previous_round_optim_w = DeclearnOptimizer(model, previous_round_optim)

            model.set_init_params({'n_features': 4, 'n_classes': 2})
            for _ in range(n_iter):

                with previous_round_optim_w.optimizer_processing():
                    previous_round_optim_w.init_training()
                    model.train(data, targets)
                    previous_round_optim_w.step()

            current_round_optim_w = DeclearnOptimizer(model, current_round_optim)
            #self.check_optimizer_states(previous_round_optim_w, current_round_optim_w, is_same_optimizer=False)
            # check that YogiModule has been reloaded accordingly
            previous_round_optim_state = copy.deepcopy(previous_round_optim_w.save_state())

            current_round_optim_w.load_state(copy.deepcopy(previous_round_optim_state), load_from_state=True)
            current_round_optim_state = current_round_optim_w.save_state()

            for curr_module_state, prev_module_state in zip(
            current_round_optim_state['states']['modules'],
            previous_round_optim_state['states']['modules']
            ):
                if curr_module_state[0] == common_opi_module.name:
                    self.assertEqual(curr_module_state, prev_module_state)
                else:
                    self.assertNotEqual(curr_module_state, prev_module_state)

    def test_declearnoptimizer_08_loading_state_failure(self):

        learning_rate, w_decay = .1234, 3.
        bad_states = (123, set((1, 2, 3,)), ['a', 'b', 'c'],)
        for model in self._torch_model_wrappers:
            dec_optim = FedOptimizer(lr=learning_rate,
                                     decay=w_decay,
                                     modules=[AdamModule(), YogiMomentumModule()],
                                     regularizers=[LassoRegularizer()])

            optim_wrapper = DeclearnOptimizer(model, dec_optim)
            for bad_state in bad_states:
                with self.assertRaises(FedbiomedOptimizerError):
                    optim_wrapper.load_state(bad_state)

    def test_declearnoptimizer_09_declearn_optimizers_1_sklearnModel(self):
        # performs a number of optimization steps with randomly created optimizers
        # FIXME: nothing is being asserted here...
        nb_tests = 10  # number of time the following test will be executed
        learning_rate = .12345
        w_decay = .54321

        data = np.array([[1, 1, 1, 1,],
                        [1, 0, 1, 0],
                        [1, 1, 1, 1]])

        targets = np.array([[1], [0], [1], [1]])

        for _ in range(nb_tests):
            optim, modules , regs = self.create_random_declearn_optimizer(learning_rate, w_decay)
            for model in self._sklearn_model_wrappers:
                optim_w = DeclearnOptimizer(model, optim)
                model.set_init_params({'n_features': 4, 'n_classes': 2})

                with optim_w.optimizer_processing():
                    optim_w.init_training()
                    model.train(data, targets)
                    optim_w.step()

    def test_declearnoptimizer_09_declearn_optimizers_2_torchmodel(self):
        # performs a number of optimization steps with randomly created optimizers
        # FIXME: nothing is being asserted here...
        data = torch.Tensor([[1,1,1,1],
                             [1,0,0,1]])
        targets = torch.Tensor([[1, 1], [0, 1]])

        learning_rate = .12345
        w_decay = .54321

        loss_func = torch.nn.MSELoss()
        nb_tests = 10
        for _ in range(nb_tests):
            optim, modules , regs = self.create_random_declearn_optimizer(learning_rate, w_decay)
            for model in self._torch_model_wrappers:
                optim_w = DeclearnOptimizer(model, optim)

                optim_w.init_training()
                optim_w.zero_grad()
                output = model.model.forward(data)
                loss = loss_func(output, targets)
                loss.backward()
                optim_w.step()

    def test_declearnoptimizer_10_multiple_scaffold(self):
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
        declearn_optim = FedOptimizer(lr=1.)
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

    def test_declearnoptimizer_04_loading_states_from_non_compatible_optimizer(self):
        # here we test that it is possible to load optimizer from previous optimizer state that is framework-native or corss-framework
        declearn_learning_rate = .12345
        torch_learning_rate = .234
        w_decay = .54321

        for model in self.torch_models:
            model = TorchModel(model)
            torch_optim = NativeTorchOptimizer(model, torch.optim.SGD(model.model.parameters(), lr=torch_learning_rate))


            dec_optim = FedOptimizer(lr=declearn_learning_rate,
                                                    decay=w_decay,
                                            modules=[YogiMomentumModule(), AdaGradModule()],
                                            regularizers=[LassoRegularizer()])
            dec_optim = DeclearnOptimizer(model, dec_optim)

            dec_optim_state = dec_optim.save_state()

            torch_optim.load_state(dec_optim_state, load_from_state=True)
            torch_optim_state = torch_optim.save_state()

            self.assertIsInstance(torch_optim.optimizer, torch.optim.Optimizer)
            self.assertIsNone(torch_optim_state)
            for previous_optim, current_optim in ((torch_optim, dec_optim, ), (dec_optim, torch_optim,),):
                pass
                # TODO: test that in `Round` Test
                # previous_optim_state = previous_optim.save_state()
                # current_optim.load_state(copy.deepcopy(previous_optim_state), load_from_state=True)
                # current_optim_state = current_optim.save_state()


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
            YogiMomentumModule(),
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


class TestDeclearnOptimizerImport(unittest.TestCase):
    def test_1_list_optim(self):
        modules = list_optim_modules()
        self.assertIsInstance(modules, dict)

        reg = list_optim_regularizers()
        self.assertIsInstance(reg, dict)


if __name__ == "__main__":
    unittest.main()
