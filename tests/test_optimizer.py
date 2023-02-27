# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the declearn-interfacing Optimizer class."""

import json
import unittest
from unittest import mock

import declearn
import numpy as np
from declearn.model.api import Vector
from declearn.optimizer import Optimizer as DeclearnOptimizer
from declearn.optimizer.modules import OptiModule
from declearn.optimizer.regularizers import Regularizer

from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.optimizer import Optimizer


class TestOptimizer(unittest.TestCase):
    """Unit tests for the declearn-interfacing Optimizer class."""

    def test_optimizer_01_init(self) -> None:
        """Test that Optimizer instantiation works as expected.

        Note: additional syntaxes to pass actual modules or regularizers
        specifications are not tested here, as they belong to the domain
        of the declearn dependency (which runs such tests itself).
        """
        modules = [mock.create_autospec(OptiModule, instance=True)]
        regularizers = [mock.create_autospec(Regularizer, instance=True)]
        optim = Optimizer(
            lr=1e-3,
            decay=1e-4,
            modules=modules,
            regularizers=regularizers,
        )
        d_opt = getattr(optim, "_optimizer")
        self.assertIsInstance(d_opt, DeclearnOptimizer)
        self.assertEqual(d_opt.lrate, 1e-3)
        self.assertEqual(d_opt.w_decay, 1e-4)
        self.assertEqual(d_opt.modules, modules)
        self.assertEqual(d_opt.regularizers, regularizers)

    def test_optimizer_02_init_fails(self) -> None:
        """Test that Optimizer instantiation errors are caught and wrapped."""
        # This would cause a KeyError in declearn (unregistered module name).
        with self.assertRaises(FedbiomedOptimizerError):
            Optimizer(lr=1e-3, modules=["invalid-module-name"])
        # This would cause a TypeError in declearn (invalid module type).
        with self.assertRaises(FedbiomedOptimizerError):
            Optimizer(lr=1e-3, modules=[mock.MagicMock()])

    def test_optimizer_03_init_round(self) -> None:
        """Test that `Optimizer.init_round` works as expected."""
        regul = mock.create_autospec(Regularizer, instance=True)
        optim = Optimizer(lr=1e-3, regularizers=[regul])
        optim.init_round()
        regul.on_round_start.assert_called_once()

    def test_optimizer_04_init_round_fails(self) -> None:
        """Test that `Optimizer.init_round` exceptions are wrapped."""
        regul = mock.create_autospec(Regularizer, instance=True)
        regul.on_round_start.side_effect = RuntimeError
        optim = Optimizer(lr=1e-3, regularizers=[regul])
        with self.assertRaises(FedbiomedOptimizerError):
            optim.init_round()

    def test_optimizer_05_step(self) -> None:
        """Test that the `Optimizer.step` performs expected computations.

        Note: this code is mostly redundant with that of the declearn unit
        tests for `declearn.optimizer.Optimizer` that is distributed under
        Apache-2.0 license.
        """
        # Set up an optimizer with mock attributes.
        lrate = mock.MagicMock()
        decay = mock.MagicMock()
        modules = [
            mock.create_autospec(OptiModule, instance=True) for _ in range(3)
        ]
        regularizers = [
            mock.create_autospec(Regularizer, instance=True) for _ in range(2)
        ]
        optim = Optimizer(
            lr=lrate,
            decay=decay,
            modules=modules,
            regularizers=regularizers,
        )
        # Set up mock Vector inputs. Run them through the Optimizer.
        grads = mock.create_autospec(Vector, instance=True)
        weights = mock.create_autospec(Vector, instance=True)
        updates = optim.step(grads, weights)
        # Check that the inputs went through the expected plug-ins pipeline.
        inputs = grads  # initial inputs
        for reg in regularizers:
            reg.run.assert_called_once_with(inputs, weights)
            inputs = reg.run.return_value
        for mod in modules:
            mod.run.assert_called_once_with(inputs)
            inputs = mod.run.return_value
        # Check that the learning rate was properly applied.
        lrate.__neg__.assert_called_once()  # -1 * lrate
        lrate.__neg__.return_value.__mul__.assert_called_once_with(inputs)
        output = lrate.__neg__.return_value.__mul__.return_value
        # Check that the weight-decay term was properly applied.
        decay.__mul__.assert_called_once_with(weights)
        output.__isub__.assert_called_once_with(decay.__mul__.return_value)
        # Check that the outputs match the expected ones.
        self.assertIs(updates, output.__isub__.return_value)

    def test_optimizer_06_step_fails(self) -> None:
        """Test that the `Optimizer.step` exceptions are properly wrapped."""
        optim = Optimizer(lr=0.001)
        with self.assertRaises(FedbiomedOptimizerError):
            optim.step(grads=None, weights=None)

    def test_optimizer_07_get_aux(self) -> None:
        """Test `Optimizer.set_aux` using a mock Module."""
        # Set up an Optimizer, and mock modules, one of which emits aux vars.
        mockaux = mock.MagicMock()
        mod_aux = mock.create_autospec(OptiModule, instance=True)
        mod_aux.collect_aux_var.return_value = mockaux
        setattr(mod_aux, "aux_name", "mock-module-1")
        mod_nox = mock.create_autospec(OptiModule, instance=True)
        mod_nox.collect_aux_var.return_value = {}
        setattr(mod_nox, "aux_name", "mock-module-2")
        optim = Optimizer(lr=0.001, modules=[mod_aux, mod_nox])
        # Call 'get_aux' and assert that the results match expectations.
        aux = optim.get_aux()
        self.assertDictEqual(aux, {"mock-module-1": mockaux})
        mod_aux.collect_aux_var.assert_called_once()
        mod_nox.collect_aux_var.assert_called_once()

    def test_optimizer_08_get_aux_none(self) -> None:
        """Test `Optimizer.get_aux` when there are no aux-var to share."""
        optim = Optimizer(lr=0.001)
        self.assertDictEqual(optim.get_aux(), {})

    def test_optimizer_09_get_aux_fails(self) -> None:
        """Test that `Optimizer.get_aux` exceptions are wrapped."""
        module = mock.create_autospec(OptiModule, instance=True)
        module.collect_aux_var.side_effect = RuntimeError
        optim = Optimizer(lr=0.001, modules=[module])
        with self.assertRaises(FedbiomedOptimizerError):
            optim.get_aux()

    def test_optimizer_10_set_aux(self) -> None:
        """Test `Optimizer.set_aux` using a mock Module."""
        # Set up an Optimizer, a mock module and mock aux-var inputs.
        module = mock.create_autospec(OptiModule, instance=True)
        setattr(module, "aux_name", "mock-module")
        optim = Optimizer(lr=0.001, modules=[module])
        state = mock.MagicMock()
        # Call 'set_aux' and assert that the information was passed.
        optim.set_aux({"mock-module": state})
        module.process_aux_var.assert_called_once_with(state)

    def test_optimizer_11_set_aux_none(self) -> None:
        """Test `Optimizer.set_aux` when there are no aux-var to share."""
        optim = Optimizer(lr=0.001)
        self.assertIsNone(optim.set_aux({}))

    def test_optimizer_12_set_aux_fails(self) -> None:
        """Test `Optimizer.set_aux` exception-catching."""
        optim = Optimizer(lr=0.001)
        with self.assertRaises(FedbiomedOptimizerError):
            optim.set_aux({"missing": {}})

    def test_optimizer_13_get_state_mock(self) -> None:
        """Test that `Optimizer.get_state` returns a dict and calls modules."""
        module = mock.create_autospec(OptiModule, instance=True)
        optim = Optimizer(lr=0.001, modules=[module])
        state = optim.get_state()
        self.assertIsInstance(state, dict)
        module.get_config.assert_called_once()
        module.get_state.assert_called_once()

    def test_optimizer_14_get_state_fails(self) -> None:
        """Test that `Optimizer.get_state` exceptions are wrapped."""
        module = mock.create_autospec(OptiModule, instance=True)
        module.get_state.side_effect = RuntimeError
        optim = Optimizer(lr=0.001, modules=[module])
        with self.assertRaises(FedbiomedOptimizerError):
            optim.get_state()

    def test_optimizer_15_get_state(self) -> None:
        """Test that `Optimizer.get_state` is declearn-JSON-serializable.

        Use a practical case to test so, with an Adam module and a FedProx
        regularizer.
        """
        # Set up an Optimizer and run a step to build Vector states.
        optim = Optimizer(lr=0.001, modules=["adam"], regularizers=["fedprox"])
        grads = Vector.build({
            "kernel": np.random.normal(size=(8, 4)),
            "bias": np.random.normal(size=(4,))
        })
        weights = Vector.build({
            "kernel": np.random.normal(size=(8, 4)),
            "bias": np.random.normal(size=(4,))
        })
        optim.step(grads, weights)
        # Check that states can be accessed, dumped to JSON and reloaded.
        state = optim.get_state()
        sdump = json.dumps(state, default=declearn.utils.json_pack)
        self.assertIsInstance(sdump, str)
        sload = json.loads(sdump, object_hook=declearn.utils.json_unpack)
        self.assertIsInstance(sload, dict)
        self.assertEqual(sload.keys(), state.keys())

    def test_optimizer_16_load_state(self) -> None:
        """Test that `Optimizer.load_state` works properly.

        Use a practical case to test so, with an Adam module and a FedProx
        regularizer.
        """
        # Set up an Optimizer and run a step to build Vector states.
        optim = Optimizer(lr=0.001, modules=["adam"], regularizers=["fedprox"])
        grads = Vector.build({
            "kernel": np.random.normal(size=(8, 4)),
            "bias": np.random.normal(size=(4,))
        })
        weights = Vector.build({
            "kernel": np.random.normal(size=(8, 4)),
            "bias": np.random.normal(size=(4,))
        })
        optim.step(grads, weights)
        # Gather the state of that Optimizer and build a new one from it.
        state = optim.get_state()
        opt_b = Optimizer.load_state(state)
        # Check that the loaded Optimizer is the same as the original one.
        self.assertIsInstance(opt_b, Optimizer)
        self.assertEqual(opt_b.get_state(), state)
        upd_a = optim.step(grads, weights)
        upd_b = opt_b.step(grads, weights)
        self.assertEqual(upd_a, upd_b)

    def test_optimizer_17_load_state_fails(self) -> None:
        """Test that `Optimizer.load_state` exceptions are wrapped."""
        with self.assertRaises(FedbiomedOptimizerError):
            Optimizer.load_state({})


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
