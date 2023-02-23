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

    def test_init(self) -> None:
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
        self.assertEqual(optim._lr, 1e-3)
        self.assertEqual(optim._decay, 1e-4)
        self.assertIsInstance(optim._optimizer, DeclearnOptimizer)
        self.assertEqual(optim._optimizer.modules, modules)
        self.assertEqual(optim._optimizer.regularizers, regularizers)

    def test_init_fails(self) -> None:
        """Test that Optimizer instantiation errors are caught and wrapped."""
        # This would cause a KeyError in declearn (unregistered module name).
        with self.assertRaises(FedbiomedOptimizerError):
            Optimizer(lr=1e-3, modules=["invalid-module-name"])
        # This would cause a TypeError in declearn (invalid module type).
        with self.assertRaises(FedbiomedOptimizerError):
            Optimizer(lr=1e-3, modules=[mock.MagicMock()])

    def test_step(self) -> None:
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
        assert updates is output.__isub__.return_value

    def test_step_fails(self) -> None:
        """Test that the `Optimizer.step` exceptions are properly wrapped."""
        optim = Optimizer(lr=0.001)
        with self.assertRaises(FedbiomedOptimizerError):
            optim.step(grads=None, weights=None)

    def test_get_aux_none(self) -> None:
        """Test `Optimizer.get_aux` when there are no aux-var to share."""
        optim = Optimizer(lr=0.001)
        self.assertEqual(optim.get_aux(), {})

    def test_get_aux_scaffold(self) -> None:
        """Test `Optimizer.get_aux` when there are Scaffold aux-var to share."""
        optim = Optimizer(lr=0.001, modules=["scaffold-client"])
        # Run an optimizer step (required for Scaffold to produce aux-vars).
        grads = Vector.build({
            "kernel": np.random.normal(size=(8, 4)),
            "bias": np.random.normal(size=(4,))
        })
        weights = Vector.build({
            "kernel": np.random.normal(size=(8, 4)),
            "bias": np.random.normal(size=(4,))
        })
        optim.step(grads, weights)
        # Access the auxiliary variables and verify their formatting.
        aux = optim.get_aux()
        self.assertIsInstance(aux, dict)
        self.assertEqual(aux.keys(), {"scaffold"})
        self.assertIsInstance(aux["scaffold"], dict)

    def test_set_aux_none(self) -> None:
        """Test `Optimizer.set_aux` when there are no aux-var to share."""
        optim = Optimizer(lr=0.001)
        self.assertIsNone(optim.set_aux({}))

    def test_set_aux_fails(self) -> None:
        """Test `Optimizer.set_aux` exception-catching."""
        optim = Optimizer(lr=0.001)
        with self.assertRaises(FedbiomedOptimizerError):
            optim.set_aux({"missing": {}})

    def test_set_aux_scaffold(self) -> None:
        """Test `Optimizer.set_aux` when there are Scaffold aux-var to share."""
        # Set up an Optimizer with a server-side Scaffold module.
        optim = Optimizer(lr=0.001, modules=["scaffold-server"])
        # Set up random-valued client-emitted Scaffold auxiliary variables.
        state = Vector.build({
            "kernel": np.random.normal(size=(8, 4)),
            "bias": np.random.normal(size=(4,))
        })
        aux = {"scaffold": {"client": {"state": state}}}
        # Test that these can be input into the server-side Optimizer.
        optim.set_aux(aux)

    def test_get_state_mock(self) -> None:
        """Test that `Optimizer.get_state` returns a dict and calls modules."""
        module = mock.create_autospec(OptiModule, instance=True)
        optim = Optimizer(lr=0.001, modules=[module])
        state = optim.get_state()
        self.assertIsInstance(state, dict)
        module.get_config.assert_called_once()
        module.get_state.assert_called_once()

    def test_get_state_json(self) -> None:
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

    def test_load_state(self) -> None:
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

    def test_load_state_fails(self) -> None:
        """Test that `Optimizer.load_state` exceptions are wrapped."""
        with self.assertRaises(FedbiomedOptimizerError):
            Optimizer.load_state({})
