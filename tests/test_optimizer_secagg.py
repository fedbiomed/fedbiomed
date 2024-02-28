# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SecAgg-related tools for optimizer auxiliary variables."""

import dataclasses
import unittest
import secrets
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import MagicMock

import declearn
import declearn.model.torch
import numpy as np
import torch
from declearn.optimizer.modules import AuxVar

from fedbiomed.common.optimizers import (
    EncryptedAuxVar,
    Optimizer,
    flatten_auxvar_for_secagg,
    unflatten_auxvar_after_secagg,
)
from fedbiomed.common.optimizers.declearn import ScaffoldClientModule
from fedbiomed.common.secagg import SecaggCrypter
from fedbiomed.common.serializer import Serializer


ArraySpec = Tuple[List[int], str]
ValueSpec = List[Tuple[str, int, Union[None, ArraySpec, declearn.model.api.VectorSpec]]]


@dataclasses.dataclass
class SimpleAuxVar(AuxVar):
    """Mock AuxVar subclass that stores a unique sum-aggregatable field."""

    value: float


@dataclasses.dataclass
class NoSecaggAuxVar(AuxVar, register=False):
    """Mock AuxVar subclass that does not support SecAgg."""

    def prepare_for_secagg(
        self,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        raise NotImplementedError("This module does not support SecAgg.")


def generate_scaffold_aux_var(n_feats: int = 32) -> Dict[str, AuxVar]:
    """Generate auxiliary variables from a Scaffold Optimizer.

    Args:
        n_feats: Dimensionality of features of the mocked linear
            model. Parameters will have `n_feats + 1` size due
            to the addition of a bias.

    Use mock (random-valued) model weights and input gradients,
    and return auxiliary variables after running a single local
    step.
    """
    # Ensure no GPU is used.
    declearn.utils.set_device_policy(gpu=False)
    # Run a mock training step using a Scaffold optimizer.
    gradients = declearn.model.torch.TorchVector(
        {"kernel": torch.randn((n_feats, 1)), "bias": torch.randn((1,))}
    )
    weights = declearn.model.torch.TorchVector(
        {"kernel": torch.randn((n_feats, 1)), "bias": torch.randn((1,))}
    )
    optimizer = Optimizer(lr=0.001, modules=[ScaffoldClientModule()])
    optimizer.step(gradients, weights)
    # Return the resulting auxiliary variables.
    return optimizer.get_aux()



class TestFlattenAuxVarForSecagg(unittest.TestCase):
    """Unit tests for 'flatten_auxvar_for_secagg'."""

    @staticmethod
    def assert_flattened_auxvar_types(
        aux_var: Dict[str, AuxVar],
        flattened: List[float],
        enc_specs: List[ValueSpec],
        cleartext: List[Optional[Dict[str, Any]]],
        clear_cls: List[Tuple[str, Type[AuxVar]]],
    ) -> None:
        """Verify that flattened aux var match documented specifications."""
        # Assert that 'flattened' is a list of float values.
        assert isinstance(flattened, list)
        assert all(isinstance(x, float) for x in flattened)
        # Assert that 'enc_specs' lists module-wise specifications.
        assert isinstance(enc_specs, list)
        assert len(enc_specs) == len(aux_var)
        for specs in enc_specs:
            assert isinstance(specs, list)
            assert all(isinstance(s, tuple) and (len(s) == 3) for s in specs)
        # Assert that 'cleartext' is a list of module-wise optional dicts.
        assert isinstance(cleartext, list)
        assert all(isinstance(x, (dict, type(None))) for x in cleartext)
        assert len(cleartext) == len(aux_var)
        # Assert that 'clear_cls' is a list of module-wise (name, type) tuples.
        assert isinstance(clear_cls, list)
        assert len(clear_cls) == len(aux_var)
        assert clear_cls == [(key, type(val)) for key, val in aux_var.items()]

    def test_flatten_auxvar_simple(self) -> None:
        """Test flattening a dict with a single 'SimpleAuxVar' instance."""
        aux_var = {"simple": SimpleAuxVar(0.0)}  # type: Dict[str, AuxVar]
        outputs = flatten_auxvar_for_secagg(aux_var)
        self.assert_flattened_auxvar_types(aux_var, *outputs)
        assert outputs[0] == [0.0]  # single flattened value

    def test_flatten_auxvar_scaffold(self) -> None:
        """Test flattening auxiliary variables from a Scaffold optimizer."""
        # Run a mock training step using a Scaffold optimizer.
        aux_var = generate_scaffold_aux_var(n_feats=32)
        # Flatten the resulting auxiliary variables.
        outputs = flatten_auxvar_for_secagg(aux_var)
        self.assert_flattened_auxvar_types(aux_var, *outputs)
        assert len(outputs[0]) == 33  # number of model parameters

    def test_flatten_auxvar_raises_no_secagg(self) -> None:
        """Test flattening an 'AuxVar' that does not implement SecAgg."""
        aux_var = {"module": NoSecaggAuxVar()}  # type: Dict[str, AuxVar]
        with self.assertRaises(NotImplementedError):
            flatten_auxvar_for_secagg(aux_var)


class TestUnflattenAuxVarAfterSecagg(unittest.TestCase):
    """Unit tests for 'unflatten_auxvar_after_secagg'.

    Here, we leave apart the SecAgg part, and merely test that unflattening
    a freshly-flattened instance works properly.
    """

    def test_flatten_auxvar_simple(self) -> None:
        """Test flattening a dict with a single 'SimpleAuxVar' instance."""
        # Set up simple auxiliary variables, flatten and unflatten them.
        aux_var = {"simple": SimpleAuxVar(0.0)}  # type: Dict[str, AuxVar]
        flattened = flatten_auxvar_for_secagg(aux_var)
        aux_bis = unflatten_auxvar_after_secagg(*flattened)
        # Assert that unflattened auxiliary variables match original ones.
        assert isinstance(aux_bis, dict)
        assert aux_var == aux_bis

    def test_flatten_auxvar_scaffold(self) -> None:
        """Test flattening auxiliary variables from a Scaffold optimizer."""
        # Run a mock training step using a Scaffold optimizer.
        aux_var = generate_scaffold_aux_var()
        # Flatten and unflatten resulting auxiliary variables.
        flattened = flatten_auxvar_for_secagg(aux_var)
        aux_bis = unflatten_auxvar_after_secagg(*flattened)
        # Assert that unflattened auxiliary variables match original ones.
        assert isinstance(aux_bis, dict)
        assert aux_var == aux_bis


class TestEncryptedAuxVar(unittest.TestCase):
    """Unit tests for the 'EncryptedAuxVar' data structure."""

    @staticmethod
    def assert_dict_serializable(enc_aux: EncryptedAuxVar) -> None:
        """Test that an 'EncryptedAuxVar' instance can be rebuilt from dict."""
        dump = enc_aux.to_dict()
        assert isinstance(dump, dict)
        enc_bis = EncryptedAuxVar.from_dict(dump)
        assert enc_bis.encrypted == enc_aux.encrypted
        assert enc_bis.enc_specs == enc_aux.enc_specs
        assert enc_bis.cleartext == enc_aux.cleartext
        assert enc_bis.clear_cls == enc_aux.clear_cls

    @staticmethod
    def assert_string_serializable(enc_aux: EncryptedAuxVar) -> None:
        """Test that an 'EncryptedAuxVar' can be (de)serialized."""
        dump = enc_aux.to_dict()
        d_bis = Serializer.loads(Serializer.dumps(dump))
        enc_bis = EncryptedAuxVar.from_dict(d_bis)
        assert enc_bis.encrypted == enc_aux.encrypted
        assert enc_bis.enc_specs == enc_aux.enc_specs
        assert enc_bis.cleartext == enc_aux.cleartext
        assert enc_bis.clear_cls == enc_aux.clear_cls

    def test_serialization_simple(self) -> None:
        """Test that EncryptedAuxVar wrapping SimpleAuxVar can be serialized."""
        aux_var = {"simple": SimpleAuxVar(1.0)}  # type: Dict[str, AuxVar]
        flat_value, *flat_specs = flatten_auxvar_for_secagg(aux_var)
        enc_aux = EncryptedAuxVar([[int(x) for x in flat_value]], *flat_specs)
        self.assert_dict_serializable(enc_aux)
        self.assert_string_serializable(enc_aux)

    def test_serialization_scaffold(self) -> None:
        """Test that EncryptedAuxVar wrapping ScaffoldAuxVar can be serialized."""
        aux_var = generate_scaffold_aux_var()
        flat_value, *flat_specs = flatten_auxvar_for_secagg(aux_var)
        enc_aux = EncryptedAuxVar([[int(x) for x in flat_value]], *flat_specs)
        self.assert_dict_serializable(enc_aux)
        self.assert_string_serializable(enc_aux)

    def test_aggregate_simple(self) -> None:
        """Test that 'EncryptedAuxVar' aggregation works as expected.

        Here, leave encryption/decryption apart to focus solely on the
        aggregation of two instances.
        """
        # Set up a couple of 'SimpleAuxVar', flatten and wrap them.
        aux_a = {"simple": SimpleAuxVar(0.0)}  # type: Dict[str, AuxVar]
        aux_b = {"simple": SimpleAuxVar(1.0)}  # type: Dict[str, AuxVar]
        flat_a, *specs_a = flatten_auxvar_for_secagg(aux_a)
        flat_b, *specs_b = flatten_auxvar_for_secagg(aux_b)
        enc_a = EncryptedAuxVar([[int(x) for x in flat_a]], *specs_a)
        enc_b = EncryptedAuxVar([[int(x) for x in flat_b]], *specs_b)
        # Run the aggregation and verify that the output is an EncryptedAuxVar.
        agg = enc_a + enc_b
        assert isinstance(agg, EncryptedAuxVar)
        assert agg.encrypted == enc_a.encrypted + enc_b.encrypted
        # Sum-aggregated pseudo-encrypted values and unflatten the results.
        sum_values = [
            float(sum(values[i] for values in agg.encrypted))
            for i in range(len(agg.encrypted[0]))
        ]
        res = unflatten_auxvar_after_secagg(
            sum_values, agg.enc_specs, agg.cleartext, agg.clear_cls
        )
        # Verify that the obtained auxiliary variables match expectations.
        assert isinstance(res, dict)
        assert res.keys() == aux_a.keys()
        assert isinstance(res["simple"], SimpleAuxVar)
        assert res["simple"].value == 1.0

    def test_aggregate_raises_wrong_type(self) -> None:
        """Test aggregating 'EncryptedAuxVar' with another object."""
        # Set up EncryptedAuxVar wrapping simple auxiliary variables.
        aux_var = {"simple": SimpleAuxVar(1.0)}  # type: Dict[str, AuxVar]
        flat_value, *flat_specs = flatten_auxvar_for_secagg(aux_var)
        enc_aux = EncryptedAuxVar([[int(x) for x in flat_value]], *flat_specs)
        # Verify that a TypeError is raised when trying to aggregate.
        with self.assertRaises(TypeError):
            enc_aux.aggregate(MagicMock())

    def test_aggregate_raises_wrong_specs(self) -> None:
        """Test aggregating 'EncryptedAuxVar' with mismatching specs."""
        # Generate ScaffoldAuxVar instances with distinct shapes.
        aux_a = generate_scaffold_aux_var(n_feats=16)
        aux_b = generate_scaffold_aux_var(n_feats=32)
        # Flatten and wrap them (rouding up values for type correctness).
        flat_a, *specs_a = flatten_auxvar_for_secagg(aux_a)
        flat_b, *specs_b = flatten_auxvar_for_secagg(aux_b)
        enc_a = EncryptedAuxVar([[int(x) for x in flat_a]], *specs_a)
        enc_b = EncryptedAuxVar([[int(x) for x in flat_b]], *specs_b)
        # Verify that a ValueError is raised when trying to aggregate.
        with self.assertRaises(ValueError):
            enc_a.aggregate(enc_b)

    def test_aggregate_raises_wrong_classes(self) -> None:
        """Test aggregating 'EncryptedAuxVar' wrapping distinct types."""
        # Generate auxiliary variables of different nature.
        aux_a = {"simple": SimpleAuxVar(1.0)}  # type: Dict[str, AuxVar]
        aux_b = generate_scaffold_aux_var()
        # Flatten and wrap them (rouding up values for type correctness).
        flat_a, *specs_a = flatten_auxvar_for_secagg(aux_a)
        flat_b, *specs_b = flatten_auxvar_for_secagg(aux_b)
        enc_a = EncryptedAuxVar([[int(x) for x in flat_a]], *specs_a)
        enc_b = EncryptedAuxVar([[int(x) for x in flat_b]], *specs_b)
        # Verify that a ValueError is raised when trying to aggregate.
        with self.assertRaises(ValueError):
            enc_a.aggregate(enc_b)


class TestAuxVarSecAgg(unittest.TestCase):
    """Functional tests of Secure Aggregation of optimizer auxiliary variables."""

    @staticmethod
    def perform_secure_aggregation(
        aux_a: Dict[str, AuxVar],
        aux_b: Dict[str, AuxVar],
    ) -> Dict[str, AuxVar]:
        """Perform Secure Aggregation of Optimizer auxiliary variables."""
        # Set up a SecAgg crypter and private and public parameters.
        biprime = int.from_bytes(  # 1024-bits biprime number
            b'\xe2+!\x9a\xdc\xc3.\xcaY\x1b\xd6\xfdH\xfc1\xaeG6\xc0O\xa5\x9a'
            b'\x8bi)i \xac=\x88\xb5\xfdo\xac\xadS\x80\xb3xL\xa6\xc7\xca]\x17'
            b'\xb1\x16\rB\x8f"\xb1*\x12.J`\xc8AW\x92\xd0\t\x14*fwx"o\xff\xca'
            b'\xec\x8e\x86G\x7f\x9c\xdf?\x00}&\xa8b\xcd\n!\xa9\x1f\xc0\x99{'
            b'\x91h"\xe6,j\x87\xf6\xa6\xee0\xc5_\xdbi\x93\xea\x80qJ\x12\xbc'
            b'\xd7,AE\xb5\xdc\xf1\xf5\x962\xcdms',
            byteorder="big",
        )
        skey_a = secrets.randbits(2048)
        skey_b = secrets.randbits(2048)

        def encrypt(params: List[float], s_key: int) -> List[int]:
            """Perform Joye-Libert encryption."""
            nonlocal biprime
            return SecaggCrypter().encrypt(
                params=params, key=s_key,
                num_nodes=2, current_round=1, biprime=biprime,
            )

        def sum_decrypt(params: List[List[int]]) -> List[float]:
            """Perform Joye-Libert sum-decryption."""
            nonlocal biprime, skey_a, skey_b
            averaged = SecaggCrypter().aggregate(
                params=params, key=-(skey_a + skey_b), total_sample_size=2,
                num_nodes=2, current_round=1, biprime=biprime,
            )
            return [value * 2 for value in averaged]

        # Flatten, encrypt and wrap up auxiliary variables.
        flat_a, *specs_a = flatten_auxvar_for_secagg(aux_a)
        flat_b, *specs_b = flatten_auxvar_for_secagg(aux_b)
        enc_a = EncryptedAuxVar([encrypt(flat_a, skey_a)], *specs_a)
        enc_b = EncryptedAuxVar([encrypt(flat_b, skey_b)], *specs_b)
        # Aggregate both instances, decrypt and unflatten the result.
        flat = enc_a + enc_b
        flat_decrypted = sum_decrypt(flat.encrypted)
        return unflatten_auxvar_after_secagg(
            flat_decrypted, flat.enc_specs, flat.cleartext, flat.clear_cls
        )

    def test_secagg_auxvar_simple(self) -> None:
        """Test secure-aggregating simple auxiliary variables."""
        # Set up two simple auxiliary variables and perform their aggregation.
        aux_a = {"simple": SimpleAuxVar(0.0)}  # type: Dict[str, AuxVar]
        aux_b = {"simple": SimpleAuxVar(1.0)}  # type: Dict[str, AuxVar]
        result = self.perform_secure_aggregation(aux_a, aux_b)
        # Verify that results have proper type and value.
        assert isinstance(result, dict) and (result.keys() == {"simple"})
        assert isinstance(result["simple"], SimpleAuxVar)
        assert abs(result["simple"].value - 1.0) < 1e-3

    def test_secagg_auxvar_scaffold(self) -> None:
        """Test secure-aggregating Scaffold auxiliary variables."""
        # Set up two sets of Scaffold auxiliary variables and aggregate them.
        aux_a = generate_scaffold_aux_var()
        aux_b = generate_scaffold_aux_var()
        expect = {key: val_a + aux_b[key] for key, val_a in aux_a.items()}
        result = self.perform_secure_aggregation(aux_a, aux_b)
        # Verify that SecAgg results have expected type/format.
        assert isinstance(result, dict)
        assert result.keys() == expect.keys() == {"scaffold"}
        exp_scaffold = expect["scaffold"]
        res_scaffold = result["scaffold"]
        assert isinstance(exp_scaffold, declearn.optimizer.modules.ScaffoldAuxVar)
        assert isinstance(res_scaffold, declearn.optimizer.modules.ScaffoldAuxVar)
        # Verify that SecAgg and raw aggregation results match.
        assert res_scaffold.clients == exp_scaffold.clients
        assert res_scaffold.state is exp_scaffold.state is None
        assert isinstance(exp_scaffold.delta, declearn.model.torch.TorchVector)
        assert isinstance(res_scaffold.delta, declearn.model.torch.TorchVector)
        for key, val_exp in exp_scaffold.delta.coefs.items():
            val_res = res_scaffold.delta.coefs[key]
            assert np.allclose(val_exp.cpu().numpy(), val_res.cpu().numpy(), atol=1e-3)
