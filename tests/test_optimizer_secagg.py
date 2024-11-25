# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SecAgg-related tools for optimizer auxiliary variables."""

import copy
import dataclasses
import functools
import unittest
import secrets
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import MagicMock, patch
#############################################################
# Import NodeTestCase before importing FedBioMed Module
from fedbiomed.common.constants import SecureAggregationSchemes
from fedbiomed.common.models import TorchModel
from fedbiomed.common.optimizers.generic_optimizers import DeclearnOptimizer
from fedbiomed.common.optimizers.optimizer import Optimizer as FedOptimizer
from fedbiomed.common.secagg._secagg_crypter import SecaggLomCrypter
from fedbiomed.researcher.requests._requests import FederatedRequest
from fedbiomed.researcher.secagg._secure_aggregation import SecureAggregation
from testsupport.base_case import NodeTestCase
#############################################################
import declearn
import declearn.model.torch
from fedbiomed.node.secagg._secagg_round import SecaggRound
import numpy as np

import torch
import torch.nn as nn
from fedbiomed.common.optimizers import (
    AuxVar,
    EncryptedAuxVar,
    Optimizer,
    flatten_auxvar_for_secagg,
    unflatten_auxvar_after_secagg,
)
from fedbiomed.common.optimizers.declearn import (
    ScaffoldAuxVar,
    ScaffoldClientModule,
    ScaffoldServerModule
)
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
        {
            "kernel": torch.clamp(torch.randn((n_feats, 1)), -2.5, 2.5),
            "bias": torch.clamp(torch.randn((1,)), -2.5, 2.5),
        }
    )
    weights = declearn.model.torch.TorchVector(
        {
            "kernel": torch.clamp(torch.randn((n_feats, 1)), 5.0, 5.0),
            "bias": torch.clamp(torch.randn((1,)), 5.0, 5.0),
        }
    )
    optimizer = Optimizer(lr=0.001, modules=[ScaffoldClientModule()])
    optimizer.step(gradients, weights)
    # Return the resulting auxiliary variables.
    return optimizer.get_aux()



class TestFlattenAuxVarForSecagg(unittest.TestCase):
    """Unit tests for 'flatten_auxvar_for_secagg'."""


    def assert_flattened_auxvar_types(
        self,
        aux_var: Dict[str, AuxVar],
        flattened: List[float],
        enc_specs: List[ValueSpec],
        cleartext: List[Optional[Dict[str, Any]]],
        clear_cls: List[Tuple[str, Type[AuxVar]]],
    ) -> None:
        """Verify that flattened aux var match documented specifications."""
        # Assert that 'flattened' is a list of float values.
        self.assertIsInstance(flattened, list)
        self.assertTrue(all(isinstance(x, float) for x in flattened))
        # Assert that 'enc_specs' lists module-wise specifications.
        self.assertIsInstance(enc_specs, list)
        self.assertEqual(len(enc_specs), len(aux_var))
        for specs in enc_specs:
            self.assertIsInstance(specs, list)
            self.assertTrue(all(isinstance(s, tuple) and (len(s) == 3) for s in specs))
        # Assert that 'cleartext' is a list of module-wise optional dicts.
        self.assertIsInstance(cleartext, list)
        self.assertTrue(all(isinstance(x, (dict, type(None))) for x in cleartext))
        self.assertEqual(len(cleartext), len(aux_var))
        # Assert that 'clear_cls' is a list of module-wise (name, type) tuples.
        self.assertIsInstance(clear_cls, list)
        self.assertEqual(len(clear_cls), len(aux_var))
        self.assertListEqual(clear_cls, [(key, type(val)) for key, val in aux_var.items()])

    def test_flatten_auxvar_01_simple(self) -> None:
        """Test flattening a dict with a single 'SimpleAuxVar' instance."""
        aux_var = {"simple": SimpleAuxVar(0.0)}  # type: Dict[str, AuxVar]
        outputs = flatten_auxvar_for_secagg(aux_var)
        self.assert_flattened_auxvar_types(aux_var, *outputs)
        self.assertEqual(outputs[0], [0.0])  # single flattened value

    def test_flatten_auxvar_02_scaffold(self) -> None:
        """Test flattening auxiliary variables from a Scaffold optimizer."""
        # Run a mock training step using a Scaffold optimizer.
        aux_var = generate_scaffold_aux_var(n_feats=32)
        # Flatten the resulting auxiliary variables.
        outputs = flatten_auxvar_for_secagg(aux_var)
        self.assert_flattened_auxvar_types(aux_var, *outputs)
        self.assertEqual(len(outputs[0]), 33)  # number of model parameters

    def test_flatten_auxvar_03_raises_no_secagg(self) -> None:
        """Test flattening an 'AuxVar' that does not implement SecAgg."""
        aux_var = {"module": NoSecaggAuxVar()}  # type: Dict[str, AuxVar]
        with self.assertRaises(NotImplementedError):
            flatten_auxvar_for_secagg(aux_var)


class TestUnflattenAuxVarAfterSecagg(unittest.TestCase):
    """Unit tests for 'unflatten_auxvar_after_secagg'.

    Here, we leave apart the SecAgg part, and merely test that unflattening
    a freshly-flattened instance works properly.
    """

    def test_flatten_auxvar_01_simple(self) -> None:
        """Test flattening a dict with a single 'SimpleAuxVar' instance."""
        # Set up simple auxiliary variables, flatten and unflatten them.
        aux_var = {"simple": SimpleAuxVar(0.0)}  # type: Dict[str, AuxVar]
        flattened = flatten_auxvar_for_secagg(aux_var)
        aux_bis = unflatten_auxvar_after_secagg(*flattened)
        # Assert that unflattened auxiliary variables match original ones.
        self.assertIsInstance(aux_bis, dict)
        self.assertEqual(aux_var, aux_bis)

    def test_flatten_auxvar_02_scaffold(self) -> None:
        """Test flattening auxiliary variables from a Scaffold optimizer."""
        # Run a mock training step using a Scaffold optimizer.
        aux_var = generate_scaffold_aux_var()
        # Flatten and unflatten resulting auxiliary variables.
        flattened = flatten_auxvar_for_secagg(aux_var)
        aux_bis = unflatten_auxvar_after_secagg(*flattened)
        # Assert that unflattened auxiliary variables match original ones.
        self.assertIsInstance(aux_bis, dict)
        self.assertEqual(aux_var, aux_bis)


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

    def test_serialization_01_simple(self) -> None:
        """Test that EncryptedAuxVar wrapping SimpleAuxVar can be serialized."""
        aux_var = {"simple": SimpleAuxVar(1.0)}  # type: Dict[str, AuxVar]
        flat_value, *flat_specs = flatten_auxvar_for_secagg(aux_var)
        enc_aux = EncryptedAuxVar([[int(x) for x in flat_value]], *flat_specs)
        self.assert_dict_serializable(enc_aux)
        self.assert_string_serializable(enc_aux)

    def test_serialization_02_scaffold(self) -> None:
        """Test that EncryptedAuxVar wrapping ScaffoldAuxVar can be serialized."""
        aux_var = generate_scaffold_aux_var()
        flat_value, *flat_specs = flatten_auxvar_for_secagg(aux_var)
        enc_aux = EncryptedAuxVar([[int(x) for x in flat_value]], *flat_specs)
        self.assert_dict_serializable(enc_aux)
        self.assert_string_serializable(enc_aux)

    def test_serialization_03_aggregation_simple(self) -> None:
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
        self.assertIsInstance(agg, EncryptedAuxVar)
        self.assertEqual(agg.encrypted, enc_a.encrypted + enc_b.encrypted)
        # Sum-aggregated pseudo-encrypted values and unflatten the results.
        sum_values = [
            float(sum(values[i] for values in agg.encrypted))
            for i in range(len(agg.encrypted[0]))
        ]
        res = unflatten_auxvar_after_secagg(
            sum_values, agg.enc_specs, agg.cleartext, agg.clear_cls
        )
        # Verify that the obtained auxiliary variables match expectations.
        self.assertIsInstance(res, dict)
        self.assertEqual(res.keys(), aux_a.keys())
        self.assertIsInstance(res["simple"], SimpleAuxVar)
        self.assertEqual(res["simple"].value, 1.0)

    def test_serialization_04_aggregate_raises_wrong_type(self) -> None:
        """Test aggregating 'EncryptedAuxVar' with another object."""
        # Set up EncryptedAuxVar wrapping simple auxiliary variables.
        aux_var = {"simple": SimpleAuxVar(1.0)}  # type: Dict[str, AuxVar]
        flat_value, *flat_specs = flatten_auxvar_for_secagg(aux_var)
        enc_aux = EncryptedAuxVar([[int(x) for x in flat_value]], *flat_specs)
        # Verify that a TypeError is raised when trying to aggregate.
        with self.assertRaises(TypeError):
            enc_aux.concatenate(MagicMock())

    def test_serialization_05_aggregate_raises_wrong_specs(self) -> None:
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
            enc_a.concatenate(enc_b)

    def test_serialization_06_aggregate_raises_wrong_classes(self) -> None:
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
            enc_a.concatenate(enc_b)


class TestAuxVarSecAgg(unittest.TestCase):
    """Functional tests of Secure Aggregation of optimizer auxiliary variables."""
    def setUp(self) -> None:
        self.biprime = int.from_bytes(  # 1024-bits biprime number
                b'\xe2+!\x9a\xdc\xc3.\xcaY\x1b\xd6\xfdH\xfc1\xaeG6\xc0O\xa5\x9a'
                b'\x8bi)i \xac=\x88\xb5\xfdo\xac\xadS\x80\xb3xL\xa6\xc7\xca]\x17'
                b'\xb1\x16\rB\x8f"\xb1*\x12.J`\xc8AW\x92\xd0\t\x14*fwx"o\xff\xca'
                b'\xec\x8e\x86G\x7f\x9c\xdf?\x00}&\xa8b\xcd\n!\xa9\x1f\xc0\x99{'
                b'\x91h"\xe6,j\x87\xf6\xa6\xee0\xc5_\xdbi\x93\xea\x80qJ\x12\xbc'
                b'\xd7,AE\xb5\xdc\xf1\xf5\x962\xcdms',
                byteorder="big",
            )
        self.skey_a = secrets.randbits(2040)
        
        self.data1 = torch.Tensor([[1, 1, 1 ,1],
                            [1, 0, 0, 1], 
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1]])
        
        self.targets1 = torch.Tensor([[1, 1], [1, 0], [1,1], [0,0], [0,1]])


        self.data2 = torch.Tensor([[1, 1, 1 ,1],
                            [1, 0, 0, 1], 
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 1, 1 ,1],
                            [1, 0, 0, 1], 
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1]])
        self.targets2 = torch.Tensor([[1, 1],
                                    [1, 0],
                                    [1,1],
                                    [0,0],
                                    [0,1],
                                    [0,1],
                                    [0,1],
                                    [1, 1],
                                    [1, 0],
                                    [1,1],
                                    [0,0],
                                    [0,1],
                                    [0,1],
                                    [0,1]])

        self.data3 = torch.Tensor([[1, 1, 1 ,1],
                            [1, 0, 0, 1], 
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1],
                            [1, 1, 1 ,1],
                            [1, 0, 0, 1], 
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 1]])
        self.targets3 = torch.Tensor([[1, 1], [1, 0], [1,1], [0,0], [0,1],
                                    [1, 1], [1, 0], [1,1], [0,0], [0,1]])
        
        self.model = nn.Linear(4,2)
        self.model_parameters = (torch.Tensor([[-0.3392, -0.4828, -0.1458,  0.2349],
                                               [ 0.1231,  0.4977, -0.1024, -0.2559]]),
                                torch.Tensor([ 0.3919, -0.2463]))

        for p, l in zip(self.model.parameters(), self.model_parameters):
            l.requires_grad = True
            p = l

    
    def _aggregate_model_weights(self, model_weights, avg_weights):
        for i, (f, w) in enumerate(zip(model_weights, avg_weights)):
            total_n_samples = sum(avg_weights)
            model_weights[i] = [x*w/total_n_samples for x in f]
        return list(map(lambda *x: sum(x), *model_weights))

    def _aggregate_aux_var(self, aux_vars, avg_weights):
        (self.targets1.shape[0], self.targets2.shape[0], self.targets3.shape[0])
        for ax, w in zip(aux_vars, avg_weights):
            total_n_samples = sum(avg_weights)
            for k, v in ax['scaffold'].to_dict()['delta'].coefs.items():

                ax['scaffold'].to_dict()['delta'].coefs[k] = v * w / total_n_samples
            
        exp_aux_var = sum((ax['scaffold'] for ax in aux_vars))#aux_var1['scaffold'] + aux_var2['scaffold']  + aux_var3['scaffold'] 
        return exp_aux_var

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
        skey_a = secrets.randbits(2040)
        skey_b = secrets.randbits(2040)

        def encrypt(params: List[float], s_key: int) -> List[int]:
            """Perform Joye-Libert encryption."""
            nonlocal biprime
            return SecaggCrypter().encrypt(
                params=params,
                key=s_key,
                num_nodes=2,
                current_round=1,
                biprime=biprime,
                clipping_range=3,
            )

        def sum_decrypt(
            params: List[List[int]],
            n_params: int,
        ) -> List[float]:
            """Perform Joye-Libert sum-decryption."""
            nonlocal biprime, skey_a, skey_b
            averaged = SecaggCrypter().aggregate(
                params=params,
                key=-(skey_a + skey_b),
                total_sample_size=2,
                num_nodes=2,
                current_round=1,
                biprime=biprime,
                clipping_range=3,
                num_expected_params=n_params,
            )
            return [value * 2 for value in averaged]

        # Flatten, encrypt and wrap up auxiliary variables.
        flat_a, *specs_a = flatten_auxvar_for_secagg(aux_a)
        flat_b, *specs_b = flatten_auxvar_for_secagg(aux_b)

        enc_a = EncryptedAuxVar([encrypt(flat_a, skey_a)], *specs_a)
        enc_b = EncryptedAuxVar([encrypt(flat_b, skey_b)], *specs_b)
        n_params = sum(spec[1] for mod in enc_a.enc_specs for spec in mod)
        # Aggregate both instances, decrypt and unflatten the result.
        flat = enc_a + enc_b
        flat_decrypted = sum_decrypt(flat.encrypted, n_params)
        return unflatten_auxvar_after_secagg(
            flat_decrypted, flat.enc_specs, flat.cleartext, flat.clear_cls
        )

    def test_secagg_auxvar_01_simple(self) -> None:
        """Test secure-aggregating simple auxiliary variables."""
        # Set up two simple auxiliary variables and perform their aggregation.
        aux_a = {"simple": SimpleAuxVar(0.0)}  # type: Dict[str, AuxVar]
        aux_b = {"simple": SimpleAuxVar(1.0)}  # type: Dict[str, AuxVar]
        result = self.perform_secure_aggregation(aux_a, aux_b)
        # Verify that results have proper type and value.
        self.assertIsInstance(result, dict)
        self.assertEqual(result.keys(), {"simple"})
        self.assertIsInstance(result["simple"], SimpleAuxVar)
        self.assertLess(abs(result["simple"].value - 1.0), 1e-3)

    def test_secagg_auxvar_02_scaffold(self) -> None:
        """Test secure-aggregating Scaffold auxiliary variables."""
        # Set up two sets of Scaffold auxiliary variables and aggregate them.
        aux_a = generate_scaffold_aux_var()
        aux_b = generate_scaffold_aux_var()
        expect = {key: val_a + aux_b[key] for key, val_a in aux_a.items()}

        print(expect["scaffold"].delta.coefs)
        result = self.perform_secure_aggregation(aux_a, aux_b)
        print(result["scaffold"].delta.coefs)
        # Verify that SecAgg results have expected type/format.
        self.assertIsInstance(result, dict)
        self.assertEqual(result.keys(),  expect.keys())
        self.assertEqual(result.keys(), {"scaffold"})
        exp_scaffold = expect["scaffold"]
        res_scaffold = result["scaffold"]
        self.assertIsInstance(exp_scaffold, ScaffoldAuxVar)
        self.assertIsInstance(res_scaffold, ScaffoldAuxVar)
        # Verify that SecAgg and raw aggregation results match.
        self.assertEqual(res_scaffold.clients, exp_scaffold.clients)
        self.assertIsNone(res_scaffold.state)
        self.assertIsNone(exp_scaffold.state)
        self.assertIsInstance(exp_scaffold.delta, declearn.model.torch.TorchVector)
        self.assertIsInstance(res_scaffold.delta, declearn.model.torch.TorchVector)
        for key, val_exp in exp_scaffold.delta.coefs.items():
            val_res = res_scaffold.delta.coefs[key]
            self.assertTrue(np.allclose(val_exp.cpu().numpy(), val_res.cpu().numpy(), atol=1e-2))


    def test_secagg_weights_auxvar_03_jls(self):
        for num_nodes in (2, 3, 5, 8, 10):
            for current_round in range(1, 5):
                for use_weighted_avg in (True, False):

                    torch_model = TorchModel(self.model)
                    optim = _get_scaffold_optimizer_node(torch_model, self.data1, self.targets1)

                    aux_var = optim.get_aux()

                    flat_a, enc_specs_a, cleartext_a, clear_cls_a = flatten_auxvar_for_secagg(aux_var)
                    flat_w = optim._model.flatten()

                    researcher_optim = _get_scaffold_optimizer_researcher(torch_model)

                    # encrypt material

                    encrypted_aux_var = SecaggCrypter().encrypt(
                            params=flat_a,
                            key=self.skey_a,
                            num_nodes=num_nodes,
                            current_round=current_round,
                            biprime=self.biprime,
                            clipping_range=3,
                            weight=self.targets1.shape[0] if use_weighted_avg else None
                        )

                    encrypted_model_weights = SecaggCrypter().encrypt(
                            params=flat_w,
                            key=self.skey_a,
                            num_nodes=num_nodes,
                            current_round=current_round,
                            biprime=self.biprime,
                            clipping_range=3,
                            weight=self.targets1.shape[0] if use_weighted_avg else None
                        )
                    
                    enc_a = EncryptedAuxVar([encrypted_aux_var], enc_specs_a, cleartext_a, clear_cls_a)

                    enc_aux_var = functools.reduce(lambda x, y: x+y, tuple(enc_a for _ in range(num_nodes)))  # concatenate!
                    n_params = sum(spec[1] for mod in enc_a.enc_specs for spec in mod)
                    # decrypt
                    aux_var_decrypted = SecaggCrypter().aggregate(
                            params=enc_aux_var.encrypted,
                            key=-(self.skey_a * num_nodes),
                            total_sample_size= self.targets1.shape[0]*num_nodes if use_weighted_avg else num_nodes,
                            num_nodes=num_nodes,
                            current_round=current_round,
                            biprime=self.biprime,
                            clipping_range=3,
                            num_expected_params=n_params,
                        )
                    
                    aux_var_decrypted = unflatten_auxvar_after_secagg(aux_var_decrypted,
                                                                      enc_specs_a*num_nodes, 
                                                                      cleartext_a*num_nodes,
                                                                      clear_cls_a*num_nodes)
                    aggregate_encrypted_model_w = [encrypted_model_weights] *num_nodes  # concatenate encrypted model weights
                    
                    deciphered_model_weights = SecaggCrypter().aggregate(
                        params=aggregate_encrypted_model_w,
                        key=-(self.skey_a * num_nodes),
                        total_sample_size= self.targets1.shape[0]*num_nodes if use_weighted_avg else num_nodes, 
                        num_nodes=num_nodes,
                        current_round=current_round,
                        biprime=self.biprime,
                        clipping_range=3,
                        num_expected_params=n_params,
                    )

                    researcher_optim._model.set_weights(researcher_optim._model.unflatten(deciphered_model_weights))
                    researcher_optim.set_aux(aux_var_decrypted)

                    # compare values of model before and after encryption
                    for (k_n, n_layer), (k_r, r_layer) in zip(optim._model.model.state_dict().items(),
                                                              researcher_optim._model.model.state_dict().items()):
                        for n_val, r_val in zip(n_layer, r_layer):
                            self.assertTrue(torch.any(torch.isclose(n_val.cpu(), r_val.cpu(), atol=1e-3)))

                    # compare values of scaffold correction states before and after encryption
                    correct_states_before = aux_var['scaffold'].to_dict()['delta'].coefs
                    correct_states_after = researcher_optim.get_aux()['scaffold'].to_dict()['state'].coefs

                    for (k_before, before_layer), (k_after, after_layer) in zip(correct_states_before.items(), correct_states_after.items()):
                        for before_val, after_val in zip(before_layer, after_layer):
                            self.assertTrue(torch.any(torch.isclose(before_val.cpu(), after_val.cpu(), atol=1e-3)))

    def test_secagg_weights_auxvar_04_jls_2(self):
        for current_round in range(1, 5):

            torch_model = TorchModel(self.model)

            # operation on node 1
            optim = _get_scaffold_optimizer_node(torch_model, self.data1, self.targets1)
            aux_var1 = optim.get_aux()
            
            flat_a1, enc_specs_a1, cleartext_a, clear_cls_a = flatten_auxvar_for_secagg(aux_var1)
            flat_w1 = optim._model.flatten()

            encrypted_aux_var1 = SecaggCrypter().encrypt(
                    params=flat_a1,
                    key=self.skey_a,
                    num_nodes=3,
                    current_round=current_round,
                    biprime=self.biprime,
                    clipping_range=3,
                    weight=self.targets1.shape[0]
                )
            enc_a1 = EncryptedAuxVar([encrypted_aux_var1], enc_specs_a1, cleartext_a, clear_cls_a)
            encrypted_model_weights1 = SecaggCrypter().encrypt(
                    params=flat_w1,
                    key=self.skey_a,
                    num_nodes=3,
                    current_round=current_round,
                    biprime=self.biprime,
                    clipping_range=3,
                    weight=self.targets1.shape[0]
                )

            # operation on node 2
            optim = _get_scaffold_optimizer_node(torch_model, self.data2, self.targets2,)

            aux_var2 = optim.get_aux()
            flat_a2, enc_specs_a2, cleartext_a, clear_cls_a = flatten_auxvar_for_secagg(aux_var2)
            flat_w2 = optim._model.flatten()

            encrypted_aux_var2 = SecaggCrypter().encrypt(
                    params=flat_a2,
                    key=self.skey_a,
                    num_nodes=3,
                    current_round=current_round,
                    biprime=self.biprime,
                    clipping_range=3,
                    weight=self.targets2.shape[0]
                )

            encrypted_model_weights2 = SecaggCrypter().encrypt(
                    params=flat_w2,
                    key=self.skey_a,
                    num_nodes=3,
                    current_round=current_round,
                    biprime=self.biprime,
                    clipping_range=3,
                    weight=self.targets2.shape[0]
                )
        

            enc_a2 = EncryptedAuxVar([encrypted_aux_var2], enc_specs_a1, cleartext_a, clear_cls_a)

            # operation on node 3
            optim = _get_scaffold_optimizer_node(torch_model, self.data3, self.targets3)
            aux_var3 = optim.get_aux()
            flat_a3, enc_specs_a3, cleartext_a, clear_cls_a = flatten_auxvar_for_secagg(aux_var3)
            flat_w3 = optim._model.flatten()

            encrypted_aux_var3 = SecaggCrypter().encrypt(
                    params=flat_a3,
                    key=self.skey_a,
                    num_nodes=3,
                    current_round=current_round,
                    biprime=self.biprime,
                    clipping_range=3,
                    weight=self.targets3.shape[0]
                )

            encrypted_model_weights3 = SecaggCrypter().encrypt(
                    params=flat_w3,
                    key=self.skey_a,
                    num_nodes=3,
                    current_round=current_round,
                    biprime=self.biprime,
                    clipping_range=3,
                    weight=self.targets3.shape[0]
                )
        

            enc_a3 = EncryptedAuxVar([encrypted_aux_var3], enc_specs_a1, cleartext_a, clear_cls_a)
            researcher_optim = _get_scaffold_optimizer_researcher(torch_model)

            enc_aux_var = enc_a1 + enc_a2 + enc_a3  # concatenate!
            n_params = sum(spec[1] for mod in enc_a1.enc_specs for spec in mod)
            total_n_samples = self.targets1.shape[0] + self.targets2.shape[0] + self.targets3.shape[0]
            # decrypt

            aux_var_decrypted = SecaggCrypter().aggregate(
                    params=enc_aux_var.encrypted,
                    key=-(self.skey_a *3),
                    total_sample_size=total_n_samples,
                    num_nodes=3,
                    current_round=current_round,
                    biprime=self.biprime,
                    clipping_range=3,
                    num_expected_params=n_params,
                )

            aux_var_decrypted = unflatten_auxvar_after_secagg(aux_var_decrypted,
                                                            enc_specs_a1 + enc_specs_a2 + enc_specs_a3, 
                                                            cleartext_a*3,
                                                            clear_cls_a*3)

            
            aggregate_encrypted_model_w = [
                encrypted_model_weights1, encrypted_model_weights2, encrypted_model_weights3
                ]  # concatenate encrypted model weights
            
            deciphered_model_weights = SecaggCrypter().aggregate(
                params=aggregate_encrypted_model_w,
                key=-(self.skey_a * 3),
                total_sample_size=total_n_samples,
                num_nodes=3,
                current_round=current_round,
                biprime=self.biprime,
                clipping_range=3,
                num_expected_params=n_params, # works because size of aux_var=size of model
            )

            researcher_optim._model.set_weights(researcher_optim._model.unflatten(deciphered_model_weights))
            researcher_optim.set_aux(aux_var_decrypted)

            ref = [flat_w1, flat_w2, flat_w3]

            flat_avg_weights = self._aggregate_model_weights(ref, (self.targets1.shape[0], self.targets2.shape[0],
                                                                self.targets3.shape[0]))
            exp_model_weights = researcher_optim._model.unflatten(flat_avg_weights)

            # compare values of model before and after encryption
            for (k_n, n_layer), (k_r, r_layer) in zip(exp_model_weights.items(),
                                                      researcher_optim._model.model.state_dict().items()):
                for n_val, r_val in zip(n_layer, r_layer):

                    self.assertTrue(torch.any(torch.isclose(
                        n_val.cpu().type(dtype=torch.float64),
                        r_val.cpu().type(dtype=torch.float64), atol=1e-3)))

            # compare values of scaffold correction states before and after encryption
            exp_aux_var = self._aggregate_aux_var((aux_var1, aux_var2, aux_var3),
                            (self.targets1.shape[0], self.targets2.shape[0], self.targets3.shape[0]))

            correct_states_before = exp_aux_var.to_dict()['delta'].coefs
            correct_states_after = researcher_optim.get_aux()['scaffold'].to_dict()['state'].coefs

            for (k_before, before_layer), (k_after, after_layer) in zip(correct_states_before.items(), correct_states_after.items()):
                for before_val, after_val in zip(before_layer, after_layer):
                    self.assertTrue(torch.any(torch.isclose(before_val.cpu(), after_val.cpu(), atol=1e-3)))

    
# idea: test encrypted value if array is full of zeros: check how works encryption in this case
class TestAuxVarSecAggLOM(NodeTestCase):
    @patch('fedbiomed.node.secagg._secagg_round.DHManager')
    def test_secagg_01_scaffold_and_weights_lom_encryption(self, dhmanager_patch):
        for num_nodes in (2, 3, 5, 10, 20, 23):

            data = torch.Tensor([[1, 1, 1 ,1],
                                [1, 0, 0, 1], 
                                [1, 0, 0, 0],
                                [0, 0, 0, 0],
                                [1, 0, 0, 1]])
            targets = torch.Tensor([[1, 1], [1, 0], [1,1], [0,0], [0,1]])

            torch_model = TorchModel(nn.Linear(4,2))
            optim = _get_scaffold_optimizer_node(torch_model, data, targets)

            aux_var = optim.get_aux()

            flat_a, enc_specs_a, cleartext_a, clear_cls_a = flatten_auxvar_for_secagg(aux_var)

            party_a = list(cleartext_a[0]['clients'])[0]

            self.env._values["ID"] = party_a
            # simulate correct encryption on Node side (model_weights+ )
            self.env._values["SECURE_AGGREGATION"] = True
            exp_id = 'experiment_id_1234'
            c_round = 1
            sample_size = len(targets)

            secagg_args = {'secagg_scheme': 2,
                        #'secagg_random': .123,
                        'secagg_clipping_range':3,
                        'parties': [party_a]*num_nodes,
                        'secagg_dh_id': 'secagg_id_xxxx'}
            
            dhmanager_patch.get.return_value = {'parties': secagg_args['parties'],
                                                'context': {party_a: b"\xba\xb2\xf2'\xa3\xa7\xb6\xee\x15uM\xf6j\x0f\xa9\xf8B}/ \x81]3\xa9p\x8b\xee\x9e:\xa8i("}
            }

            encrypted_model_weights, encrypted_aux_vars = [], []
            # process with party_a
            ## encrypt only aux var for party a
            for n in range(num_nodes):
                lom_secagg_party_a = SecaggRound(secagg_args, experiment_id=exp_id)
                encrypted_aux_var_a = lom_secagg_party_a.scheme.encrypt(flat_a, c_round, weight=sample_size)
                encrypted_aux_var_a = EncryptedAuxVar([encrypted_aux_var_a], enc_specs_a, cleartext_a, clear_cls_a)
                encrypted_aux_vars.append(encrypted_aux_var_a)

                ## encrypt only  model weights
                flat_w = optim._model.flatten()
                encrypted_model_weights_a = lom_secagg_party_a.scheme.encrypt(flat_w, c_round, weight=sample_size)
                encrypted_model_weights.append(encrypted_model_weights_a)

            # decrypt
            lom_secagg_agg = SecaggLomCrypter()

            n_params = len(targets) * num_nodes

            decrypted_weights = lom_secagg_agg.aggregate(encrypted_model_weights,
                                                         total_sample_size=n_params)
            decrypted_aux_var = lom_secagg_agg.aggregate([encauxvar.encrypted[0] for encauxvar in encrypted_aux_vars],
                                                         total_sample_size=n_params)


            
            researcher_optim = _get_scaffold_optimizer_researcher(torch_model)
            researcher_optim._model.set_weights(researcher_optim._model.unflatten(decrypted_weights))
            retrieved_aux_var = unflatten_auxvar_after_secagg(decrypted_aux_var,
                                                              enc_specs_a * num_nodes, 
                                                              cleartext_a * num_nodes,
                                                              clear_cls_a * num_nodes
                                                              )
            researcher_optim.set_aux(retrieved_aux_var)

            for (k_n, n_layer), (k_r, r_layer) in zip(optim._model.model.state_dict().items(),
                                                      researcher_optim._model.model.state_dict().items()):
                for n_val, r_val in zip(n_layer, r_layer):
                    self.assertTrue(torch.any(torch.isclose(n_val, r_val, atol=1e-3)))

            # compare values of scaffold correction states before and after encryptions
            correct_states_before = aux_var['scaffold'].to_dict()['delta'].coefs
            correct_states_after = researcher_optim.get_aux()['scaffold'].to_dict()['state'].coefs
            for (k_before, before_layer), (k_after, after_layer) in zip(correct_states_before.items(), correct_states_after.items()):
                for before_val, after_val in zip(before_layer, after_layer):
                    self.assertTrue(torch.any(torch.isclose(before_val.cpu(), after_val.cpu(), atol=1e-3)))


def _get_scaffold_optimizer_node( model, data, targets, loss_fct = torch.nn.MSELoss(), lr = .1):
    optim = DeclearnOptimizer(model, 
                              FedOptimizer(lr=lr, modules=[ScaffoldClientModule()]))
    optim.init_training()
    optim.zero_grad()

    out = optim._model.model.forward(data)
    loss = loss_fct(out, targets)
    loss.backward()
    optim.step()
    return optim


def _get_scaffold_optimizer_researcher(model, lr=.7):
    researcher_optim = DeclearnOptimizer(model, 
                                    FedOptimizer(lr=lr, modules=[ScaffoldServerModule()]))

    for p in researcher_optim._model.model.parameters():
        p.data.fill_(0)
    return researcher_optim
    # @patch('fedbiomed.researcher.secagg._secagg_context.Requests')
    # @patch('fedbiomed.researcher.secagg._secagg_context.SecaggDHContext')
    # @patch('fedbiomed.common.secagg_manager.SecaggDhManager')
    # @patch('fedbiomed.node.secagg._secagg_round.DHManager')
    # def test_secagg_xx_wrapper_lom_aux_var(self, dhmanager_patch,  secaggdhmanager_patch, server_start_patch,request_patch):
    #     #dhmanager_patch.get.return_value = {'parties';[]}

    #     # TODO: test with 2 clients, and with more clients

    #     request_patch.return_value.send = MagicMock(spec=FederatedRequest)
    #     request_patch.return_value.send.return_value.__enter__.return_value.policy.has_stopped_any = MagicMock(return_value=False)
    #     request_patch.return_value.send.return_value.__enter__.return_value.errors = MagicMock(return_value=False)
    #     # modifying environment
    #     self.env._values["SECURE_AGGREGATION"] = True

    #     aux_a = generate_scaffold_aux_var()
    #     aux_b = generate_scaffold_aux_var()

    #     expect = {key: val_a + aux_b[key] for key, val_a in aux_a.items()}
    #     # Flatten, encrypt and wrap up auxiliary variables.
    #     flat_a, enc_specs_a, cleartext_a, clear_cls_a = flatten_auxvar_for_secagg(aux_a)
    #     flat_b, enc_specs_b, cleartext_b, clear_cls_b = flatten_auxvar_for_secagg(aux_b)
    #     print(flat_a, "/n", enc_specs_a, "/n",cleartext_a, "/n", clear_cls_a, "/n" ,expect)
        
    #     #  `sample_size`=1
    #     print()
    #     print(cleartext_a[0])
    #     c_round = 1
    #     sample_size = 1
    #     exp_id = 'experiment_id_1234'
    #     party_a, party_b = list(cleartext_a[0]['clients'])[0], list(cleartext_b[0]['clients'])[0]
    #     self.env._values["ID"] = party_a
    #     secagg_args = {'secagg_scheme': 2,
    #                    #'secagg_random': .123,
    #                    'secagg_clipping_range':None,
    #                    'parties': [party_a, party_b],
    #                    'secagg_dh_id': 'secagg_id_xxxx'}
    #     dhmanager_patch.get.return_value = {'parties': secagg_args['parties'],
    #                                         'context': {party_b: b"\xba\xb2\xf2'\xa3\xa7\xb6\xee\x15uM\xf6j\x0f\xa9\xf8B}/ \x81]3\xa9p\x8b\xee\x9e:\xa8i("}
    #     }
    #     lom_secagg = SecaggRound(secagg_args, experiment_id=exp_id)

    #     encrypted_a = lom_secagg.scheme.encrypt(flat_a, current_round=c_round, weight=sample_size)

    #     dhmanager_patch.get.return_value = {'parties': secagg_args['parties'],
    #                                         'context': {party_a: b"\xba\xb2\xf2'\xa3\xa7\xb6\xee\x15uM\xf6j\x0f\xa9\xf8B}/ \x81]3\xa9p\x8b\xee\x9e:\xa8i("}
    #     }
    #     encrypted_b = lom_secagg.scheme.encrypt(flat_b, current_round=c_round, weight=sample_size)

    #     encrypted_a = EncryptedAuxVar(encrypted_a, enc_specs_a, cleartext_a, clear_cls_a)
    #     encrypted_b = EncryptedAuxVar(encrypted_b, enc_specs_b, cleartext_b, clear_cls_b)
    #     print(flat_a, encrypted_a)

    #     print("expecting", enc_specs_a)
    #     # mimicks `_preaggregate_encrypted_optim_auxvar` method
        
    #     # aggregation
    #     secagg = SecureAggregation(scheme=SecureAggregationSchemes.LOM)

    #     self.assertEqual(encrypted_a.get_num_expected_params(), encrypted_b.get_num_expected_params())
    #     #n_samples = sum([encrypted_a.get_num_expected_params(), encrypted_b.get_num_expected_params()])
    #     n_samples = encrypted_a.get_num_expected_params()
    #     encrypted_model_weights = {p: enc for p, enc in zip([party_a, party_b], (encrypted_a, encrypted_b,))}
    #     import pdb; pdb.set_trace()
    #     secagg.setup(parties=[party_a, party_b], experiment_id=exp_id)
        
    #     flatten = secagg.aggregate(
    #                                total_sample_size=n_samples,
    #                                model_params=encrypted_model_weights,
    #                                clipping_range=None)