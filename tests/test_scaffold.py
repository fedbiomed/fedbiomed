
import os
import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.functional import federated_averaging
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.responses import Responses
from testsupport.fake_uuid import FakeUuid
import torch
from torch.nn import Linear
from fedbiomed.researcher.aggregators.scaffold import Scaffold

import copy
import random

from testsupport.base_case import ResearcherTestCase


class TestScaffold(ResearcherTestCase):
    '''
    Tests the Scaffold class
    '''

    # before the tests
    def setUp(self):
        self.model = Linear(10, 3)
        self.n_nodes = 4
        self.node_ids = [f'node_{i}'for i in range(self.n_nodes)]
        self.fds = FederatedDataSet({node: {} for node in self.node_ids})
        self.models = {node_id: Linear(10, 3).state_dict() for i, node_id in enumerate(self.node_ids)}
        self.zero_model = copy.deepcopy(self.model)  # model where all parameters are equals to 0
        self.responses = Responses([])
        for node_id in self.node_ids:
            self.responses.append({'node_id': node_id, 'optimizer_args': {'lr' : [.1]}})
        self.responses = Responses([self.responses])

        self.weights = [{node_id: random.random()} for (node_id, _) in zip(self.node_ids, self.models)]

        # setting all coefficients of `zero_model` to 0
        for p in self.zero_model.parameters():
            p.data.fill_(0)

    # after the tests
    def tearDown(self):
        pass

    def assert_coherent_states(self, agg: Scaffold) -> None:
        """Raise if the states of a Scaffold aggregator are not coherent.

        In this context, coherence means that:
            - the global state is the average of node-wise ones
            - the delta variables match their definition (c_i - c)
        """
        # Check that the global state is the average of local ones.
        for key, val in agg.global_state.items():
            avg = sum(states[key] for states in agg.nodes_states.values()) / len(agg.nodes_states)
            self.assertTrue((val == avg).all())
        # Check that delta variables match their definition.
        for node_id in self.node_ids:
            self.assertTrue(all(
                (agg.nodes_deltas[node_id][key] == agg.nodes_states[node_id][key] - val).all()
                for key, val in agg.global_state.items()
            ))

    def test_1_init_correction_states(self):
        """Test that 'init_correction_states' works properly."""
        agg = Scaffold(server_lr=1., fds=self.fds)
        global_model = self.model.state_dict()
        agg.init_correction_states(global_model)
        # Check that the global state has proper keys and zero values.
        self.assertEqual(agg.global_state.keys(), global_model.keys())
        self.assertTrue(all(
            (agg.global_state[key] == 0).all() for key in agg.global_state
        ))
        # Check that node dicts have proper keys.
        self.assertEqual(list(agg.nodes_states), self.node_ids)
        self.assertEqual(list(agg.nodes_deltas), self.node_ids)
        # Check that node-wise states and deltas have proper keys and values.
        for node in self.node_ids:
            self.assertEqual(agg.nodes_states[node].keys(), global_model.keys())
            self.assertEqual(agg.nodes_deltas[node].keys(), global_model.keys())
            self.assertTrue(all(
                (agg.nodes_states[node][key] == 0).all() for key in global_model
            ))
            self.assertTrue(all(
                (agg.nodes_deltas[node][key] == 0).all() for key in global_model
            ))

    def test_2_update_correction_state_all_nodes(self):
        """Test that 'update_correction_states' works properly.

        Case: all nodes were sampled, all with zero-valued updates.
        """
        # Instantiate a Scaffold aggregator and initialize its states.
        agg = Scaffold(server_lr=1., fds=self.fds)
        agg.init_correction_states(self.model.state_dict())
        agg.nodes_lr = {k: [1] * self.n_nodes for k in self.node_ids}
        # Test with zero-valued client models, i.e. updates equal to model.
        agg.update_correction_states(
            {node_id: self.model.state_dict() for node_id in self.node_ids},
            n_updates=1,
        )
        # Check that the local states were properly updated.
        for node_id in self.node_ids:
            self.assertTrue(all(
                (agg.nodes_states[node_id][key] == val).all()
                for key, val in self.model.state_dict().items()
            ))
        self.assert_coherent_states(agg)

    def test_3_update_correction_state_single_node(self):
        """Test that 'update_correction_states' works properly.

        Case: a single node was sampled, with random-valued updates.
        """
        # Instantiate a Scaffold aggregator and initialize its states.
        agg = Scaffold(server_lr=1., fds=self.fds)
        agg.init_correction_states(self.model.state_dict())
        agg.nodes_lr = {k: [1] * self.n_nodes for k in self.node_ids}
        # Test when a single client has non-zero-updates after 4 steps.
        updates = {
            key: torch.rand_like(val)
            for key, val in self.model.state_dict().items()
        }
        agg.update_correction_states({self.node_ids[0]: updates}, n_updates=4)
        # Check that this client's local state was properly updated.
        # Note that the previous delta is zero.
        self.assertTrue(all(
            (agg.nodes_states[self.node_ids[0]][key] == (val / 4.0)).all()
            for key, val in updates.items()
        ))
        # Check that other clients' local state was left unaltered.
        for node_id in self.node_ids[1:]:
            self.assertTrue(all(
                (agg.nodes_states[node_id][key] == 0.).all()
                for key in self.model.state_dict()
            ))
        # Check that the global state and deltas were properly updated.
        self.assert_coherent_states(agg)

    def test_4_aggregate(self):
        """Test that 'aggregate' works properly."""

        training_plan = MagicMock()
        training_plan.get_model_params = MagicMock(return_value = self.node_ids)

        agg = Scaffold(server_lr=.2, fds=self.fds)
        n_round = 0

        weights = {node_id: 1./self.n_nodes for node_id in self.node_ids}
        # assuming that global model has all its coefficients to 0
        aggregated_model_params_scaffold = agg.aggregate(
            model_params=copy.deepcopy(self.models),
            weights=weights,
            global_model=copy.deepcopy(self.zero_model.state_dict()),
            training_plan=training_plan,
            training_replies=self.responses,
            n_round=n_round
        )
        aggregated_model_params_fedavg = FedAverage().aggregate(
            copy.deepcopy(self.models), weights
        )
        # we check that fedavg and scaffold give proportional results provided:
        # - all previous correction state model are set to 0 (round 0)
        # - model proportions are the same
        # then:
        # fedavg: x_i <- x_i / n_nodes
        # scaffold: x_i <- server_lr * x_i / n_nodes
        for v_s, v_f in zip(
            aggregated_model_params_scaffold.values(),
            aggregated_model_params_fedavg.values()
        ):
            self.assertTrue(torch.isclose(v_s, v_f * .2).all())

        # check that at the end of aggregation, all correction states are non zeros (
        for deltas in agg.nodes_deltas.values():
            for layer in deltas.values():
                self.assertFalse(torch.nonzero(layer).all())

    def test_5_setting_scaffold_with_wrong_parameters(self):
        """test_5_setting_scaffold_with_wrong_parameters: tests that scaffold is
        returning an error when set with incorrect parameters
        """
        #  test 1: `server_lr` should be different than 0
        for x in (0, 0.):
            with self.assertRaises(FedbiomedAggregatorError):
                Scaffold(server_lr = x)

        # test 2: calling `init_correction_states` without any federated dataset
        with self.assertRaises(FedbiomedAggregatorError):
            scaffold = Scaffold()
            scaffold.init_correction_states(self.model.state_dict())

        # test 3: `n_updates` should be a positive and non zero integer
        training_plan = MagicMock()
        for x in (-1, .2, 0, 0., -3.2):
            with self.assertRaises(FedbiomedAggregatorError):
                scaffold = Scaffold()
                scaffold.check_values(n_updates=x, training_plan=training_plan)

        # test 4: `FederatedDataset` has not been specified
        with self.assertRaises(FedbiomedAggregatorError):
            scaffold = Scaffold()
            scaffold.check_values(n_updates=1, training_plan=training_plan)
        with self.assertRaises(FedbiomedAggregatorError):
            scaffold = Scaffold()
            scaffold.check_values(n_updates=None, training_plan=training_plan)

    def test_6_create_aggregator_args(self):
        agg = Scaffold(fds=self.fds)
        agg_thr_msg, agg_thr_file = agg.create_aggregator_args(self.model.state_dict(),
                                                               self.node_ids)

        for node_id in self.node_ids:
            for (k, v), (k0, v0) in zip(agg.nodes_deltas[node_id].items(),
                                        self.zero_model.state_dict().items()):
                self.assertTrue(torch.isclose(v, v0).all())


        # check that each element returned by method contains key 'aggregator_name'
        for node_id in self.node_ids:
            self.assertTrue(agg_thr_msg[node_id].get('aggregator_name', False))

            self.assertTrue(agg_thr_file[node_id].get('aggregator_name', False))

        # check `agg_thr_file` contains node correction state
        for node_id in self.node_ids:
            self.assertDictEqual(agg_thr_file[node_id]['aggregator_correction'], agg.nodes_deltas[node_id])
        # checking case where a node has been added to the training (repeating same tests above)
        self.n_nodes += 1
        self.node_ids.append(f'node_{self.n_nodes}')
        self.fds.data()[f'node_{self.n_nodes}'] = {}
        agg_thr_msg, agg_thr_file = agg.create_aggregator_args(self.model.state_dict(),
                                                               self.node_ids)

        for node_id in self.node_ids:
            self.assertTrue(agg_thr_msg[node_id].get('aggregator_name', False))
            self.assertTrue(agg_thr_file[node_id].get('aggregator_name', False))

        # check `agg_thr_file` contains node correction state
        for node_id in self.node_ids:
            self.assertDictEqual(agg_thr_file[node_id]['aggregator_correction'], agg.nodes_deltas[node_id])

    @patch('uuid.uuid4')
    def test_7_save_state(self, uuid_patch):
        uuid_patch.return_value = FakeUuid()
        server_lr = .5
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        bkpt_path = '/path/to/my/breakpoint'
        scaffold = Scaffold(server_lr, fds=fds)
        scaffold.init_correction_states(self.model.state_dict())
        with patch("fedbiomed.common.serializer.Serializer.dump") as save_patch:
            state = scaffold.save_state(breakpoint_path=bkpt_path, global_model=self.model.state_dict())
        self.assertEqual(save_patch.call_count, self.n_nodes + 1,
                        f"'Serializer.dump' should be called {self.n_nodes} times: once for each node + \
                        one more time for global_state")

        for node_id in self.node_ids:
            self.assertEqual(state['parameters']['aggregator_correction'][node_id],
                             os.path.join(bkpt_path, 'aggregator_correction_' + str(node_id) + '.mpk'))

        self.assertEqual(state['parameters']['server_lr'], server_lr)
        self.assertEqual(state['parameters']['global_state_filename'], os.path.join(bkpt_path,
                                                                                    'global_state_'
                                                                                    + str(FakeUuid.VALUE) + '.mpk'))
        self.assertEqual(state['class'], Scaffold.__name__)
        self.assertEqual(state['module'], Scaffold.__module__)

    def test_8_load_state(self):
        """Test that 'load_state' triggers the proper amount of calls."""
        server_lr = .5
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        bkpt_path = '/path/to/my/breakpoint'
        scaffold = Scaffold(server_lr, fds=fds)

        # create a state (not actually saving the associated contents)
        with patch("fedbiomed.common.serializer.Serializer.dump"):
            state = scaffold.save_state(
                breakpoint_path=bkpt_path, global_model=self.model.state_dict()
            )

        # action
        with patch("fedbiomed.common.serializer.Serializer.load") as load_patch:
            scaffold.load_state(state)

        self.assertEqual(load_patch.call_count, self.n_nodes + 1,
                         f"'Serializer.load' should be called {self.n_nodes} times: once for each node + \
                         one more time for global_state")

    def test_9_load_state_2(self):
        """Test that 'load_state' properly assigns loaded values."""
        server_lr = .5
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        bkpt_path = '/path/to/my/breakpoint'
        scaffold = Scaffold(server_lr, fds=fds)

        # create a state (not actually saving the associated contents)
        with patch("fedbiomed.common.serializer.Serializer.dump"):
            state = scaffold.save_state(
                breakpoint_path=bkpt_path, global_model=self.model.state_dict()
            )

        # action
        with patch(
            "fedbiomed.common.serializer.Serializer.load",
            return_value=self.model.state_dict()
        ):
            scaffold.load_state(state)

        # tests
        for node_id in self.node_ids:
            for (k,v), (k_ref, v_ref) in zip(scaffold.nodes_deltas[node_id].items(),
                                             self.model.state_dict().items()):
                self.assertTrue(torch.isclose(v, v_ref).all())

            for (k, v), (k_0, v_0) in zip(scaffold.global_state.items(),
                                          self.model.state_dict().items()):
                self.assertTrue(torch.isclose(v, v_0).all())
        self.assertEqual(scaffold.server_lr, server_lr)

    def test_10_set_nodes_learning_rate_after_training(self):

        n_rounds = 3

        # test case were learning rates change from one layer to another
        lr = [.1,.2,.3]
        n_model_layer = len(lr)  # number of layers model contains

        training_replies = {r:
            Responses( [{'node_id': node_id, 'optimizer_args': {'lr': lr}}
                          for node_id in self.node_ids])
            for r in range(n_rounds)}

        #assert n_model_layer == len(lr), "error in test: n_model_layer must be equal to the length of list of learning rate"
        training_plan = MagicMock()
        get_model_params_mock = MagicMock()

        get_model_params_mock.__len__ = MagicMock(return_value=n_model_layer)
        training_plan.get_model_params.return_value = get_model_params_mock

        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        scaffold = Scaffold(fds=fds)
        for n_round in range(n_rounds):
            node_lr = scaffold.set_nodes_learning_rate_after_training(training_plan=training_plan,
                                                                      training_replies=training_replies,
                                                                      n_round=n_round)
            test_node_lr = {node_id: lr for node_id in self.node_ids}

            self.assertDictEqual(node_lr, test_node_lr)

        # same test with a mix of nodes present in training_replies and non present

        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids + ['node_99']})
        training_plan.get_learning_rate = MagicMock(return_value=lr)
        scaffold = Scaffold(fds=fds)
        for n_round in range(n_rounds):
            node_lr = scaffold.set_nodes_learning_rate_after_training(training_plan=training_plan,
                                                                      training_replies=training_replies,
                                                                      n_round=n_round)

        # test case where len(lr) != n_model_layer
        lr += [.333]
        training_plan.get_learning_rate = MagicMock(return_value=lr)

        for n_round in range(n_rounds):
            with self.assertRaises(FedbiomedAggregatorError):
                scaffold.set_nodes_learning_rate_after_training(training_plan=training_plan,
                                                                training_replies=training_replies,
                                                                n_round=n_round)
# TODO:
# ideas for further tests:
# test 1: check that with one client only, correction terms are zeros
# test 2: check that for 2 clients, correction terms have opposite values
if __name__ == '__main__':  # pragma: no cover
    unittest.main()
