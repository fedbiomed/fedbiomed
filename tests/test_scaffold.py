
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.optimizers.generic_optimizers import NativeTorchOptimizer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.functional import federated_averaging
from fedbiomed.researcher.datasets import FederatedDataSet
from testsupport.fake_uuid import FakeUuid
import torch
import torch.nn as nn
from torch.nn import Linear
from fedbiomed.researcher.aggregators.scaffold import Scaffold

import copy
import random



class TestScaffold(unittest.TestCase):
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
        self.replies = {}
        for node_id in self.node_ids:
            self.replies.update({
                node_id: {
                    'node_id': node_id,
                    'optimizer_args': {
                        'lr' : {
                            k: .1 for k in self.model.state_dict().keys()
                        }
                    }
                }}
            )
        self.replies = [self.replies]
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
        agg.nodes_lr = {k:
            {
                layer: 1 for layer in self.model.state_dict().keys()
             } for k in self.node_ids}
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
        agg.nodes_lr = {k:
            {
                layer: 1. for layer in self.models[k].keys()
        }
            for k in self.node_ids}
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
        training_plan.get_model_params = MagicMock(return_value = Linear(10, 3).state_dict())

        agg = Scaffold(server_lr=.2, fds=self.fds)
        n_round = 0

        weights = {node_id: 1./self.n_nodes for node_id in self.node_ids}
        # assuming that global model has all its coefficients to 0
        aggregated_model_params_scaffold = agg.aggregate(
            model_params=copy.deepcopy(self.models),
            weights=weights,
            global_model=copy.deepcopy(self.zero_model.state_dict()),
            training_plan=training_plan,
            training_replies=self.replies[0],
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
        agg_args = agg.create_aggregator_args(self.model.state_dict(),
                                                               self.node_ids)

        for node_id in self.node_ids:
            for (k, v), (k0, v0) in zip(agg.nodes_deltas[node_id].items(),
                                        self.zero_model.state_dict().items()):
                self.assertTrue(torch.isclose(v, v0).all())


        # check that each element returned by method contains key 'aggregator_name'
        for node_id in self.node_ids:
            self.assertTrue(agg_args[node_id].get('aggregator_name', False))
            self.assertTrue(agg_args[node_id].get('aggregator_name', False))
            self.assertDictEqual(agg_args[node_id]['aggregator_correction'], agg.nodes_deltas[node_id])


        # checking case where a node has been added to the training (repeating same tests above)
        self.n_nodes += 1
        self.node_ids.append(f'node_{self.n_nodes}')
        self.fds.data()[f'node_{self.n_nodes}'] = {}
        agg_args = agg.create_aggregator_args(self.model.state_dict(),
                                                               self.node_ids)

        for node_id in self.node_ids:
            self.assertTrue(agg_args[node_id].get('aggregator_name', False))
            self.assertDictEqual(agg_args[node_id].get('aggregator_correction'), agg.nodes_deltas[node_id])

    @patch('uuid.uuid4')
    def test_7_save_state_breakpoint(self, uuid_patch):
        uuid_patch.return_value = FakeUuid()
        server_lr = .5
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        bkpt_path = '/path/to/my/breakpoint'
        scaffold = Scaffold(server_lr, fds=fds)
        scaffold.init_correction_states(self.model.state_dict())


        with patch("fedbiomed.common.serializer.Serializer.dump") as save_patch:
            state = scaffold.save_state_breakpoint(breakpoint_path=bkpt_path, global_model=self.model.state_dict())

        self.assertEqual(save_patch.call_count, 2,
                        f"'Serializer.dump' should be called 2 times")

        for node_id in self.node_ids:
            self.assertIsInstance(state['parameters'], str)

        self.assertEqual(state['class'], Scaffold.__name__)
        self.assertEqual(state['module'], Scaffold.__module__)

    def test_8_load_state_breakpoint(self):
        """Test that 'load_state_breakpoint' triggers the proper amount of calls."""
        server_lr = .5
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        bkpt_path = '/path/to/my/breakpoint'
        scaffold = Scaffold(server_lr, fds=fds)

        # create a state (not actually saving the associated contents)
        with patch("fedbiomed.common.serializer.Serializer.dump"):
            state = scaffold.save_state_breakpoint(
                breakpoint_path=bkpt_path, global_model=self.model.state_dict()
            )

        # action
        with patch("fedbiomed.common.serializer.Serializer.load") as load_patch:
            scaffold.load_state_breakpoint(state)

        self.assertEqual(load_patch.call_count, 2,
                         f"'Serializer.load' should be called 2, for global model and parameters")

    def test_9_set_nodes_learning_rate_after_training(self):
        n_rounds = 3
        # test case were learning rates change from one layer to another
        lr = {'layer-1': .1,
              'layer-2': .2,
              'layer-3': .3}
        n_model_layer = len(lr)  # number of layers model contains
        training_replies = {node_id: {'node_id': node_id, 'optimizer_args': {'lr': lr}} for node_id in self.node_ids}
        training_plan = MagicMock()
        get_model_params_mock = MagicMock()
        get_model_params_mock.__len__ = MagicMock(return_value=n_model_layer)
        training_plan.get_model_params.return_value = get_model_params_mock
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        scaffold = Scaffold(fds=fds)
        for n_round in range(n_rounds):
            node_lr = scaffold.set_nodes_learning_rate_after_training(training_plan=training_plan,
                                                                      training_replies=training_replies)
            test_node_lr = {node_id: lr for node_id in self.node_ids}
            self.assertDictEqual(node_lr, test_node_lr)

        # same test with a mix of present and absent nodes in training_replies
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids + ['node_99']})
        optim_w = MagicMock(spec=NativeTorchOptimizer)
        optim_w.get_learning_rate = MagicMock(return_value=lr)
        training_plan.optimizer = MagicMock(return_value=optim_w)
        scaffold = Scaffold(fds=fds)
        for n_round in range(n_rounds):
            node_lr = scaffold.set_nodes_learning_rate_after_training(training_plan=training_plan,
                                                                      training_replies=training_replies)

        # test case where len(lr) != n_model_layer
        lr.update({'layer-4': .333})
        training_plan.get_learning_rate = MagicMock(return_value=lr)
        for n_round in range(n_rounds):
            with self.assertRaises(FedbiomedAggregatorError):
                scaffold.set_nodes_learning_rate_after_training(training_plan=training_plan,
                                                                training_replies=training_replies)


class TestIntegrationScaffold(unittest.TestCase):
    # For testing training_plan setter of Experiment
    class FakeModelTorch(TorchTrainingPlan):
        """ Should inherit TorchTrainingPlan to pass the condition
            `issubclass` of `TorchTrainingPlan`
        """

        class ComplexModel(nn.Module):
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

        def __init__(self):
            super().__init__()
            # for test Exprmient 26 (test_experiment_26_run_once_with_scaffold_and_training_args)
            # this has be done to avoid mocking a private attribute (`_dp_controller`), which is inappropriate
            self._dp_controller = MagicMock()
            do_nothing = lambda x: x
            self._dp_controller.side_effect = do_nothing

        def init_model(self, args):
            return self.ComplexModel()

        def init_optimizer(self):
            lr_1,  lr_block, lr = .1, .2, .3
            return torch.optim.SGD(
                [
                    {'params': self.model().block.parameters(), 'lr': lr_block},
                    {'params': self.model().conv1.parameters(), 'lr': lr_1},
                    {'params': self.model().conv2.parameters()},
                    {'params': self.model().classifier.parameters()},
                    {'params': self.model().upsampler.parameters()},
                    {'params': self.model().bn.parameters()}
                            ], lr=lr)

        def training_step(self):
            pass

        def training_data(self):
            pass

    def setUp(self):
        self.model = self.FakeModelTorch.ComplexModel()
        self.n_nodes = 4
        self.node_ids = [f'node_{i}'for i in range(self.n_nodes)]
        self.fds = FederatedDataSet({node: {} for node in self.node_ids})
        self.models = {node_id: copy.deepcopy(self.model.state_dict()) for i, node_id in enumerate(self.node_ids)}
        self.weights = {node_id: 1 / self.n_nodes for node_id in self.node_ids}


        self.weights = [{node_id: random.random()} for (node_id, _) in zip(self.node_ids, self.models)]


    # after the tests
    def tearDown(self):
        pass

    def test_1_aggregate_and_training_plan(self):
        # for bug 746: incorrect handling of pytorch model having several learning rates per layer
        n_updates = 10

        # test different values for `share_persistent_buffers`
        share_persistent_buffers_options = (False, True)

        for share_persistent_buffers_option in share_persistent_buffers_options:
            tp = TestIntegrationScaffold.FakeModelTorch()
            training_args = TrainingArgs({'share_persistent_buffers': share_persistent_buffers_option}, only_required=False)
            tp.post_init({}, training_args , {})

            # create training replies
            replies = {0: {}}
            for node_id in self.node_ids:
                replies[0].update({
                    node_id: {
                        'node_id': node_id,
                        'optimizer_args': {
                            'lr' : tp.optimizer().get_learning_rate()}
                    }
                })

            global_params = tp.after_training_params()
            local_models = {node_id: copy.deepcopy(tp.after_training_params()) for i, node_id in enumerate(self.node_ids)}
            scaffold = Scaffold(server_lr =1)

            scaffold.set_fds(self.fds)
            agg_params = scaffold.aggregate(local_models,
                                            self.weights,
                                            global_model=global_params,
                                            training_plan=tp,
                                            training_replies=replies,
                                            node_ids=self.node_ids,
                                            n_updates=n_updates,
                                            n_round=0)
            for node_id in self.node_ids:
                # checking that `nodes_lr` have been populated accordingly
                self.assertDictEqual(scaffold.nodes_lr[node_id], tp.optimizer().get_learning_rate())

    def test_2_bug_977_key_error(self):
        # this bug happens when training a model with scaffold: train it, then save and load it from a breakpoint,
        # and resume the training: a key error happen, complaining that node_id doesnot exist in dictionary
        # this test intent to reproduce the error
        tp = TestIntegrationScaffold.FakeModelTorch()
        training_args = TrainingArgs({}, only_required=False)
        tp.post_init({}, training_args , {})
        n_updates = 10
        server_lr = 1
        # step 1: do some training
        replies = {0: {}}
        for node_id in self.node_ids:
            replies[0].update({
                node_id: {
                    'node_id': node_id,
                    'optimizer_args': {
                        'lr' : tp.optimizer().get_learning_rate()}
                }
            })

        global_params = tp.after_training_params()
        local_models = {node_id: copy.deepcopy(tp.after_training_params()) for i, node_id in enumerate(self.node_ids)}
        scaffold = Scaffold(server_lr =server_lr)

        scaffold.set_fds(self.fds)
        agg_params = scaffold.aggregate(local_models,
                                        self.weights,
                                        global_model=global_params,
                                        training_plan=tp,
                                        training_replies=replies,
                                        node_ids=self.node_ids,
                                        n_updates=n_updates,
                                        n_round=0)

        # step 2: save breakpoint
        with tempfile.TemporaryDirectory() as tmp_path:
            saved_state = scaffold.save_state_breakpoint(
                tmp_path,
                global_params,
            )

            del scaffold, global_params, local_models

            # step 3: load scaffold from breakpoint
            ## re-instantiate scaffold
            loaded_scaffold = Scaffold(server_lr =.00123)
            loaded_scaffold.set_fds(self.fds)
            loaded_scaffold.load_state_breakpoint(saved_state)

        # step 4: simulate the execution of another round
        replies[1] = {}
        for node_id in self.node_ids:
            replies.update({
                    node_id: {
                        'node_id': node_id,
                        'optimizer_args': {
                            'lr' : tp.optimizer().get_learning_rate()}
                    }
                })

        global_params = tp.after_training_params()
        local_models = {node_id: copy.deepcopy(tp.after_training_params()) for i, node_id in enumerate(self.node_ids)}

        agg_params_2 = loaded_scaffold.aggregate(local_models,
                                                 self.weights,
                                                 global_model=global_params,
                                                 training_plan=tp,
                                                 training_replies=replies,
                                                 node_ids=self.node_ids,
                                                 n_updates=n_updates,
                                                 n_round=1)

        # checks that parameters are reloaded accordingly
        for (k_g, v_g) in loaded_scaffold.global_state.items():

            # tests that `c = 1/N sum_{i=1}^n c_i`
            self.assertTrue(torch.isclose(v_g * len(self.node_ids),
                                          torch.mean(torch.stack(
                                              [loaded_scaffold.nodes_states[node_id][k_g] for node_id in self.node_ids]
                                          ))).all())
        # delta_i = (c_i - c)
        for node_id in self.node_ids:
            for (k_ns, v_ns), (k_d, v_d) in zip(loaded_scaffold.nodes_states[node_id].items(),
                                                loaded_scaffold.nodes_deltas[node_id].items()):
                self.assertTrue(torch.isclose(
                    v_d, v_ns - loaded_scaffold.global_state[k_ns]).all())


# TODO:
# ideas for further tests:
# test 1: check that with one client only, correction terms are zeros
# test 2: check that for 2 clients, correction terms have opposite values
if __name__ == '__main__':  # pragma: no cover
    unittest.main()
