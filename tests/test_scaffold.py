
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

    def test_1_scaling(self):
        agg = Scaffold(server_lr=.1)
        # boundary conditions
        # if learning rate of server equals 0, scaled models equals initial model
        agg.server_lr = 0
        
        model_params = self.models
        scaled_models = agg.scaling(copy.deepcopy(model_params), self.model.state_dict())
        
        for k, v in scaled_models.items():
            self.assertTrue(torch.isclose(v, self.model.state_dict()[k]).all())

        # if learning rate  of server equals 1, scaled models equals average of local models
        # (assuming `federated_averaging`` function is working)
        agg = Scaffold(server_lr=1)
        scaled_models = agg.scaling(model_params, self.model.state_dict())
        model_params2 = [v for  v in model_params.values()]
        avg = federated_averaging(model_params2, [1/self.n_nodes] * self.n_nodes)
      
        for (k_t, v_t), (k_i, v_i) in zip(scaled_models.items(), avg.items()):

            self.assertTrue(torch.isclose(v_t, v_i).all())

    @patch('fedbiomed.researcher.datasets.FederatedDataSet.node_ids')        
    def test_2_update_correction_state(self, mock_federated_dataset):
        mock_federated_dataset.return_value = self.node_ids

        # case where N = S (all nodes are involved in Aggregator)
        agg = Scaffold(server_lr=1.)
        agg.init_correction_states(self.model.state_dict(), self.node_ids)  # settig correction parameters to 0
        agg.nodes_lr = { k :[1] * self.n_nodes for k in self.node_ids}
        
        fds = FederatedDataSet({})
        agg.set_fds(fds)
        agg.update_correction_states({node_id: self.zero_model.state_dict() for node_id in self.node_ids},
                                     self.model.state_dict(), n_updates=1)

        for node_id in self.node_ids:
            for (k, v), (k_i, v_i) in zip(agg.global_state.items(), self.model.state_dict().items()):
                self.assertTrue(torch.isclose(v, v_i).all())
        # let's do another update where corrections are non zeros tensors (corection terms should cancel out)
                
        agg.update_correction_states({node_id: self.zero_model.state_dict() for node_id in self.node_ids},
                                     self.model.state_dict(), n_updates=1)


        for (k, v), (k_i, v_i) in zip(agg.global_state.items(), self.model.state_dict().items()):
            self.assertTrue(torch.isclose(v, v_i).all())
                
        # case where model has not been updated: it implies correction are set to 0 (if N==S)
        agg.init_correction_states(self.model.state_dict(), self.node_ids)
        global_correction_term_before_update = copy.deepcopy(agg.global_state)

        agg.update_correction_states({node_id: self.model.state_dict() for node_id in self.node_ids},
                                     self.model.state_dict(), n_updates=random.randint(1, 10))

        for (k, v), (k_i, v_i) in zip(agg.global_state.items(), global_correction_term_before_update.items()):

            self.assertTrue(torch.isclose(v , v_i).all())
                
                
        # case where there is more than one update (4 updates): correction terms should be devided by 4 (ie by n_updates)
        n_updates = 4
        agg.init_correction_states(self.model.state_dict(), self.node_ids)
        correction_terms_before_update = copy.deepcopy(agg.nodes_correction_states)
        agg.update_correction_states({node_id: self.zero_model.state_dict() for node_id in self.node_ids},
                                     self.model.state_dict(), n_updates=n_updates)

        for (k, v), (k_i, v_i) in zip(agg.global_state.items(), self.model.state_dict().items()):
            self.assertTrue(torch.isclose(v , v_i / n_updates).all())

    @patch('fedbiomed.researcher.datasets.FederatedDataSet.node_ids')      
    def test_3_update_correction_state_2(self, mock_federated_dataset):
        mock_federated_dataset.return_value = self.node_ids
        # case where S = 2 (only 2 nodes are selected during the round) and there are no updates
        # (meaning ACG_i = 0), then, new correction terms equals 1/2 * former correction terms
        S = 2

        agg = Scaffold(server_lr=.2)
        fds = FederatedDataSet({})
        agg.set_fds(fds)
        agg.init_correction_states(self.model.state_dict(), self.node_ids)
        current_round_nodes = random.sample(self.node_ids, k=S)
        agg.nodes_lr = { k :[.1] * self.n_nodes for k in self.node_ids}

        agg.update_correction_states({node_id: Linear(10, 3).state_dict() for node_id in self.node_ids},
                                     self.zero_model.state_dict(), n_updates=1)  # making correction terms non zeros
        global_state_terms_before_update = copy.deepcopy(agg.global_state)

        agg.update_correction_states({node_id: self.model.state_dict() for node_id in current_round_nodes},
                                     self.model.state_dict(), n_updates=1)
        
        for (k, v), (k_i, v_i) in zip(agg.global_state.items(), global_state_terms_before_update.items()):
            self.assertTrue(torch.isclose(v, .5 * v_i ).all())

        # TODO: check also for node_correction states

    @patch('fedbiomed.researcher.datasets.FederatedDataSet.node_ids')  
    def test_4_aggregate(self, mock_federated_dataset):
        
        mock_federated_dataset.return_value = self.node_ids

        training_plan = MagicMock()
        training_plan.get_model_params = MagicMock(return_value = self.node_ids)

        agg = Scaffold(server_lr=.2)
        fds = FederatedDataSet({})
        agg.set_fds(fds)
        n_round = 0
        
        weights = {node_id: 1./self.n_nodes for node_id in self.node_ids}
        # assuming that global model has all its coefficients to 0
        aggregated_model_params_scaffold = agg.aggregate(copy.deepcopy(self.models),
                                                         weights,
                                                         copy.deepcopy(self.zero_model.state_dict()),
                                                         training_plan,
                                                         self.responses,
                                                         self.node_ids,
                                                         n_round=n_round)

        aggregated_model_params_fedavg = FedAverage().aggregate(copy.deepcopy(self.models), weights)
        # we check that fedavg and scaffold give proportional results provided:
        # - all previous correction state model are set to 0 (round 0)
        # - model proportions are the same
        # then:
        # fedavg: x_i <- x_i / n_nodes
        # scaffold: x_i <- server_lr * x_i / n_nodes
        for (k,v), (k_i, v_i) in zip(aggregated_model_params_scaffold.items(),
                                     aggregated_model_params_fedavg.items()):

            self.assertTrue(torch.isclose(v, v_i * .2).all())
            
        # check that at the end of aggregation, all correction states are non zeros (
        for (k, v) in agg.nodes_correction_states.items():
            
            for layer in v.values():
                self.assertFalse(torch.nonzero(layer).all())
        # TODO: test methods when proportions are different

    def test_5_setting_scaffold_with_wrong_parameters(self):
        """test_5_setting_scaffold_with_wrong_parameters: tests that scaffold is
        returning an error when set with incorrect parameters
        """
        #  test 1: `server_lr` should be different than 0
        for x in (0, 0.):
            with self.assertRaises(FedbiomedAggregatorError):
                Scaffold(server_lr = x)
                
        # test 2: calling `update_correction_states` without any federated dataset
        with self.assertRaises(FedbiomedAggregatorError):
            scaffold = Scaffold()
            scaffold.update_correction_states({node_id: self.model.state_dict() for node_id in self.node_ids},
                                              self.model.state_dict())
        # test 3: `n_updates` should be a positive and non zero integer
        training_plan = MagicMock()
        for x in (-1, .2, 0, 0., -3.2):
            with self.assertRaises(FedbiomedAggregatorError):
                scaffold = Scaffold()
                scaffold.check_values(n_updates=x, training_plan=training_plan)
                
        # test 4: `FederatedDataset` has not been specified
        
        with self.assertRaises(FedbiomedAggregatorError):
            scaffold = Scaffold()
            # scaffold._fds = None
            scaffold.check_values(n_updates=1, training_plan=training_plan)
        with self.assertRaises(FedbiomedAggregatorError):
            scaffold = Scaffold()
            scaffold.check_values(n_updates=None, training_plan=training_plan)
    
    def test_6_create_aggregator_args(self):
        agg = Scaffold()
        agg_thr_msg, agg_thr_file = agg.create_aggregator_args(self.model.state_dict(),
                                                               self.node_ids)
        
        for node_id in self.node_ids:
            for (k, v), (k0, v0) in zip(agg.nodes_correction_states[node_id].items(),
                                        self.zero_model.state_dict().items()):
                self.assertTrue(torch.isclose(v, v0).all())
                
                
        # check that each element returned by method contains key 'aggregator_name'
        for node_id in self.node_ids:
            self.assertTrue(agg_thr_msg[node_id].get('aggregator_name', False))
        
            self.assertTrue(agg_thr_file[node_id].get('aggregator_name', False))
        
        # check `agg_thr_file` contains node correction state
        for node_id in self.node_ids:
            self.assertDictEqual(agg_thr_file[node_id]['aggregator_correction'], agg.nodes_correction_states[node_id])
            
        # checking case where a node has been added to the training (repeating same tests above)
        self.n_nodes += 1
        self.node_ids.append(f'node_{self.n_nodes}')
        agg_thr_msg, agg_thr_file = agg.create_aggregator_args(self.model.state_dict(),
                                                               self.node_ids)
        
        for node_id in self.node_ids:
            self.assertTrue(agg_thr_msg[node_id].get('aggregator_name', False))
            self.assertTrue(agg_thr_file[node_id].get('aggregator_name', False))
        
        # check `agg_thr_file` contains node correction state
        for node_id in self.node_ids:
            self.assertDictEqual(agg_thr_file[node_id]['aggregator_correction'], agg.nodes_correction_states[node_id])

    @patch('uuid.uuid4')
    def test_7_save_state(self, uuid_patch):
        uuid_patch.return_value = FakeUuid()
        server_lr = .5
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        bkpt_path = '/path/to/my/breakpoint'
        training_plan = MagicMock()
        scaffold = Scaffold(server_lr, fds=fds)
        scaffold.init_correction_states(self.model.state_dict(), self.node_ids)
        state = scaffold.save_state(training_plan=training_plan,
                                    breakpoint_path=bkpt_path,
                                    global_model=self.model.state_dict())
        self.assertEqual(training_plan.save.call_count, self.n_nodes + 1,
                        f"training_plan 'save' method should be called {self.n_nodes} times for each nodes + \
                        one more time for global_state")

        for node_id in self.node_ids:
            self.assertEqual(state['parameters']['aggregator_correction'][node_id],
                             os.path.join(bkpt_path, 'aggregator_correction_' + str(node_id) + '.pt'))
        
        self.assertEqual(state['parameters']['server_lr'], server_lr)
        self.assertEqual(state['parameters']['global_state_filename'], os.path.join(bkpt_path,
                                                                                    'global_state_'
                                                                                    + str(FakeUuid.VALUE) + '.pt'))
        self.assertEqual(state['class'], Scaffold.__name__)
        self.assertEqual(state['module'], Scaffold.__module__)

    def test_8_load_state(self):
        """test_8_load_state: check how many time `save` method of training plan is called"""
        server_lr = .5
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        bkpt_path = '/path/to/my/breakpoint'
        training_plan = MagicMock()
        scaffold = Scaffold(server_lr, fds=fds)
        
        # first we create a state before trying to load it
        scaffold.init_correction_states(self.model.state_dict(), self.node_ids)
        state = scaffold.save_state(training_plan=training_plan,
                                    breakpoint_path=bkpt_path,
                                    global_model=self.model.state_dict())
        
        training_plan.reset_mock()
        # action
        scaffold.load_state(state, training_plan)
        
        training_plan.load.return_value = self.model.state_dict()
        self.assertEqual(training_plan.load.call_count, self.n_nodes + 1,
                         f"training_plan 'load' method should be called {self.n_nodes} times for each nodes + \
                         one more time for global_state")

    def test_9_load_state_2(self):
        """test_9_load_state_2: check loaded parameters are correctly reloaded"""
        server_lr = .5
        fds = FederatedDataSet({node_id: {} for node_id in self.node_ids})
        bkpt_path = '/path/to/my/breakpoint'
        training_plan = MagicMock()
        scaffold = Scaffold(server_lr, fds=fds)
        
        # first we create a correction state before trying to load it
        #scaffold.init_correction_states(self.model.state_dict(), self.node_ids)
        state = scaffold.save_state(training_plan=training_plan,
                                    breakpoint_path=bkpt_path,
                                    global_model=self.model.state_dict())
        
        training_plan.load = MagicMock(return_value=self.model.state_dict())

        # action
        scaffold.load_state(state, training_plan)
        # tests
        for node_id in self.node_ids:
            for (k,v), (k_ref, v_ref) in zip(scaffold.nodes_correction_states[node_id].items(), 
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
