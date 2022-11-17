
from curses import update_lines_cols
from platform import node
from re import U
import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.functional import federated_averaging
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.responses import Responses
import torch
from torch.nn import Linear
import numpy as np
from fedbiomed.researcher.aggregators.scaffold import Scaffold

import copy
import random


import testsupport.mock_researcher_environ  # noqa (remove flake8 false warning)



class TestScaffold(unittest.TestCase):
    '''
    Tests the Scaffold class
    '''

    # before the tests
    def setUp(self):
        self.model = Linear(10, 3)
        self.n_nodes = 4
        self.node_ids = [f'node_{i}'for i in range(self.n_nodes)] 
        self.models = [{node_id: Linear(10, 3).state_dict()} for i, node_id in enumerate(self.node_ids)]
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
        # if learning rate of server equals 0, scaled models equals inital model
        agg.server_lr = 0
        
        model_params = {list(node_content.keys())[0]: list(node_content.values())[0] for node_content in self.models}
        
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
        
        weights = [{node_id: 1./self.n_nodes} for node_id in self.node_ids]
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
        for (k,v) in agg.nodes_correction_states.items():
            
            for layer in v.values():
                self.assertFalse(torch.nonzero(layer).all())
        # TODO: test methods when proportions are differents

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
    
    def test_6_create_aggregator_args(self):
        agg = Scaffold()
        agg_thr_msg, agg_thr_file = agg.create_aggregator_args(self.model.state_dict(),
                                                               self.node_ids)
        
        for node_id in self.node_ids:
            for (k, v), (k0, v0) in zip(agg.nodes_correction_states[node_id].items(), self.zero_model.state_dict().items()):
                self.assertTrue(torch.isclose(v, v0).all())

# TODO:
# ideas for further tests:
# test 1: check that with one client only, correction terms are zeros
# test 2: check that for 2 clients, correction terms have opposite values
if __name__ == '__main__':  # pragma: no cover
    unittest.main()
