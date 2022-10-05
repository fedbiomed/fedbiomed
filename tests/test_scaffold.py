
from curses import update_lines_cols
from platform import node
from re import U
import unittest
from unittest.mock import MagicMock, patch
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.datasets import FederatedDataSet
import torch
from torch.nn import Linear
import numpy as np
from fedbiomed.researcher.aggregators.scaffold import Scaffold

import copy
from random import random, randint, sample
#import random

import testsupport.mock_researcher_environ  # noqa (remove flake8 false warning)



class TestScaffold(unittest.TestCase):
    '''
    Test the Scaffold class
    '''

    # before the tests
    def setUp(self):
        self.model = Linear(10, 3)
        self.n_nodes = 4
        self.node_ids = [f'node_{i}'for i in range(self.n_nodes)] 
        self.models = [{node_id: Linear(10, 3).state_dict()} for i, node_id in enumerate(self.node_ids)]
        self.zero_model = copy.deepcopy(self.model)  # model where all parameters are equals to 0
        
        self.weights = [random() for _ in self.models]
        
        # setting all coefficients of `zero_model` to 0
        for p in self.zero_model.parameters():
            p.data.fill_(0)
        print("ZERO MODEL", self.zero_model.state_dict())

    # after the tests
    def tearDown(self):
        pass
    
    
    def test_1_scaling(self):
        agg = Scaffold(server_lr=.1)
        # boundary conditions
        # if learning rate of server equals 0, scaled models equals inital model
        agg.server_lr = 0
        scaled_models = agg.scaling(self.models, self.model.state_dict())
        for i  in range(self.n_nodes):
            node_id = list(scaled_models[i].keys())[0]
            for k, v in scaled_models[i][node_id].items():
                self.assertTrue(torch.isclose(v, self.model.state_dict()[k]).all())

        # if learning rate  of server equals 1, scaled models equals local models
        agg = Scaffold(server_lr=1)
        scaled_models = agg.scaling(self.models, self.model.state_dict())
        for i  in range(self.n_nodes):
            node_id = list(scaled_models[i].keys())[0]
            for (k_t, v_t), (k_i, v_i) in zip(scaled_models[i][node_id].items(), self.models[i][node_id].items()):
                self.assertTrue(torch.isclose(v_t, v_i).all())

    @patch('fedbiomed.researcher.datasets.FederatedDataSet.node_ids')        
    def test_2_update_correction_state(self, mock_federated_dataset):
        mock_federated_dataset.return_value = self.node_ids
        mock_fds = MagicMock()
        # case where N = S (all nodes are involved in Aggregator)
        agg = Scaffold(server_lr=1.)
        agg.init_correction_states(self.model.state_dict(), self.node_ids)  # settig correction parameters to 0
        agg.nodes_lr = { k :[1] * self.n_nodes for k in self.node_ids}
        
        fds = FederatedDataSet({})
        agg.set_fds(fds)
        agg.update_correction_states(self.zero_model.state_dict(), self.model.state_dict(),self.node_ids, n_updates=1)

        for node_id in self.node_ids:
            for (k, v), (k_i, v_i) in zip(agg.nodes_correction_states[node_id].items(), self.model.state_dict().items()):
                self.assertTrue(torch.isclose(v, v_i).all())
        # let's do another update where corrections are non zeros tensors (corection terms should cancel out)
                
        agg.update_correction_states(self.zero_model.state_dict(), self.model.state_dict(), self.node_ids, n_updates=1)

        for node_id in self.node_ids:
            for (k, v), (k_i, v_i) in zip(agg.nodes_correction_states[node_id].items(), self.model.state_dict().items()):
                self.assertTrue(torch.isclose(v, v_i).all())
                
        # case where model has not been updated: it implies correction are set to 0
        agg.init_correction_states(self.model.state_dict(), self.node_ids)
        correction_terms_before_update = copy.deepcopy(agg.nodes_correction_states)

        agg.update_correction_states(self.model.state_dict(), self.model.state_dict(), self.node_ids, n_updates=randint(1, 10))
        for node_id in self.node_ids:
            for (k, v), (k_i, v_i) in zip(agg.nodes_correction_states[node_id].items(), correction_terms_before_update[node_id].items()):

                self.assertTrue(torch.isclose(v , v_i).all())

    @patch('fedbiomed.researcher.datasets.FederatedDataSet.node_ids')      
    def test_3_update_correction_state(self, mock_federated_dataset):
        mock_federated_dataset.return_value = self.node_ids
        # case where S = 2 (only 2 nodes are selected during the round) and there are no updates
        # then, new correction terms equals 1/2 * former correction terms
        S = 2
        ## resetting
        agg = Scaffold(server_lr=.2)
        fds = FederatedDataSet({})
        agg.set_fds(fds)
        agg.init_correction_states(self.model.state_dict(), self.node_ids)
        current_round_nodes = sample(self.node_ids, k=S)
        agg.nodes_lr = { k :[.1] * self.n_nodes for k in self.node_ids}
        print(self.model.state_dict(), self.zero_model.state_dict())
        print("NODE_ID", self.node_ids)
        agg.update_correction_states(Linear(10, 3).state_dict(), self.zero_model.state_dict(),self.node_ids)  # making correction terms non zeros
        correction_terms_before_update = copy.deepcopy(agg.nodes_correction_states)
        print(agg.nodes_correction_states)
        print("ZERO", self.zero_model.state_dict())
        agg.update_correction_states(self.model.state_dict(), self.model.state_dict(), current_round_nodes, n_updates=1)
        for node_id in self.node_ids:
            for (k, v), (k_i, v_i) in zip(agg.nodes_correction_states[node_id].items(), correction_terms_before_update[node_id].items()):
                print(v, "VALUE", v_i)
                self.assertTrue(torch.isclose(v, .5 * v_i ).all())
        
        
    def test_3_aggregate(self):
        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
