from testsupport.base_case import ResearcherTestCase

import unittest
from fedbiomed.common.constants import TrainingPlans
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.datasets import FederatedDataSet


class TestAggregator(ResearcherTestCase):
    '''
    Test the Aggregator class
    '''

    # before the tests
    def setUp(self):
        self.weights = [
            [1.0, -1.0],  # what happens ? should we code it/test it ?
            [2.0],
            [0.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
        ]

        self.aggregator = Aggregator()

    # after the tests
    def tearDown(self):
        pass

    def test_aggregator_01_save_state(self):

        expected_state = {'class': 'Aggregator',
                          'module': 'fedbiomed.researcher.aggregators.aggregator',
                          'parameters': None}

        state = self.aggregator.save_state()
        self.assertDictEqual(expected_state, state, 'State of aggregator has not been saved correctly')

    def test_aggregator_02_load_state(self):

        state = {
            'parameters': {'param' : True}
        }
        self.aggregator.load_state(state)
        self.assertDictEqual(self.aggregator._aggregator_args,
                             state['parameters'],
                             'The state of the aggregator class has not been loaded correctly')

    def test_3_set_training_plan_type(self):
        self.aggregator.set_training_plan_type(TrainingPlans.SkLearnTrainingPlan)
        self.assertEqual(self.aggregator._training_plan_type, TrainingPlans.SkLearnTrainingPlan)
        self.aggregator.set_training_plan_type(TrainingPlans.TorchTrainingPlan)
        self.assertEqual(self.aggregator._training_plan_type, TrainingPlans.TorchTrainingPlan)

    def test_4_set_fds(self):
        fds = FederatedDataSet({})
        self.aggregator.set_fds(fds)
        self.assertEqual(fds, self.aggregator._fds)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
