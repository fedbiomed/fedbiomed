import unittest
from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.training_plans import SKLearnTrainingPlan, TorchTrainingPlan
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.datasets import FederatedDataSet


class TestAggregator(unittest.TestCase):
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

    def test_aggregator_01_normalize_weights(self):

        for i in range(0, len(self.weights)):

            results = Aggregator.normalize_weights(self.weights[i])

            # results has the same size than input
            self.assertEqual(len(results), len(self.weights[i]))

            # sum of results[] must be equal tu 1.0
            sum = 0.0
            for j in range(0, len(results)):
                sum = sum + results[j];

            # results is a normalized list of values
            self.assertEqual(sum, 1.0)

        # special case: empty list
        weights = []
        results = Aggregator.normalize_weights(weights)
        self.assertEqual(len(results), 0)

        # special case: null content
        weights = [0.0, 0.0]
        results = Aggregator.normalize_weights(weights)
        self.assertEqual(len(results), len(weights))
        self.assertEqual(results[0], 1.0 / len(weights))

    def test_aggregator_02_save_state(self):

        expected_state = {'class': 'Aggregator',
                          'module': 'fedbiomed.researcher.aggregators.aggregator',
                          'parameters': None}

        state = self.aggregator.save_state()
        self.assertDictEqual(expected_state, state, 'State of aggregator has not been saved correctly')

    def test_aggregator_03_load_state(self):

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
