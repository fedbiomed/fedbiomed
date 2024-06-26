import unittest
from fedbiomed.common.utils import quantize, multiply, divide, reverse_quantize

class TestSecaggUtils(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
    
    def tearDown(self) -> None:
        super().tearDown()

    def quantize_and_aggregate(self, weights, n_nodes, clipping_range, target_range, multipliers=None):
        # Quantize weights
        quantized_weights = [quantize(w, clipping_range, target_range) for w in weights]
        if multipliers:
            for i in range(n_nodes):
                quantized_weights[i] = multiply(quantized_weights[i], multipliers[i])
                weights[i] = multiply(weights[i], multipliers[i])
        # Sum quantized weights
        sum_quantized_weights = [sum(w) for w in zip(*quantized_weights)]
        sum_weights = [sum(w) for w in zip(*weights)]
        # Divide by the appropriate factor
        divisor = sum(multipliers) if multipliers else n_nodes
        aggregate_quantized_weights = divide(sum_quantized_weights, divisor)
        aggregate_weights = divide(sum_weights, divisor)
        # Reverse quantize
        reverse_aggregate_weights = reverse_quantize(aggregate_quantized_weights, clipping_range, target_range)
        return aggregate_weights, reverse_aggregate_weights

    def test_01_unweighted_average(self):
        n_nodes = 3
        clipping_range = 2
        target_range = 2**16
        weights = [[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]]
        aggregate_weights, reverse_aggregate_weights = self.quantize_and_aggregate(weights, n_nodes, clipping_range, target_range)
        for elem_1, elem_2 in zip(aggregate_weights, reverse_aggregate_weights):
            self.assertAlmostEqual(elem_1, elem_2, places=3)

    def test_02_weighted_average(self):
        n_nodes = 3
        clipping_range = 2
        target_range = 2**16
        weights = [[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]]
        multipliers = [1, 2, 3]
        aggregate_weights, reverse_aggregate_weights = self.quantize_and_aggregate(weights, n_nodes, clipping_range, target_range, multipliers)
        for elem_1, elem_2 in zip(aggregate_weights, reverse_aggregate_weights):
            self.assertAlmostEqual(elem_1, elem_2, places=3)
