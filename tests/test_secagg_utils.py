import copy
import unittest
from fedbiomed.common.utils import quantize, multiply, divide, reverse_quantize


class TestSecaggUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.weights_collection = (
            [[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]],
            [[.1], [.2], [.3]],
            [[.1, .2, .3]], 
            [[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [.1, .11, .12], [.1, .13, .14]],
                                   )

        self.multipliers_collection = (
            [1, 2, 3],
            [1, 2, 3],
            [2],
            [5, 4, 3, 2, 1]
        )

        self.clipping_range = 2
        self.target_range = 2**16

    def tearDown(self) -> None:
        super().tearDown()

    def quantize_and_aggregate(self, weights, n_nodes, clipping_range, target_range, multipliers=None):
        # Quantize weights
        quantized_weights = [quantize(w, clipping_range, target_range) for w in weights]
        if multipliers is not None:
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
        # here we compare result of an average of weights with the average of quantized weight that has been unqunatized
        # ie divide(weights) == reverse_quantize(divide(quantize(weights)))

        for weights in self.weights_collection:
            n_nodes = len(weights)
            aggregate_weights, reverse_aggregate_weights = self.quantize_and_aggregate(weights, n_nodes,
                                                                                       self.clipping_range, self.target_range)
            for elem_1, elem_2 in zip(aggregate_weights, reverse_aggregate_weights):
                self.assertAlmostEqual(elem_1, elem_2, places=3)

    def test_02_weighted_average(self):
        # here we compare result of a weighted sum of weights with the weighted sum of quantized weight that has been unqunatized
        # ie divide(multiply(weights)) == reverse_quantize(divide(multiply(quantize(weights))))

        for weights, multipliers in zip(self.weights_collection, self.multipliers_collection):
            n_nodes = len(weights)
            aggregate_weights, reverse_aggregate_weights = self.quantize_and_aggregate(weights, n_nodes, self.clipping_range, self.target_range, multipliers)
            for elem_1, elem_2 in zip(aggregate_weights, reverse_aggregate_weights):
                self.assertAlmostEqual(elem_1, elem_2, places=3)

    def test_03_multiply_divide(self):
        # here we test that methods multiply and divide are reversibles
        # ie weights == divide(multiply(weights))
        weights_collection = ([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]],
                              [[.11, .21, .54]],
                              [[.1], [.2], [.3]],
                              [[.1, .5, 77], [.5, .67, .44, .55, .02], [.2, .4], [.1, .1, .1, 0.]]
                              )   
        multipliers_collection = ([1, 2, 3], [3], [1, 2, 3], [1, 2, 3], [2, 3, 1, 2])

        for weights, multipliers in zip(weights_collection, multipliers_collection):
            multiplied_qtities = copy.deepcopy(weights)
            divided_qtities = copy.deepcopy(weights)
            for i in range(len(multipliers)):
                multiplied_qtities[i] = multiply(weights[i], multipliers[i])
            for i in range(len(multipliers)):
                divided_qtities[i] = divide(multiplied_qtities[i], multipliers[i])
            for w_1, w_2 in zip(weights, divided_qtities):
                for elem_1, elem_2 in zip(w_1, w_2):
                    self.assertAlmostEqual(elem_1, elem_2, places=5)
