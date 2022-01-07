# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

import unittest

from fedbiomed.researcher.aggregators.aggregator import Aggregator

class TestAggregator(unittest.TestCase):
    '''
    Test the Aggregator class
    '''

    # before the tests
    def setUp(self):
        self.weights = [
            [ 1.0, -1.0 ] , # what happends ? should we code it/test it ?
            [ 2.0 ],
            [ 0.0, 1.0],
            [ 1.0, 2.0, 3.0, 4.0 ],
        ]


    # after the tests
    def tearDown(self):
        pass

    def test_aggregator(self):

        for i in range(0, len(self.weights)):

            results = Aggregator.normalize_weights(self.weights[i])

            # results has the same size than input
            self.assertEqual( len(results) , len(self.weights[i]))

            # sum of results[] must be equal tu 1.0
            sum = 0.0
            for j in range(0, len(results)):
                sum = sum + results[j];

            # results is a normalized list of values
            self.assertEqual( sum , 1.0)

        # special case: empty list
        weights = []
        results = Aggregator.normalize_weights(weights)
        self.assertEqual( len(results) , 0 )

        # special case: null content
        weights = [ 0.0, 0.0 ]
        results = Aggregator.normalize_weights(weights)
        self.assertEqual( len(results) , len(weights) )
        self.assertEqual( results[0] , 1.0 / len(weights) )

if __name__ == '__main__': # pragma: no cover
    unittest.main()
