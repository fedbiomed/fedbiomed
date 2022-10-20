import unittest
from unittest.mock import patch

import testsupport.mock_node_environ  ## noqa (remove flake8 false warning)

from fedbiomed.node.environ import environ
from fedbiomed.node.secagg import SecaggServkeySetup, SecaggBiprimeSetup

class TestSecaggResearcher(unittest.TestCase):
    """ Test for researcher's secagg module"""

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self):
        pass

    def tearDown(self) -> None:
        pass

    def test_secagg_01_init(self):
        """Instantiate secagg classes"""

        # prepare
        kwargs = {
            'researcher_id': "my researcher",
            'secagg_id': "my secagg",
            'sequence': 123,
            'parties': ['my party'],
        }

        # test
        secagg = SecaggServkeySetup(**kwargs)

        # check
        self.assertEqual(secagg.researcher_id(), kwargs['researcher_id'])
        self.assertEqual(secagg.secagg_id(), kwargs['secagg_id'])
        self.assertEqual(secagg.sequence(), kwargs['sequence'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
