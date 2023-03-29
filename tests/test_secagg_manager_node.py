import unittest
from unittest.mock import patch

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from fedbiomed.common.constants import _BaseEnum
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.secagg_manager import SecaggServkeyManager, SecaggBiprimeManager
from fedbiomed.node.secagg_manager import SecaggManager


class TestSecaggManager(NodeTestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_secagg_manager_01_initialization(self):

        # Test server key manager
        secagg_setup = SecaggManager(0)()
        self.assertIsInstance(secagg_setup, SecaggServkeyManager)

        # Test biprime manager
        secagg_setup = SecaggManager(1)()
        self.assertIsInstance(secagg_setup, SecaggBiprimeManager)

        # Raise element type erro
        with self.assertRaises(FedbiomedSecaggError):
            SecaggManager(2)()

        # Raise missing component for element type error
        with patch('fedbiomed.node.secagg_manager.SecaggElementTypes') as element_types_patch:
            class FakeSecaggElementTypes(_BaseEnum):
                DUMMY: int = 0
            element_types_patch.return_value = FakeSecaggElementTypes(0)
            element_types_patch.__iter__.return_value = [
                FakeSecaggElementTypes(0)
            ]

            with self.assertRaises(FedbiomedSecaggError):
                SecaggManager(0)()        

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
