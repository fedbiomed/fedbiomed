import unittest
import os
import tempfile

from unittest.mock import patch

from fedbiomed.common.constants import _BaseEnum
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.secagg_manager import SecaggDhManager, SecaggServkeyManager
from fedbiomed.node.secagg_manager import SecaggManager


class TestSecaggManager(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp_dir.name, 'test.json')

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_secagg_manager_01_initialization(self):

        # Test server key manager
        secagg_setup = SecaggManager(self.db, 0)()
        self.assertIsInstance(secagg_setup, SecaggServkeyManager)

        # Test DH manager
        secagg_setup = SecaggManager(self.db, 1)()
        self.assertIsInstance(secagg_setup, SecaggDhManager)

        # Raise element type error
        with self.assertRaises(FedbiomedSecaggError):
            SecaggManager(self.db, 3)()

        # Raise missing component for element type error
        with patch('fedbiomed.node.secagg_manager.SecaggElementTypes') as element_types_patch:
            class FakeSecaggElementTypes(_BaseEnum):
                DUMMY: int = 0
            element_types_patch.return_value = FakeSecaggElementTypes(0)
            element_types_patch.__iter__.return_value = [
                FakeSecaggElementTypes(0)
            ]

            with self.assertRaises(FedbiomedSecaggError):
                SecaggManager(self.db, 0)()

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
