
import unittest

from fedbiomed.common.cli import CommonCLI


class TestCommonCLI(unittest.TestCase):

    def setUp(self) -> None:
        self.cli = CommonCLI()
        pass

    def tearDown(cls) -> None:
        pass

    def test_01_common_cli_getters_and_setters(self):

        self.cli.description = "My CLI"

        self.assertEqual(self.description, "My CLI")
        self.assertEqual(self.cli.parser, self.cli._parser)

if __name__ == "__main__":
    unittest.main()