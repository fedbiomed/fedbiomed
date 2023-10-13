import unittest
from unittest.mock import patch


class MockRequestGrpc(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.messaging_patch = patch("fedbiomed.researcher.requests.GrpcServer")
        cls.messaging_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        super().setUpClass()
        cls.messaging_patch.stop()
