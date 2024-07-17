import unittest
from unittest.mock import patch, MagicMock, PropertyMock

from fedbiomed.researcher.requests import FederatedRequest, PolicyController

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


class MockRequestModule:

    def setUp(self, module = None, send_fed_req_conf: dict = {}) -> None:

        module = module if module else "fedbiomed.researcher.requests.Requests"
        self.patch_requests = patch(module)

        self.mock_requests = self.patch_requests.start()

        self.mock_federated_request = MagicMock(spec=FederatedRequest, **send_fed_req_conf)
        self.mock_policy  = MagicMock(spec=PolicyController)
        self.fake_search_reply = {}
        type(self.mock_federated_request).policy  = PropertyMock(return_value=self.mock_policy)
        self.mock_requests.return_value.send.return_value = self.mock_federated_request
        self.mock_requests.return_value.search.return_value = self.fake_search_reply
        self.mock_policy.has_stopped_any.return_value = False
        self.mock_federated_request.__enter__.return_value = self.mock_federated_request

    def tearDown(self) -> None:
        self.patch_requests.stop()

