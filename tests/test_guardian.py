import hashlib
import unittest
from unittest.mock import patch

import requests

from fedbiomed.common.exceptions import FedbiomedGuardianError
from fedbiomed.node.guardian import (
    GuardianClient,
    compute_training_plan_checksum,
)


class FakeGuardianResponse:
    """Mimics a `requests.Response` returned by the guardian service."""

    def __init__(self, status_code=200, payload=None, raises=False):
        self.status_code = status_code
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises:
            raise ValueError("not a JSON payload")
        return self._payload


class TestComputeTrainingPlanChecksum(unittest.TestCase):
    def test_checksum_01_matches_plain_sha256(self):
        """Checksum is a plain sha256 a third party can reproduce"""
        source = "class MyTrainingPlan:\n    pass\n"
        self.assertEqual(
            compute_training_plan_checksum(source),
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )

    def test_checksum_02_is_deterministic(self):
        """Same source always yields the same checksum"""
        source = "class MyTrainingPlan:\n    pass\n"
        self.assertEqual(
            compute_training_plan_checksum(source),
            compute_training_plan_checksum(source),
        )

    def test_checksum_03_differs_on_any_change(self):
        """A single character change yields a different checksum"""
        self.assertNotEqual(
            compute_training_plan_checksum("class A: pass"),
            compute_training_plan_checksum("class B: pass"),
        )

    def test_checksum_04_bad_type(self):
        """Non-string source raises FedbiomedGuardianError"""
        for source in (None, 12, b"bytes", ["a"]):
            with self.assertRaises(FedbiomedGuardianError):
                compute_training_plan_checksum(source)


class TestGuardianClient(unittest.TestCase):
    def setUp(self):
        self.checksum = "a" * 64
        self.capabilities = {"training_plan_checksum": self.checksum}
        self.client = GuardianClient("http://localhost:8000")

        self.post_patch = patch("fedbiomed.node.guardian.requests.post")
        self.post_mock = self.post_patch.start()

    def tearDown(self):
        self.post_patch.stop()

    def test_client_01_bad_service_url(self):
        """An empty or non-string service URL is refused"""
        for url in ("", "   ", None, 8000):
            with self.assertRaises(FedbiomedGuardianError):
                GuardianClient(url)

    def test_client_02_verify_url(self):
        """Trailing slashes are normalized and /verify is appended"""
        self.assertEqual(
            GuardianClient("http://localhost:8000").verify_url(),
            "http://localhost:8000/verify",
        )
        self.assertEqual(
            GuardianClient(" http://localhost:8000/// ").verify_url(),
            "http://localhost:8000/verify",
        )

    def test_client_03_valid_capability(self):
        """A positive answer is returned with its reason"""
        self.post_mock.return_value = FakeGuardianResponse(
            payload={"valid": True, "reason": "checksums match"}
        )

        valid, reason = self.client.verify(
            capabilities=self.capabilities,
            checksum=self.checksum,
            context={"node_id": "node-1"},
        )

        self.assertTrue(valid)
        self.assertEqual(reason, "checksums match")

    def test_client_04_request_payload(self):
        """Both checksums and the round context are sent to /verify"""
        self.post_mock.return_value = FakeGuardianResponse(payload={"valid": True})

        self.client.verify(
            capabilities=self.capabilities,
            checksum=self.checksum,
            context={"node_id": "node-1", "round": 3, "training": False},
        )

        args, kwargs = self.post_mock.call_args
        self.assertEqual(args[0], "http://localhost:8000/verify")
        self.assertEqual(kwargs["json"]["capabilities"], self.capabilities)
        self.assertEqual(kwargs["json"]["training_plan_checksum"], self.checksum)
        self.assertEqual(kwargs["json"]["node_id"], "node-1")
        self.assertEqual(kwargs["json"]["round"], 3)
        self.assertIs(kwargs["json"]["training"], False)
        self.assertIn("timeout", kwargs)

    def test_client_05_context_cannot_override_checksum(self):
        """Context entries never shadow the node computed checksum"""
        self.post_mock.return_value = FakeGuardianResponse(payload={"valid": True})

        self.client.verify(
            capabilities=self.capabilities,
            checksum=self.checksum,
            context={"training_plan_checksum": "spoofed", "capabilities": {"x": 1}},
        )

        _, kwargs = self.post_mock.call_args
        self.assertEqual(kwargs["json"]["training_plan_checksum"], self.checksum)
        self.assertEqual(kwargs["json"]["capabilities"], self.capabilities)

    def test_client_06_rejected_capability(self):
        """A negative answer is returned rather than raising"""
        self.post_mock.return_value = FakeGuardianResponse(
            payload={"valid": False, "reason": "checksum mismatch"}
        )

        valid, reason = self.client.verify(self.capabilities, self.checksum, {})

        self.assertFalse(valid)
        self.assertEqual(reason, "checksum mismatch")

    def test_client_07_missing_reason(self):
        """A missing reason falls back to an empty string"""
        self.post_mock.return_value = FakeGuardianResponse(payload={"valid": False})

        valid, reason = self.client.verify(self.capabilities, self.checksum, {})

        self.assertFalse(valid)
        self.assertEqual(reason, "")

    def test_client_08_non_200_status(self):
        """A non success status code raises FedbiomedGuardianError"""
        for status_code in (400, 403, 500):
            self.post_mock.return_value = FakeGuardianResponse(
                status_code=status_code, payload={"valid": True}
            )
            with self.assertRaises(FedbiomedGuardianError):
                self.client.verify(self.capabilities, self.checksum, {})

    def test_client_09_non_json_payload(self):
        """A body that is not JSON raises FedbiomedGuardianError"""
        self.post_mock.return_value = FakeGuardianResponse(raises=True)

        with self.assertRaises(FedbiomedGuardianError):
            self.client.verify(self.capabilities, self.checksum, {})

    def test_client_10_malformed_payload(self):
        """A payload without a boolean 'valid' entry raises rather than passing"""
        for payload in (
            {},
            {"reason": "ok"},
            {"valid": "true"},
            {"valid": 1},
            [],
            None,
        ):
            self.post_mock.return_value = FakeGuardianResponse(payload=payload)
            with self.assertRaises(FedbiomedGuardianError):
                self.client.verify(self.capabilities, self.checksum, {})

    def test_client_11_transport_errors(self):
        """Network level failures raise FedbiomedGuardianError"""
        for exception in (
            requests.exceptions.ConnectionError("refused"),
            requests.exceptions.Timeout("timed out"),
            requests.exceptions.RequestException("boom"),
        ):
            self.post_mock.side_effect = exception
            with self.assertRaises(FedbiomedGuardianError):
                self.client.verify(self.capabilities, self.checksum, {})


if __name__ == "__main__":
    unittest.main()
