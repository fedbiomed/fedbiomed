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

    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class TestComputeTrainingPlanChecksum(unittest.TestCase):
    def test_checksum_01_matches_plain_sha256(self):
        """Checksum is a prefixed plain sha256 a third party can reproduce"""
        source = "class MyTrainingPlan:\n    pass\n"
        self.assertEqual(
            compute_training_plan_checksum(source),
            "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest(),
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
        self.checksum = "sha256:" + "a" * 64
        self.capability = "eyJtaW50ZWRfaWRlbnRpdHkiOiAiYWJjIn0="
        self.client = GuardianClient("http://localhost:7900")

        self.post_patch = patch("fedbiomed.node.guardian.requests.post")
        self.post_mock = self.post_patch.start()

    def tearDown(self):
        self.post_patch.stop()

    def test_client_01_bad_service_url(self):
        """An empty or non-string service URL is refused"""
        for url in ("", "   ", None, 7900):
            with self.assertRaises(FedbiomedGuardianError):
                GuardianClient(url)

    def test_client_02_process_capability_url(self):
        """Trailing slashes are normalized and /process_capability is appended"""
        self.assertEqual(
            GuardianClient("http://localhost:7900").process_capability_url(),
            "http://localhost:7900/process_capability",
        )
        self.assertEqual(
            GuardianClient(" http://localhost:7900/// ").process_capability_url(),
            "http://localhost:7900/process_capability",
        )

    def test_client_03_allowed_capability(self):
        """A success status authorizes the round and returns the answer as reason"""
        self.post_mock.return_value = FakeGuardianResponse(
            status_code=200, text='{"allowed": true, "mock": true}\n'
        )

        valid, reason = self.client.verify(self.capability, self.checksum)

        self.assertTrue(valid)
        self.assertEqual(reason, '{"allowed": true, "mock": true}')

    def test_client_04_request_payload(self):
        """The opaque capability and the node computed digest are sent as-is"""
        self.post_mock.return_value = FakeGuardianResponse(status_code=200)

        self.client.verify(self.capability, self.checksum)

        args, kwargs = self.post_mock.call_args
        self.assertEqual(args[0], "http://localhost:7900/process_capability")
        self.assertEqual(
            kwargs["json"],
            {
                "capability_b64": self.capability,
                "request_context": {"calculated_script_digest": self.checksum},
            },
        )
        self.assertIn("timeout", kwargs)

    def test_client_05_rejected_capability(self):
        """A 422 answer is a refusal, returned rather than raised"""
        self.post_mock.return_value = FakeGuardianResponse(
            status_code=422, text="operation failed\n"
        )

        valid, reason = self.client.verify(self.capability, self.checksum)

        self.assertFalse(valid)
        self.assertEqual(reason, "operation failed")

    def test_client_06_long_reason_is_truncated(self):
        """An oversized answer is truncated before being used as reason"""
        self.post_mock.return_value = FakeGuardianResponse(
            status_code=422, text="x" * 10_000
        )

        _, reason = self.client.verify(self.capability, self.checksum)

        self.assertEqual(len(reason), 500)

    def test_client_07_unexpected_status(self):
        """Any status other than a success or a refusal raises FedbiomedGuardianError"""
        for status_code in (400, 403, 404, 500, 503):
            self.post_mock.return_value = FakeGuardianResponse(
                status_code=status_code, text="malformed"
            )
            with self.assertRaises(FedbiomedGuardianError):
                self.client.verify(self.capability, self.checksum)

    def test_client_08_transport_errors(self):
        """Network level failures raise FedbiomedGuardianError"""
        for exception in (
            requests.exceptions.ConnectionError("refused"),
            requests.exceptions.Timeout("timed out"),
            requests.exceptions.RequestException("boom"),
        ):
            self.post_mock.side_effect = exception
            with self.assertRaises(FedbiomedGuardianError):
                self.client.verify(self.capability, self.checksum)


if __name__ == "__main__":
    unittest.main()
