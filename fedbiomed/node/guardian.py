# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Client for the node-side capability guardian service.

A capability is issued by a third party for a given training plan. It is an
opaque base64 string: the node never parses it, it only forwards it. The node
computes the digest of the training plan it actually received and asks a
locally reachable guardian service whether the capability covers that digest.
Training and validation only proceed on a positive answer.
"""

import hashlib
from typing import Tuple

import requests

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedGuardianError

DEFAULT_GUARDIAN_TIMEOUT = 10

PROCESS_CAPABILITY_ENDPOINT = "/process_capability"

# Status code the guardian service answers with when it refuses the operation
REJECTED_STATUS_CODE = 422

# Maximum length of a guardian answer that is kept as the reason
MAX_REASON_LENGTH = 500


def compute_training_plan_checksum(source: str) -> str:
    """Computes the digest of a training plan source code.

    The digest is a plain SHA-256 of the exact source string, prefixed with the
    algorithm name (`sha256:<hex>`), so that a third party can reproduce it
    without having Fed-BioMed installed. This is deliberately different from
    the hashing done by
    [`TrainingPlanSecurityManager`]
    [fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager],
    which minifies the source and uses a node-configurable algorithm.

    Args:
        source: Source code of the training plan.

    Returns:
        Digest of the UTF-8 encoded source, formatted as `sha256:<hex>`.

    Raises:
        FedbiomedGuardianError: if the given source is not a string.
    """
    if not isinstance(source, str):
        raise FedbiomedGuardianError(
            f"{ErrorNumbers.FB328.value}: training plan source should be a string, "
            f"but got {type(source)}"
        )

    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


class GuardianClient:
    """Client that asks a guardian service to process a capability.

    The guardian service is expected to expose a `POST /process_capability`
    endpoint that accepts a JSON body of the form
    `{"capability_b64": <str>, "request_context": {"calculated_script_digest": <str>}}`.
    It answers with a success status when the capability authorizes the
    operation, and with status 422 when it refuses it.
    """

    def __init__(
        self,
        service_url: str,
        timeout: int = DEFAULT_GUARDIAN_TIMEOUT,
    ) -> None:
        """Constructor of the class.

        Args:
            service_url: Base URL of the guardian service, e.g.
                `http://localhost:7900`. A trailing slash is accepted.
            timeout: Timeout in seconds for the guardian request.

        Raises:
            FedbiomedGuardianError: if the given service URL is empty or not a string.
        """
        if not isinstance(service_url, str) or not service_url.strip():
            raise FedbiomedGuardianError(
                f"{ErrorNumbers.FB328.value}: guardian service URL should be a "
                f"non-empty string, but got '{service_url}'"
            )

        self._service_url = service_url.strip().rstrip("/")
        self._timeout = timeout

    def process_capability_url(self) -> str:
        """Gets the full URL of the capability processing endpoint.

        Returns:
            URL the capability is sent to.
        """
        return f"{self._service_url}{PROCESS_CAPABILITY_ENDPOINT}"

    def verify(self, capability: str, checksum: str) -> Tuple[bool, str]:
        """Asks the guardian service whether a capability authorizes a round.

        Args:
            capability: Opaque base64 capability issued by the third party. It is
                forwarded unchanged; the node does not interpret its content.
            checksum: Digest of the training plan as computed by the node.

        Returns:
            A tuple of
              * valid: True if the guardian service authorizes the round
              * reason: Answer returned by the guardian service

        Raises:
            FedbiomedGuardianError: if the guardian service is unreachable, or
                answers with a status that is neither a success nor a refusal.
        """
        url = self.process_capability_url()
        payload = {
            "capability_b64": capability,
            "request_context": {"calculated_script_digest": checksum},
        }

        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
        except requests.exceptions.RequestException as e:
            raise FedbiomedGuardianError(
                f"{ErrorNumbers.FB328.value}: cannot reach the guardian service at "
                f"{url}: {e}"
            ) from e

        reason = response.text.strip()[:MAX_REASON_LENGTH]

        if 200 <= response.status_code < 300:
            return True, reason

        if response.status_code == REJECTED_STATUS_CODE:
            return False, reason

        raise FedbiomedGuardianError(
            f"{ErrorNumbers.FB328.value}: guardian service at {url} answered with "
            f"status code {response.status_code}: {reason}"
        )
