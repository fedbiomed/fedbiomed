# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Client for the node-side capability guardian service.

A capability is issued and signed by a third party for a given training plan.
The node computes the checksum of the training plan it actually received and
asks a locally reachable guardian service whether the capability covers that
checksum. Training and validation only proceed on a positive answer.
"""

import hashlib
from typing import Any, Dict, Optional, Tuple

import requests

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedGuardianError

DEFAULT_GUARDIAN_TIMEOUT = 10

VERIFY_ENDPOINT = "/verify"


def compute_training_plan_checksum(source: str) -> str:
    """Computes the checksum of a training plan source code.

    The checksum is a plain SHA-256 of the exact source string, so that a third
    party can reproduce it without having Fed-BioMed installed. This is
    deliberately different from the hashing done by
    [`TrainingPlanSecurityManager`]
    [fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager],
    which minifies the source and uses a node-configurable algorithm.

    Args:
        source: Source code of the training plan.

    Returns:
        Hexadecimal SHA-256 digest of the UTF-8 encoded source.

    Raises:
        FedbiomedGuardianError: if the given source is not a string.
    """
    if not isinstance(source, str):
        raise FedbiomedGuardianError(
            f"{ErrorNumbers.FB328.value}: training plan source should be a string, "
            f"but got {type(source)}"
        )

    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class GuardianClient:
    """Client that asks a guardian service to validate a capability.

    The guardian service is expected to expose a `POST /verify` endpoint that
    accepts a JSON body and answers with `{"valid": <bool>, "reason": <str>}`.
    """

    def __init__(
        self,
        service_url: str,
        timeout: int = DEFAULT_GUARDIAN_TIMEOUT,
    ) -> None:
        """Constructor of the class.

        Args:
            service_url: Base URL of the guardian service, e.g.
                `http://localhost:8000`. A trailing slash is accepted.
            timeout: Timeout in seconds for the verification request.

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

    def verify_url(self) -> str:
        """Gets the full URL of the verification endpoint.

        Returns:
            URL the verification request is sent to.
        """
        return f"{self._service_url}{VERIFY_ENDPOINT}"

    def verify(
        self,
        capabilities: Dict[str, Any],
        checksum: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Asks the guardian service whether a capability authorizes a round.

        Args:
            capabilities: Encoded capability issued by the third party. It is
                forwarded unchanged; the node does not interpret its content.
            checksum: Checksum of the training plan as computed by the node.
            context: Extra round information forwarded to the guardian service,
                such as node, researcher and experiment identifiers.

        Returns:
            A tuple of
              * valid: True if the guardian service authorizes the round
              * reason: Explanation returned by the guardian service

        Raises:
            FedbiomedGuardianError: if the guardian service is unreachable,
                answers with a non-success status, or returns a payload that
                does not follow the expected format.
        """
        payload = {
            **(context or {}),
            "capabilities": capabilities,
            "training_plan_checksum": checksum,
        }

        try:
            response = requests.post(
                self.verify_url(), json=payload, timeout=self._timeout
            )
        except requests.exceptions.RequestException as e:
            raise FedbiomedGuardianError(
                f"{ErrorNumbers.FB328.value}: cannot reach the guardian service at "
                f"{self.verify_url()}: {e}"
            ) from e

        if response.status_code != 200:
            raise FedbiomedGuardianError(
                f"{ErrorNumbers.FB328.value}: guardian service at {self.verify_url()} "
                f"answered with status code {response.status_code}"
            )

        try:
            result = response.json()
        except Exception as e:
            raise FedbiomedGuardianError(
                f"{ErrorNumbers.FB328.value}: guardian service at {self.verify_url()} "
                f"did not answer with a valid JSON payload: {e}"
            ) from e

        if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
            raise FedbiomedGuardianError(
                f"{ErrorNumbers.FB328.value}: guardian service at {self.verify_url()} "
                "should answer with a boolean 'valid' entry"
            )

        return result["valid"], str(result.get("reason", ""))
