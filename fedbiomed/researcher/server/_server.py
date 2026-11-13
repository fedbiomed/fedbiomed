# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Standalone researcher server.

Holds together, in a single process, the gRPC endpoint the nodes connect to and
the HTTP API the user drives the researcher through. The two run on distinct
event loops on purpose: the request machinery of
[`FederatedRequest`][fedbiomed.researcher.requests.FederatedRequest] blocks its
calling thread and reaches into the gRPC event loop from there, so serving the
user API from that same loop would deadlock the first time a request waits for a
reply.
"""

from typing import Optional

import uvicorn

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedResearcherServerError
from fedbiomed.common.logger import logger
from fedbiomed.researcher.config import ResearcherConfig
from fedbiomed.researcher.requests import Requests

from ._app import create_app
from ._auth import ensure_token, token_path


class ResearcherServer:
    """Researcher daemon serving both the nodes and the user"""

    def __init__(
        self,
        config: ResearcherConfig,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """Constructor of the class.

        Args:
            config: Configuration of the researcher component to serve.
            host: Address to bind the user API to. Defaults to the `host` value of
                the `api` configuration section.
            port: Port to bind the user API to. Defaults to the `port` value of
                the `api` configuration section.
        """
        self._config = config
        self._host = host if host else config.get("api", "host")
        self._port = int(port) if port else int(config.get("api", "port"))
        self._api_token = ensure_token(config)
        self._requests: Optional[Requests] = None

    def get_config(self) -> ResearcherConfig:
        """Returns the configuration of the component being served"""
        return self._config

    def get_host(self) -> str:
        """Returns the address the user API is bound to"""
        return self._host

    def get_port(self) -> int:
        """Returns the port the user API is bound to"""
        return self._port

    def get_api_token(self) -> str:
        """Returns the token clients have to present to reach the user API"""
        return self._api_token

    def get_requests(self) -> Requests:
        """Returns the request handler talking to the nodes.

        Returns:
            The handler owning the gRPC server.

        Raises:
            FedbiomedResearcherServerError: the transport layer is not started yet.
        """
        if self._requests is None:
            raise FedbiomedResearcherServerError(
                f"{ErrorNumbers.FB634.value}: the transport layer is not started, "
                "so no request can be sent to the nodes."
            )

        return self._requests

    def is_transport_ready(self) -> bool:
        """Whether the nodes can reach this server.

        Unlike `get_requests` this never raises, so it can report the state of a
        server whose transport layer failed to come up.

        Returns:
            True once the gRPC server accepts connections from the nodes.
        """
        return self._requests is not None and self._requests.is_ready()

    def start_transport(self) -> None:
        """Starts the gRPC endpoint the nodes connect to.

        Blocks for as long as the gRPC server takes to bind and to wait for a first
        node, so it must not run on the event loop of the user API. Starting an
        already started transport layer does nothing.
        """
        if self._requests is not None:
            return

        self._requests = Requests(config=self._config)

    def run(self) -> None:
        """Serves the user API until the process is interrupted.

        The gRPC endpoint is brought up first, by the startup of the application,
        so the API never answers while the nodes have nothing to connect to.
        """
        logger.info(
            f"Researcher user API listening on http://{self._host}:{self._port} "
            f"- token in {token_path(self._config)}"
        )

        server = uvicorn.Server(
            uvicorn.Config(
                create_app(self),
                host=self._host,
                port=self._port,
                # Fed-BioMed configures its own handlers; letting uvicorn install
                # its logging configuration would silence them.
                log_config=None,
            )
        )
        server.run()
