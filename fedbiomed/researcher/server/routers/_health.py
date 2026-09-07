# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Liveness route of the researcher user API"""

from fastapi import APIRouter

from fedbiomed.common.constants import __version__

from .._deps import ServerDep
from .._schemas import HealthStatus, TransportStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
def health(server: ServerDep) -> HealthStatus:
    """Reports whether the server runs and whether the nodes can reach it.

    Left unauthenticated so that a supervisor can probe the server without holding
    the API token. It exposes no federation data.

    Args:
        server: Researcher server being served.

    Returns:
        Liveness of the server and state of its gRPC endpoint.
    """
    config = server.get_config()

    return HealthStatus(
        status="ok",
        researcher_id=config.get("default", "id"),
        version=str(__version__),
        transport=TransportStatus(
            host=config.get("server", "host"),
            port=int(config.get("server", "port")),
            ready=server.is_transport_ready(),
        ),
    )
