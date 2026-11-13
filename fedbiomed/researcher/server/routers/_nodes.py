# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Routes reporting the nodes of the federation.

The handlers are declared synchronous on purpose: the request machinery they call
blocks its thread while waiting for the node replies, and a synchronous route is
run by Starlette in a worker thread instead of on the API event loop.
"""

from typing import List

from fastapi import APIRouter, Depends

from .._auth import require_token
from .._deps import ServerDep
from .._schemas import NodeInfo

router = APIRouter(
    prefix="/api/v1", tags=["nodes"], dependencies=[Depends(require_token)]
)


@router.get("/nodes", response_model=List[NodeInfo])
def list_nodes(server: ServerDep) -> List[NodeInfo]:
    """Lists the nodes the server knows about.

    Sends nothing to the nodes: it reports the connection state the gRPC server
    already holds, so nodes that have since disconnected are listed too.

    Args:
        server: Researcher server being served.

    Returns:
        One entry per known node.
    """
    return [NodeInfo(**node) for node in server.get_requests().get_nodes()]


@router.post("/nodes/ping", response_model=List[str])
def ping_nodes(server: ServerDep) -> List[str]:
    """Pings every known node and reports which ones answered.

    Args:
        server: Researcher server being served.

    Returns:
        IDs of the nodes that answered within the request timeout.
    """
    return server.get_requests().ping_nodes()
