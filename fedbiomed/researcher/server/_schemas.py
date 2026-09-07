# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Payloads exchanged by the researcher user API"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TransportStatus(BaseModel):
    """State of the gRPC endpoint the nodes connect to.

    Attributes:
        host: Address the gRPC server is bound to.
        port: Port the gRPC server is bound to.
        ready: Whether the gRPC server accepts connections from the nodes.
    """

    host: str
    port: int
    ready: bool


class HealthStatus(BaseModel):
    """Liveness of the researcher server.

    Attributes:
        status: `ok` once the API answers, whatever the state of the transport.
        researcher_id: Unique ID of the researcher component being served.
        version: Fed-BioMed version of the running server.
        transport: State of the gRPC endpoint.
    """

    status: str
    researcher_id: str
    version: str
    transport: TransportStatus


class NodeInfo(BaseModel):
    """A node known to the transport layer.

    Attributes:
        node_id: Unique ID of the node.
        status: Activity of the node, one of `ACTIVE`, `WAITING` or `DISCONNECTED`.
    """

    node_id: str
    status: str


class DatasetSearchQuery(BaseModel):
    """What to look for when searching the datasets of the federation.

    Sent as a body rather than as query parameters because Fed-BioMed tags
    conventionally start with `#`, which a query string would have to escape.

    Attributes:
        tags: Tags a dataset has to carry all of. An empty list matches every
            dataset, which is what the nodes answer to a tag-less search.
        nodes: Nodes to question. Questions every known node when omitted.
    """

    tags: List[str] = Field(default_factory=list)
    nodes: Optional[List[str]] = None
