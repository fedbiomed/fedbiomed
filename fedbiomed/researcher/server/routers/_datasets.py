# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Routes reporting the datasets deployed on the nodes.

The handlers are declared synchronous on purpose: the request machinery they call
blocks its thread while waiting for the node replies, and a synchronous route is
run by Starlette in a worker thread instead of on the API event loop.
"""

from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from .._auth import require_token
from .._deps import ServerDep
from .._schemas import DatasetSearchQuery

router = APIRouter(
    prefix="/api/v1/datasets", tags=["datasets"], dependencies=[Depends(require_token)]
)


@router.get("", response_model=Dict[str, List[Dict[str, Any]]])
def list_datasets(
    server: ServerDep,
    nodes: Annotated[
        Optional[List[str]],
        Query(
            description="Only question these nodes. Questions every known node when "
            "omitted."
        ),
    ] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Reports every dataset the nodes hold.

    Every node that answered is reported, including those holding no dataset at
    all, so the caller can tell an empty node from a silent one.

    Args:
        server: Researcher server being served.
        nodes: Nodes to question, or None for all of them.

    Returns:
        Datasets of each node that answered, indexed by node ID.
    """
    return server.get_requests().list(nodes=nodes)


@router.post("/search", response_model=Dict[str, List[Dict[str, Any]]])
def search_datasets(
    server: ServerDep, query: DatasetSearchQuery
) -> Dict[str, List[Dict[str, Any]]]:
    """Looks for the datasets carrying a set of tags.

    A dataset matches when it carries every requested tag, so a tag-less query
    matches all of them. Unlike `list_datasets`, only the nodes holding at least
    one matching dataset are reported.

    Args:
        server: Researcher server being served.
        query: Tags to look for, and the nodes to look on.

    Returns:
        Matching datasets of each node that found some, indexed by node ID.
    """
    return server.get_requests().search(tags=query.tags, nodes=query.nodes)
