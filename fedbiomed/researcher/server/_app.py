# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application of the researcher user API"""

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from fedbiomed.common.constants import __version__
from fedbiomed.common.exceptions import (
    FedbiomedCommunicationError,
    FedbiomedError,
    FedbiomedResearcherServerError,
)
from fedbiomed.common.logger import logger

from .routers import datasets_router, health_router, nodes_router

if TYPE_CHECKING:
    from ._server import ResearcherServer


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Brings the gRPC endpoint up before the API accepts its first request.

    Args:
        app: Application being started.
    """
    server = app.state.server

    # Starting the transport layer blocks for seconds: it is run off the API event
    # loop, and before uvicorn binds its socket, so the API is never reachable while
    # the nodes still have no server to connect to.
    await run_in_threadpool(server.start_transport)

    yield


async def _handle_unavailable(_: Request, exc: Exception) -> JSONResponse:
    """Reports a transport layer that cannot currently serve the request.

    Args:
        _: Ignored.
        exc: Error raised while reaching the nodes.

    Returns:
        A `503` response carrying the error message.
    """
    logger.warning(f"Researcher API cannot reach the transport layer: {exc}")

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
    )


async def _handle_fedbiomed_error(_: Request, exc: Exception) -> JSONResponse:
    """Reports a Fed-BioMed error as a server error.

    Args:
        _: Ignored.
        exc: Error raised while serving the request.

    Returns:
        A `500` response carrying the error message.
    """
    logger.error(f"Researcher API failed to serve a request: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(exc)}
    )


def create_app(server: "ResearcherServer") -> FastAPI:
    """Builds the application serving the user API of a researcher server.

    Args:
        server: Researcher server the routes act on.

    Returns:
        The application, ready to be served.
    """
    app = FastAPI(
        title="Fed-BioMed researcher API",
        version=str(__version__),
        lifespan=_lifespan,
    )

    app.state.server = server

    # A transport layer that is not up yet, and a node that cannot be reached, are
    # both states the caller can retry out of, unlike the remaining Fed-BioMed errors.
    app.add_exception_handler(FedbiomedResearcherServerError, _handle_unavailable)
    app.add_exception_handler(FedbiomedCommunicationError, _handle_unavailable)
    app.add_exception_handler(FedbiomedError, _handle_fedbiomed_error)

    app.include_router(health_router)
    app.include_router(nodes_router)
    app.include_router(datasets_router)

    return app
