# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Dependencies shared by the routes of the researcher user API"""

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, Request

if TYPE_CHECKING:
    from ._server import ResearcherServer


def get_server(request: Request) -> "ResearcherServer":
    """Returns the researcher server the application is serving for.

    Args:
        request: Incoming request.

    Returns:
        The server instance held by the application.
    """
    return request.app.state.server


# The concrete type is not usable here: importing it would close an import cycle
# through the application module. The dependency is what fills the parameter.
ServerDep = Annotated[Any, Depends(get_server)]
