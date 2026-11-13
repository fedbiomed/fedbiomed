# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Authentication of the researcher user API.

The API is protected by a single token stored in the component directory. It is
created on the first start of the server and read back on every later start, so
the token a user copied stays valid across restarts.
"""

import os
import secrets
import stat
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from fedbiomed.common.config import Config
from fedbiomed.common.constants import CONFIG_FOLDER_NAME, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedResearcherServerError

API_TOKEN_FILE_NAME = "api_token"

# Number of random bytes of the generated token, before URL-safe encoding.
API_TOKEN_BYTES = 32

_bearer = HTTPBearer(auto_error=False)


def token_path(config: Config) -> str:
    """Location of the API token of a component.

    Args:
        config: Configuration of the researcher component.

    Returns:
        Absolute path of the token file.
    """
    return os.path.join(config.root, CONFIG_FOLDER_NAME, API_TOKEN_FILE_NAME)


def ensure_token(config: Config) -> str:
    """Reads the API token of a component, creating one if there is none.

    Args:
        config: Configuration of the researcher component.

    Returns:
        The token clients have to present to reach the user API.

    Raises:
        FedbiomedResearcherServerError: the token file can neither be read nor created.
    """
    path = token_path(config)

    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="UTF-8") as file_:
                token = file_.read().strip()
            if token:
                return token

        token = secrets.token_urlsafe(API_TOKEN_BYTES)

        # Created with its final mode rather than chmod-ed afterwards: the latter
        # leaves a window where the token is readable by every user of the machine.
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(descriptor, "w", encoding="UTF-8") as file_:
            file_.write(token)

        return token
    except OSError as e:
        raise FedbiomedResearcherServerError(
            f"{ErrorNumbers.FB634.value}: cannot read or create the API token file "
            f"{path}: {e}"
        ) from e


def require_token(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)],
) -> None:
    """Refuses the request unless it carries the API token of this component.

    Args:
        request: Incoming request, used to reach the server the app serves for.
        credentials: Bearer credentials of the request, None when it carries none.

    Raises:
        HTTPException: no token was presented, or it is not the expected one.
    """
    expected = request.app.state.server.get_api_token()

    if credentials is None or not secrets.compare_digest(
        credentials.credentials, expected
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
