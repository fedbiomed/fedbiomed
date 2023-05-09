# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for handling version compatibility.

This module contains functions that help managing the versions of different Fed-BioMed components.
The components currently supported are:
- researcher config file

See https://fedbiomed.org/latest/user-guide/deployment/versions for more information
"""

from packaging.version import Version
from typing import Optional, Union
from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedVersionError


FBM_Component_Version = Version
__default_version__ = Version('1.0')  # default version to assign to any component before versioning was introduced


def _create_error_msg(error_msg: str, their_version: Version, our_version: Version) -> str:
    """Utility function to put together a nice error message when versions don't exactly match.

    Args:
        error_msg: customizable part of the message. It may contain two %s placeholders which will be substituted with
            the values of their_version and our_version.
        their_version: the version that we detected in the component
        our_version: the version of the component within the current runtime

    Returns:
        A formatted message with a link to the docs appended at the end.
    """
    try:
        msg = error_msg % (str(their_version), str(our_version))
    except TypeError:
        msg = error_msg
    msg += " -> See https://fedbiomed.org/latest/user-guide/deployment/versions for more information"
    return msg

def raise_for_version_compatibility(their_version: Union[Version, str],
                                    our_version: Union[Version, str],
                                    error_msg: Optional[str] = None) -> None:
    """Check version compatibility and behave accordingly.

    Raises an exception if the versions are incompatible, otherwise outputs a warning or info message.

    Args:
        their_version: the version that we detected in the component
        our_version: the version of the component within the current runtime
        error_msg: an optional error message. It may contain two %s placeholders which will be substituted with
            the values of their_version and our_version.

    Raises:
        FedbiomedVersionError: if the versions are incompatible
    """
    if isinstance(our_version, str):
        our_version = Version(our_version)
    if isinstance(their_version, str):
        their_version = Version(their_version)
    msg = _create_error_msg(
        "Found version %s, expected version %s" if error_msg is None else error_msg,
        their_version,
        our_version
    )
    if our_version.major != their_version.major:
        logger.critical(msg)
        raise FedbiomedVersionError(msg)
    elif our_version.minor != their_version.minor:
        logger.warning(msg)
    elif our_version.micro != their_version.micro:
        logger.info(msg)



