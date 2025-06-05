# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for handling version compatibility.

This module contains functions that help managing the versions of different Fed-BioMed components.

See https://fedbiomed.org/latest/user-guide/deployment/versions or
./doc/user-guide/deployment/versions.md for more information
"""

from packaging.version import Version
from typing import Optional, Union
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedVersionError


FBM_Component_Version = Version  # for Typing
"""The Type of objects representing version numbers in Fed-BioMed"""

__default_version__ = Version('0')  # default version to assign to any component before versioning was introduced


def _create_msg_for_version_check(error_msg: str,
                                  their_version: FBM_Component_Version,
                                  our_version: FBM_Component_Version) -> str:
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


def raise_for_version_compatibility(their_version: Union[FBM_Component_Version, str],
                                    our_version: Union[FBM_Component_Version, str],
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
        our_version = FBM_Component_Version(our_version)
    if isinstance(their_version, str):
        their_version = FBM_Component_Version(their_version)
    if not isinstance(our_version, FBM_Component_Version):
        msg = f"{ErrorNumbers.FB625.value}: Component version has incorrect type `our_version` type={str(type(our_version))} value={our_version}"
        logger.critical(msg)
        raise FedbiomedVersionError(msg)
    if not isinstance(their_version, FBM_Component_Version):
        msg = f"{ErrorNumbers.FB625.value}: Component version has incorrect type `their_version` type={str(type(their_version))} value={their_version}"
        logger.critical(msg)
        raise FedbiomedVersionError(msg)
    if our_version != their_version:
        # note: the checks below rely on the short-circuiting behaviour of the or operator
        # (e.g. when checking our_version.minor < their_version.minor we have the guarantee that
        # our_version.major == their_version.major
        if our_version.major != their_version.major or \
                our_version.minor < their_version.minor or \
                (our_version.minor == their_version.minor and our_version.micro < their_version.micro):
            msg = _create_msg_for_version_check(
                f"{ErrorNumbers.FB625.value}: Found incompatible version %s, expected version %s" if error_msg is None else error_msg,
                their_version,
                our_version
            )
            logger.critical(msg)
            raise FedbiomedVersionError(msg)
        else:
            msg = _create_msg_for_version_check(
                "Found version %s, expected version %s",
                their_version,
                our_version
            )
            logger.warning(msg)
