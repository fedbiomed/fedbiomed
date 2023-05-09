from packaging.version import Version
from typing import Optional, Union
from fedbiomed.common.logger import logger
from fedbiomed.common.exceptions import FedbiomedVersionError


__default_version__ = Version('1.0')


def _create_error_msg(error_msg: str, their_version: Version, our_version: Version):
    try:
        msg = error_msg % (str(their_version), str(our_version))
    except TypeError:
        msg = error_msg
    return msg

def raise_for_version_compatibility(their_version: Union[Version, str],
                                    our_version: Union[Version, str],
                                    error_msg: Optional[str] = None):
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



