# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataclasses for database entities
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


@dataclass
class TableEntry:
    """Base class for database entries with common conversion methods"""

    def to_dict(self) -> dict:
        """Convert entry to dictionary - removes None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict):
        """Generic from_dict implementation for dataclasses"""
        try:
            return cls(**data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Invalid data input for {cls.__name__}: "
                f"{str(e)}"
            ) from e


@dataclass
class NodeProcessStateEntry(TableEntry):
    """Current node process state entry."""

    node_id: str
    node_name: str
    state: str
    action: str
    pid: int
    reason: Optional[str] = None
    actor: Optional[Dict[str, Any]] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    exit_code: Optional[int] = None
    node_args: Optional[Dict[str, Any]] = None
    background: Optional[bool] = None


@dataclass
class NodeProcessStateHistoryEntry(NodeProcessStateEntry):
    """Historical node process state entry."""


@dataclass
class NodeConnectionStateEntry(TableEntry):
    """State of the node's channel to the researcher, as the node last observed it.

    `certificate` carries the peer certificate's audit fields, never key material.
    """

    node_id: str
    state: str
    host: str
    port: str
    researcher_id: Optional[str] = None
    mtls: Optional[bool] = None
    identity_verified: Optional[bool] = None
    operation: Optional[str] = None
    reason: Optional[str] = None
    certificate: Optional[Dict[str, str]] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_error: Optional[str] = None
    last_error_at: Optional[str] = None


@dataclass
class NodeConnectionStateHistoryEntry(NodeConnectionStateEntry):
    """Historical node connection state entry."""
