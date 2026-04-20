# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Persistence helper for the managed node process state."""

from __future__ import annotations

import getpass
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fedbiomed.common.constants import CONFIG_FOLDER_NAME
from fedbiomed.node.dataset_manager._db_tables import NodeProcessStateTable


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class NodeProcessStateManager:
    """Stores the current managed node process state in the node database."""

    def __init__(self, db_path: str) -> None:
        """Initialize the state manager."""
        self._table = NodeProcessStateTable(db_path)

    @classmethod
    def from_config(cls, config) -> "NodeProcessStateManager":
        """Create a state manager from a node config object."""
        db_path = os.path.abspath(
            os.path.join(
                config.root,
                CONFIG_FOLDER_NAME,
                config.get("default", "db"),
            )
        )
        return cls(db_path)

    @staticmethod
    def build_actor(actor: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a sanitized actor dictionary for process state attribution."""
        base = {
            "source": "local",
            "local_username": getpass.getuser(),
        }
        if not actor:
            return base

        allowed = {
            "source",
            "user_id",
            "email",
            "role",
            "name",
            "surname",
            "local_username",
        }
        base.update({key: value for key, value in actor.items() if key in allowed})
        return base

    def set_state(
        self,
        *,
        node_id: str,
        node_name: str,
        state: str,
        action: str,
        pid: Optional[int],
        actor: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> None:
        """Upsert the current node process state.

        Args:
            node_id: Node identifier from the node config.
            node_name: Human-readable node name from the node config.
            state: Current process state.
            action: Lifecycle action that produced the state.
            pid: Managed process PID if known.
            actor: Optional user/source metadata.
            reason: Optional reason for the state transition.
            exit_code: Optional process exit code.
        """
        now = _utc_now()
        existing = self._table.get_by_id(node_id) or {}

        entry = {
            "node_id": node_id,
            "node_name": node_name,
            "state": state,
            "pid": pid,
            "last_action": action,
            "last_reason": reason,
            "actor": self.build_actor(actor),
            "updated_at": now,
            "started_at": existing.get("started_at"),
            "stopped_at": existing.get("stopped_at"),
            "exit_code": exit_code,
        }

        if state == "running":
            entry["started_at"] = existing.get("started_at") or now
            entry["stopped_at"] = None
            entry["exit_code"] = None
        elif state == "stopped":
            entry["stopped_at"] = now
        elif state == "starting":
            entry["started_at"] = None
            entry["stopped_at"] = None
            entry["exit_code"] = None

        self._table.update_or_insert_by_id(node_id, entry)

    def get_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return the current process state for a node."""
        return self._table.get_by_id(node_id)
