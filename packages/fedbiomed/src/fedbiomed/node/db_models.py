# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
ORM models of the node database tables.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, Enum, ForeignKey
from sqlalchemy.orm import (
    Mapped,
    MappedAsDataclass,
    mapped_column,
    relationship,
    validates,
)

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.db import DatabaseModel, UtcDateTime
from fedbiomed.common.exceptions import FedbiomedError


def _dataset_type() -> Enum:
    """Column type holding a `DatasetTypes` value, checked by SQLite."""
    return Enum(
        *[t.value for t in DatasetTypes], native_enum=False, create_constraint=True
    )


class Dataset(DatabaseModel):
    """Dataset registered on the node."""

    __tablename__ = "Datasets"

    dataset_id: Mapped[str] = mapped_column(
        primary_key=True, default_factory=lambda: f"dataset_{uuid.uuid4()}"
    )
    name: Mapped[str]
    data_type: Mapped[str] = mapped_column(_dataset_type())
    tags: Mapped[list[str]] = mapped_column(JSON)
    description: Mapped[str]
    path: Mapped[str]
    # Not set for custom datasets, which define their own shape and types
    shape: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    dtypes: Mapped[dict[str, str] | None] = mapped_column(JSON, default=None)
    dataset_parameters: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=None
    )
    # A data loading plan used by a dataset cannot be deleted
    dlp_id: Mapped[str | None] = mapped_column(ForeignKey("Dlps.dlp_id"), default=None)
    # Let the session insert the plan before a dataset using it
    dlp: Mapped["Dlp | None"] = relationship(init=False, repr=False)

    @validates("tags")
    def _validate_tags(self, _key: str, tags: list[str]) -> list[str]:
        # Searches and tag conflicts compare tags as sets of strings
        if not (isinstance(tags, list) and all(isinstance(tag, str) for tag in tags)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Invalid data input for Dataset.tags: "
                f"expected a list of strings, got {tags!r}"
            )
        return tags


class DynamicDataset(DatabaseModel):
    """Dataset produced by a processing job from a parent dataset.

    The parent is a `Dataset` (`parent_dataset_id`) or another
    `DynamicDataset` (`parent_dynamic_id`): exactly one is set. A dataset with
    derived datasets cannot be deleted before them.
    """

    __tablename__ = "DynamicDatasets"
    __table_args__ = (
        CheckConstraint("(parent_dataset_id IS NULL) <> (parent_dynamic_id IS NULL)"),
    )

    dataset_id: Mapped[str] = mapped_column(
        primary_key=True, default_factory=lambda: f"dynamic_dataset_{uuid.uuid4()}"
    )
    path: Mapped[str]
    researcher_id: Mapped[str]
    experiment_id: Mapped[str]
    processing_id: Mapped[str]
    parent_dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("Datasets.dataset_id"), index=True, default=None
    )
    parent_dynamic_id: Mapped[str | None] = mapped_column(
        ForeignKey("DynamicDatasets.dataset_id"), index=True, default=None
    )
    # Let the session insert the parent dataset before the datasets derived
    # from it
    parent_dataset: Mapped["Dataset | None"] = relationship(init=False, repr=False)
    shape: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    dtypes: Mapped[dict[str, str] | None] = mapped_column(JSON, default=None)
    name: Mapped[str | None] = mapped_column(default=None)
    tags: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    data_type: Mapped[str | None] = mapped_column(_dataset_type(), default=None)
    description: Mapped[str | None] = mapped_column(default=None)
    dataset_parameters: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=None
    )


class Dlp(DatabaseModel):
    """Data loading plan: the loading blocks applied to a type of dataset."""

    __tablename__ = "Dlps"

    dlp_id: Mapped[str] = mapped_column(primary_key=True)
    dlp_name: Mapped[str] = mapped_column(unique=True)
    target_dataset_type: Mapped[str] = mapped_column(_dataset_type())
    loading_blocks: Mapped[dict[str, str]] = mapped_column(JSON)
    key_paths: Mapped[dict[str, list[str]]] = mapped_column(JSON)

    @validates("dlp_name")
    def _validate_dlp_name(self, _key: str, dlp_name: str) -> str:
        # Typed by GUI users, so refused with a readable message
        if len(dlp_name) < 4:
            raise FedbiomedError(
                f"{ErrorNumbers.FB316.value}: Cannot save data loading plan, "
                "DLP name needs to have at least 4 characters."
            )
        return dlp_name


class Dlb(DatabaseModel):
    """Data loading block, referenced by `Dlp.loading_blocks`.

    Each block type serializes its own fields, so the block is stored as its
    serialized form (`DataLoadingBlock.serialize`).
    """

    __tablename__ = "Dlbs"

    dlb_id: Mapped[str] = mapped_column(primary_key=True)
    content: Mapped[dict[str, Any]] = mapped_column(JSON)


class TrainingPlan(DatabaseModel):
    """Training plan known to the node, with its approval status."""

    __tablename__ = "TrainingPlans"

    training_plan_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    hash: Mapped[str] = mapped_column(unique=True)
    algorithm: Mapped[str]  # hashing algorithm
    training_plan: Mapped[str]  # source code
    training_plan_type: Mapped[str]  # a `TrainingPlanStatus` value
    training_plan_status: Mapped[str]  # a `TrainingPlanApprovalStatus` value
    date_created: Mapped[datetime] = mapped_column(UtcDateTime)
    date_modified: Mapped[datetime] = mapped_column(UtcDateTime)
    date_registered: Mapped[datetime] = mapped_column(UtcDateTime)
    date_last_action: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    description: Mapped[str | None] = mapped_column(default=None)
    researcher_id: Mapped[str | None] = mapped_column(default=None)
    notes: Mapped[str | None] = mapped_column(default=None)


class NodeState(DatabaseModel):
    """State the node keeps between the rounds of an experiment."""

    __tablename__ = "NodeStates"

    state_id: Mapped[str] = mapped_column(primary_key=True)
    experiment_id: Mapped[str] = mapped_column(index=True)
    version_node_id: Mapped[str]
    optimizer_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    testing_dataset: Mapped[Any] = mapped_column(JSON, nullable=True, default=None)
    persistent_model_weights: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=None
    )


class _NodeProcessFields(MappedAsDataclass, kw_only=True):
    """Fields of a node process state, current or past."""

    node_name: Mapped[str]
    state: Mapped[str]
    action: Mapped[str]
    pid: Mapped[int]
    reason: Mapped[str | None] = mapped_column(default=None)
    actor: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    updated_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    stopped_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    exit_code: Mapped[int | None] = mapped_column(default=None)
    node_args: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    background: Mapped[bool | None] = mapped_column(default=None)


class NodeProcessState(_NodeProcessFields, DatabaseModel):
    """Current state of the managed node process."""

    __tablename__ = "NodeProcessStates"

    node_id: Mapped[str] = mapped_column(primary_key=True)


class NodeProcessStateHistory(_NodeProcessFields, DatabaseModel):
    """Past states of the managed node process, several per node."""

    __tablename__ = "NodeProcessStateHistory"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    node_id: Mapped[str] = mapped_column(index=True)


class _NodeConnectionFields(MappedAsDataclass, kw_only=True):
    """Fields of the node's channel to the researcher, current or past."""

    state: Mapped[str]
    host: Mapped[str]
    port: Mapped[str]
    researcher_id: Mapped[str | None] = mapped_column(default=None)
    mtls: Mapped[bool | None] = mapped_column(default=None)
    identity_verified: Mapped[bool | None] = mapped_column(default=None)
    operation: Mapped[str | None] = mapped_column(default=None)
    reason: Mapped[str | None] = mapped_column(default=None)
    # The peer certificate's audit fields, never key material
    certificate: Mapped[dict[str, str] | None] = mapped_column(JSON, default=None)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    updated_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    last_error: Mapped[str | None] = mapped_column(default=None)
    last_error_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)


class NodeConnectionState(_NodeConnectionFields, DatabaseModel):
    """State of the node's channel to the researcher, as the node last observed it."""

    __tablename__ = "NodeConnectionStates"

    node_id: Mapped[str] = mapped_column(primary_key=True)


class NodeConnectionStateHistory(_NodeConnectionFields, DatabaseModel):
    """Past states of the node's channel to the researcher, several per node."""

    __tablename__ = "NodeConnectionStateHistory"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    node_id: Mapped[str] = mapped_column(index=True)
