# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
ORM models of the node database tables.
"""

import uuid
from typing import Any

from sqlalchemy import JSON, CheckConstraint, Enum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.db import Base
from fedbiomed.common.exceptions import FedbiomedError


def _dataset_type() -> Enum:
    """Column type holding a `DatasetTypes` value, checked by SQLite."""
    return Enum(
        *[t.value for t in DatasetTypes],
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
    )


def _check_dataset_type(entry: Base, key: str) -> None:
    """Refuses a field value that is not a `DatasetTypes` value."""
    value = getattr(entry, key)
    if value is not None and value not in [t.value for t in DatasetTypes]:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Invalid data input for "
            f"{type(entry).__name__}.{key}: '{value}' is not one of the values "
            "defined in 'fedbiomed.common.constants.DatasetTypes'"
        )


class Dataset(Base):
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
    shape: Mapped[list[int] | dict[str, list[int]]] = mapped_column(JSON)
    dtypes: Mapped[dict[str, str]] = mapped_column(JSON)
    dataset_parameters: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=None
    )
    # A data loading plan used by a dataset cannot be deleted
    dlp_id: Mapped[str | None] = mapped_column(ForeignKey("Dlps.dlp_id"), default=None)
    # Lets the session insert the plan before a dataset using it
    dlp: Mapped["Dlp | None"] = relationship(init=False, repr=False)

    def validate(self) -> None:
        super().validate()
        _check_dataset_type(self, "data_type")


class DynamicDataset(Base):
    """Dataset produced by a processing job from a parent dataset.

    The parent is a `Dataset` or another `DynamicDataset`.
    """

    __tablename__ = "DynamicDatasets"

    dataset_id: Mapped[str] = mapped_column(
        primary_key=True, default_factory=lambda: f"dynamic_dataset_{uuid.uuid4()}"
    )
    path: Mapped[str]
    researcher_id: Mapped[str]
    experiment_id: Mapped[str]
    processing_id: Mapped[str]
    parent_dataset_id: Mapped[str] = mapped_column(index=True)
    shape: Mapped[list[int] | dict[str, list[int]] | None] = mapped_column(
        JSON, default=None
    )
    dtypes: Mapped[dict[str, str] | None] = mapped_column(JSON, default=None)
    name: Mapped[str | None] = mapped_column(default=None)
    tags: Mapped[list[str] | None] = mapped_column(JSON, default=None)
    data_type: Mapped[str | None] = mapped_column(_dataset_type(), default=None)
    description: Mapped[str | None] = mapped_column(default=None)
    dataset_parameters: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=None
    )

    def validate(self) -> None:
        super().validate()
        _check_dataset_type(self, "data_type")


class Dlp(Base):
    """Data loading plan: the loading blocks applied to a type of dataset."""

    __tablename__ = "Dlps"
    __table_args__ = (CheckConstraint("length(dlp_name) >= 4"),)

    dlp_id: Mapped[str] = mapped_column(primary_key=True)
    dlp_name: Mapped[str] = mapped_column(unique=True)
    target_dataset_type: Mapped[str] = mapped_column(_dataset_type())
    loading_blocks: Mapped[dict[str, str]] = mapped_column(JSON)
    key_paths: Mapped[dict[str, list[str]]] = mapped_column(JSON)

    def validate(self) -> None:
        super().validate()
        if len(self.dlp_name) < 4:
            raise FedbiomedError(
                f"{ErrorNumbers.FB316.value}: Cannot save data loading plan, "
                "DLP name needs to have at least 4 characters."
            )
        _check_dataset_type(self, "target_dataset_type")


class Dlb(Base):
    """Data loading block, referenced by `Dlp.loading_blocks`."""

    __tablename__ = "Dlbs"

    dlb_id: Mapped[str] = mapped_column(primary_key=True)
    loading_block_class: Mapped[str]
    loading_block_module: Mapped[str]
    map: Mapped[dict[str, list[str]]] = mapped_column(JSON)
