# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Parent of the ORM models of the components' database tables.
"""

from enum import Enum
from typing import Self

from sqlalchemy import inspect
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


class DatabaseModel(MappedAsDataclass, DeclarativeBase, kw_only=True):
    """Parent of the ORM models: dataclasses refusing unknown or missing fields.

    A JSON value changed in place (`entry.tags.append(...)`) is neither saved
    nor checked: assign a new value. `__security_log__` sets what the model's
    table writes to the security log.
    """

    class SecurityLog(Enum):
        """What a model's table writes to the security log."""

        ALL = "all"  # reads and writes
        WRITES = "writes"
        NONE = "none"

    __security_log__ = SecurityLog.ALL

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Builds a row from an entry, refusing unknown or missing fields.

        A `None` value counts as not given: the field takes its default.
        """
        try:
            return cls(**{k: v for k, v in data.items() if v is not None})
        except TypeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Invalid data input for {cls.__name__}: {e}"
            ) from e

    def to_dict(self) -> dict:
        """Returns the row's columns as a dict, without the ones that are not set."""
        columns = (attr.key for attr in inspect(self).mapper.column_attrs)
        return {k: v for k in columns if (v := getattr(self, k)) is not None}
