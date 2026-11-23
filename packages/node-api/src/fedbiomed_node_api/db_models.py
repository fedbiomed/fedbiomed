# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
ORM models of the GUI database tables.
"""

from datetime import datetime

from fedbiomed.common.db import DatabaseModel, UtcDateTime
from sqlalchemy.orm import Mapped, MappedAsDataclass, mapped_column


class _UserFields(MappedAsDataclass, kw_only=True):
    """Fields of a GUI user, registered or requesting an account."""

    __sensitive_fields__ = ("password_hash",)

    user_email: Mapped[str]
    password_hash: Mapped[str]
    user_name: Mapped[str]
    user_surname: Mapped[str]
    user_role: Mapped[int]  # a `UserRoleType` value
    creation_date: Mapped[datetime] = mapped_column(UtcDateTime)


class User(_UserFields, DatabaseModel):
    """Registered GUI user."""

    __tablename__ = "Users"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    user_email: Mapped[str] = mapped_column(unique=True)


class UserRequest(_UserFields, DatabaseModel):
    """Account request, waiting for an administrator's decision."""

    __tablename__ = "UserRequests"

    request_id: Mapped[str] = mapped_column(primary_key=True)
    request_status: Mapped[str]  # a `UserRequestStatus` value
