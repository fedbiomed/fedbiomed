# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
ORM models of the database tables shared by the components (node and researcher).
"""

from typing import Any

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from fedbiomed.common.db import DatabaseModel


class CertificateEntry(DatabaseModel):
    """Certificate of a component this one trusts."""

    __tablename__ = "Certificates"
    __sensitive_fields__ = ("certificate",)

    component_id: Mapped[str] = mapped_column(primary_key=True)
    certificate: Mapped[str]  # PEM


class SecaggElementEntry(DatabaseModel):
    """Secure aggregation context element of an experiment."""

    __tablename__ = "SecaggElements"
    __sensitive_fields__ = ("context",)

    secagg_id: Mapped[str] = mapped_column(primary_key=True)
    secagg_version: Mapped[str]
    secagg_elem: Mapped[int]  # a `SecaggElementTypes` value
    experiment_id: Mapped[str] = mapped_column(index=True)
    parties: Mapped[list[str]] = mapped_column(JSON)
    # Server key part, or Diffie-Hellman keys per node
    context: Mapped[dict[str, Any]] = mapped_column(JSON)
