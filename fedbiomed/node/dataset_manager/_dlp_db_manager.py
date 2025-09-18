# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Manage the node's database table for handling DLP / DLB
"""

from typing import Optional, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import (
    DataLoadingPlan,
)
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._db import DlbDB, DlpDB


class DlpDatabaseManager:
    """Manage the node's database table for handling DLP / DLB

    Facility for storing data, retrieving data and getting data info
    for the node. Currently uses TinyDB.
    """

    _dlp_table: DlpDB
    _dlb_table: DlbDB

    def __init__(self, path: str):
        self._dlp_table = DlpDB(path)
        self._dlb_table = DlbDB(path)

    def list_dlp_by_target_dataset_type(
        self, target_dataset_type: str = None
    ) -> list[dict]:
        """Return all dlps matching the requested target type."""
        return self._dlp_table.list_by_target_dataset_type(target_dataset_type)

    def get_dlp_by_id(self, dlp_id: str) -> tuple[dict, list[dict]]:
        """Get DLP information by its ID, along with associated DLBs."""
        dlp = self._dlp_table.get_by_id(dlp_id)
        dlb_ids = [] if dlp is None else list(dlp["loading_blocks"].values())
        dlbs = self._dlb_table.get_all_by_value("dlb_id", dlb_ids) if dlb_ids else []
        return dlp, dlbs

    def remove_dlp_by_id(self, dlp_id: str):
        """Removes a data loading plan (DLP) from the database, along with its associated
        data loading blocks (DLBs).

        Args:
            dlp_id: the DataLoadingPlan id
        """
        dlp, dlbs = self.get_dlp_by_id(dlp_id)

        # TODO: should we raise if dlp is None?
        if dlp is not None:
            self._dlp_table.delete_by_id(dlp_id)
            for dlb in dlbs:
                self._dlb_table.delete_by_id(dlb["dlb_id"])

    def save_data_loading_plan(
        self, data_loading_plan: Optional[DataLoadingPlan]
    ) -> Union[str, None]:
        """Save a DataLoadingPlan to the database.

        This function saves a DataLoadingPlan to the database, and returns its ID.

        Raises:
            FedbiomedDatasetManagerError: bad data loading plan name (size, not unique)

        Args:
            data_loading_plan: the DataLoadingPlan to be saved, or None.

        Returns:
            The `dlp_id` if a DLP was saved, or None
        """
        if data_loading_plan is None:
            return None

        if len(data_loading_plan.desc) < 4:
            _msg = (
                f"{ErrorNumbers.FB316.value}: Cannot save data loading plan, "
                "DLP name needs to have at least 4 characters."
            )
            logger.error(_msg)
            raise FedbiomedError(_msg)

        dlp_same_name = self._dlp_table.get_all_by_value("name", data_loading_plan.desc)
        if dlp_same_name:
            _msg = (
                f"{ErrorNumbers.FB316.value}: Cannot save data loading plan, "
                "DLP name needs to be unique."
            )
            logger.error(_msg)
            raise FedbiomedError(_msg)

        dlp_metadata, dlbs_metadata = data_loading_plan.serialize()
        _ = self._dlp_table.create(dlp_metadata)
        for dlb_metadata in dlbs_metadata:
            _ = self._dlb_table.create(dlb_metadata)
        return data_loading_plan.dlp_id
