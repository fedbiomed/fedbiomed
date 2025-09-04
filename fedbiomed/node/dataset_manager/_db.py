from typing import Any, Dict, List, Optional

from tinydb import Query, Storage, where
from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.db import DB, DBTable
from tinydb.table import Document

from fedbiomed.common.exceptions import FedbiomedError


class DlpDB(DB):
    """CRUD specialized for Data Loading Plans documents keyed by 'dlp_id'."""

    # ---- Read
    def get_by_id(self, id) -> Optional[Dict[str, Any]]:
        """ Get a single dlp by dlp_id (or None if missing)."""
        return self.get_by("dlp_id", id)
    
    # ---- Delete
    def delete_by_id(self, dataset_id: str) -> List[int]:
        """ Delete by dataset_id. Returns True if any docs were removed."""
        return self.delete_by("dlp_id", dataset_id)

    # ---- Update (partial)
    def update(self, value: Dict[str, Any]) -> List[int]:
        """ Partial update. Returns True if any docs were updated.
        DBTable.update returns list of updated doc_ids (cast_ keeps it as list[int]).
        """
        return self.update_by("dlp_id", value)
    
    def list_by_dataset_type(self, dataset_type: str):
        """ List all DLPs for a given dataset type."""
        if not isinstance(dataset_type, str):
                raise FedbiomedError(
                    f"Wrong input type for target_dataset_type. "
                    f"Expected str, got {type(dataset_type)} instead."
                )
        if dataset_type not in [t.value for t in DatasetTypes]:
            raise FedbiomedError(
                "target_dataset_type should be of the values defined in "
                "fedbiomed.common.constants.DatasetTypes"
            )
        return self.get_by("dataset_type", dataset_type)


class DatasetDB(DB):
    """CRUD specialized for dataset documents keyed by 'dataset_id'."""

    # ---- Read
    def get_by_id(self, id) -> Optional[Dict[str, Any]]:
        """Get a single dataset by dataset_id (or None if missing)."""
        return self.get_by("dataset_id", id)
    
    def get_by_tag(self, tags) -> List[Document]:
        """Get the list of datasets which contain all the given tags (or None if missing)."""
        return self.get_all_by("tags", tags)

    # ---- Delete
    def delete_by_id(self, dataset_id: str) -> List[int]:
        """Delete by dataset_id. Returns True if any docs were removed."""
        return self.delete_by("dataset_id", dataset_id)

    # ---- Update (partial)
    def update(self, value: Dict[str, Any]) -> List[int]:
        """
        Partial update. Returns True if any docs were updated.
        DBTable.update returns list of updated doc_ids (cast_ keeps it as list[int]).
        """
        return self.update_by("dataset_id", value)

