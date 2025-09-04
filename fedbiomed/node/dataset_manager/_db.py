from typing import Any, Dict, List, Optional

from tinydb import Query, Storage
from fedbiomed.common.db import DB, DBTable
from tinydb.table import Document


class DatasetDB():
    """CRUD specialized for dataset documents keyed by 'dataset_id'."""

    def __init__(self, db_path: str):
        db = DB(db_path)
        self._table = db._database  # type: DBTable

    # ---- Create
    def create(self, ds: Dict[str, Any]) -> str:
        """
        Insert a dataset. Returns dataset_id.
        Raises ValueError if dataset_id missing or already exists.
        """
        ds_id = ds.get("dataset_id")
        if not ds_id:
            raise ValueError("dataset requires 'dataset_id'")
        if self.read(ds_id) is not None:
            raise ValueError(f"dataset_id already exists: {ds_id}")
        # DBTable.create returns TinyDB doc_id (int); we return the domain id
        _ = self._table.create(ds)
        return ds_id

    # ---- Read
    def read(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self._table.get(Query()["dataset_id"] == dataset_id)

    def list(self) -> List[Document]:
        return self._table.all()

    # ---- Update (partial)
    def update(self, dataset_id: str, patch: Dict[str, Any]) -> bool:
        """
        Partial update. Returns True if any docs were updated.
        DBTable.update returns list of updated doc_ids (cast_ keeps it as list[int]).
        """
        updated_ids = self._table.update(patch, Query()["dataset_id"] == dataset_id)
        return bool(updated_ids)

    # ---- Delete
    def delete(self, dataset_id: str) -> bool:
        """
        Delete by dataset_id. Returns True if any docs were removed.
        """
        removed_ids = self._table.delete(Query()["dataset_id"] == dataset_id)
        return bool(removed_ids)

    # ---- Upsert helper (optional)
    def upsert(self, ds: Dict[str, Any]) -> str:
        """
        Create or update by dataset_id.
        """
        ds_id = ds.get("dataset_id")
        if not ds_id:
            raise ValueError("dataset requires 'dataset_id'")
        if self.read(ds_id) is None:
            self.create(ds)
        else:
            self.update(ds_id, ds)
        return ds_id

