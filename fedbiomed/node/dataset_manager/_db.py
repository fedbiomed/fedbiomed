from typing import Any, Dict, List, Optional

from tinydb import Query
from fedbiomed.common.db import DBTable
from tinydb.table import Document


class DatasetDB(DBTable):
    """CRUD specialized for dataset documents keyed by 'dataset_id'."""
    UNIQUE_KEY = "dataset_id"

    # C
    def create_dataset(self, ds: Dict[str, Any]) -> str:
        """Insert a dataset (fails if the same dataset_id already exists)."""
        ds_id = ds.get(self.UNIQUE_KEY)
        if not ds_id:
            raise ValueError("dataset requires 'dataset_id'")
        if self.get_by(self.UNIQUE_KEY, ds_id):
            raise ValueError(f"dataset_id already exists: {ds_id}")
        self.insert(ds)
        return ds_id

    # R
    def read_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self.get_by(self.UNIQUE_KEY, dataset_id)

    def list_datasets(self) -> List[Document]:
        return self.all()

    # U
    def update_dataset(self, dataset_id: str, patch: Dict[str, Any]) -> bool:
        return self.update_by(self.UNIQUE_KEY, dataset_id, patch)

    # D
    def delete_dataset(self, dataset_id: str) -> bool:
        return self.delete_by(self.UNIQUE_KEY, dataset_id)

    # Bulk helper for the provided payload shape
    def upsert_from_payload(self, payload: Dict[str, Any]) -> None:
        """
        Payload shape:
        {
          "Datasets": {
            "1": {...}, "2": {...}
          }
        }
        """
        items = payload.get("Datasets", {})
        for _, ds in items.items():
            ds_id = ds.get(self.UNIQUE_KEY)
            if not ds_id:
                continue
            existing = self.read_dataset(ds_id)
            if existing:
                self.update_dataset(ds_id, ds)
            else:
                self.create_dataset(ds)

