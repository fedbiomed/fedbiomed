# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Interfaces with a tinyDB database for converting search results to dict.
"""

from tinydb import Query, TinyDB, where
from typing import Any, Dict, List, Optional
from tinydb.table import Document, Table

from fedbiomed.common.exceptions import FedbiomedError


def cast_(func):
    """Decorator function for typing casting"""

    # Wraps TinyDb get, all, search and upsert methods
    def wrapped(*args, **kwargs):
        add_docs = kwargs.get("add_docs")
        if add_docs is not None:
            kwargs.pop("add_docs")

        document = func(*args, **kwargs)
        if isinstance(document, list):
            casted = [dict(r) for r in document]
        elif isinstance(document, Document):
            casted = dict(document)
        else:
            # Plain python type
            casted = document

        if add_docs:
            return casted, document
        else:
            return casted

    return wrapped


class DB: 
    def __init__(self, path, table_name: str = 'database'):
       self._db = TinyDB(path)
       self._database = DBTable(storage=self._db.storage, name=table_name)


class DBTable(Table):
    """Extends TinyDB table to cast Document type to dict"""
    
    # # ---------- CREATE ----------
    # def create(self, doc: Dict[str, Any]) -> int:
    #     return super().insert(doc)

    # def create_(self, data: Dict[str, Any]) -> int:
    #     """Insert a new dataset and return its id in the TinyDB."""
    #     return super().insert(data)
    
    # # ---------- READ ----------
    # def get_by(self, field: str, value: Any) -> Optional[Dict[str, Any]]:
    #     return super().get(where(field) == value)
    
    # def get_database_by_id(self, dataset_id: int) -> Optional[Dict[str, Any]]:
    #     """Get a single dataset by dataset_id (or None if missing)."""
    #     return super().get(doc_id=dataset_id)

    # def get_database_by_name(self, name: str) -> Optional[Dict[str, Any]]:
    #     """Get a single dataset by name (or None if missing)."""
    #     return super().get(Query().name == name)

    # # ---------- UPDATE ----------
    # def update_database_by_id(self, doc_id: int, **kwargs) -> None:
    #     """Update a dataset by id. Raise error if dataset not found."""
    #     try:
    #         super().update(fields=kwargs, doc_ids=[doc_id])
    #     except Exception as e:
    #         raise FedbiomedError(f"Could not update dataset id {doc_id}: {e}")
    
    # def update_database_by_name(self, name: str, **kwargs) -> None:
    #     """Update a dataset by name. Raise error if dataset not found."""
    #     try:
    #         super().update(fields=kwargs, cond=(Query().name == name))
    #     except Exception as e:
    #         raise FedbiomedError(f"Could not update dataset {name}: {e}")
        
    # # ---------- DELETE ----------
    # def delete_by(self, field: str, value: Any) -> bool:
    #     return bool(super().remove(where(field) == value))

    # def delete_user(self, doc_id: int) -> bool:
    #     """Delete a dataset by id. Returns True if removed."""
    #     removed = super().remove(doc_ids=[doc_id])
    #     return bool(removed)
    
    # ---------- OVERRIDE TINYDB METHODS WITH CASTING ----------
    
    @cast_
    def create(self, *args, **kwargs):
        return super().insert(*args, **kwargs)
    
    @cast_
    def search(self, *args, **kwargs):
        return super().search(*args, **kwargs)

    @cast_
    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)

    @cast_
    def all(self, *args, **kwargs):
        return super().all(*args, **kwargs)
    
    @cast_
    def update(self, *args, **kwargs):
        return super().update(*args, **kwargs)
    
    @cast_
    def delete(self, *args, **kwargs):
        return super().remove(*args, **kwargs)
    

