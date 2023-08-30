# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''
Interfaces with a tinyDB database for converting search results to dict.
'''

from tinydb.table import Table, Document


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


class DBTable(Table):
    """Extends TinyDB table to cast Document type to dict"""

    @cast_
    def search(self, *args, **kwargs):
        return super().search(*args, **kwargs)


    @cast_
    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)

    @cast_
    def all(self, *args, **kwargs):
        return super().all(*args, **kwargs)
