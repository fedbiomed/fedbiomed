# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Interfaces with the node secure aggregation element database table
"""
from abc import ABC, abstractmethod

from tinydb import TinyDB, Query

from fedbiomed.node.environ import environ


class SecaggManager(ABC):
    """Manage the node secagg element database table
    """
    def __init__(self):
        self._db = TinyDB(environ['DB_PATH'])
        self._database = Query()
        self._table = None

    def get_by_secagg_id(self, secagg_id: str):
        pass


class SecaggServkeyManager(SecaggManager):
    """Manage the node server key secagg element database table
    """
    def __init__(self):
        super().__init__()

        # don't use DB read cache to ensure coherence
        # (eg when mixing CLI commands with a GUI session)
        self._table = self._db.table(name='SecaggServkey', cache_size=0)


class SecaggBiprimeManager(SecaggManager):
    """Manage the node biprime secagg element database table
    """
    def __init__(self):
        super().__init__()

        # don't use DB read cache to ensure coherence
        # (eg when mixing CLI commands with a GUI session)
        self._table = self._db.table(name='SecaggBiprime', cache_size=0)
