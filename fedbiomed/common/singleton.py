# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Singleton metaclass. used to easily create thread safe singleton classes"""

# =======================================================================
# WARNING WARNING WARNING WARNING WARNING WARNING WARNING
#
#  do not import *ANY* fedbiomed module here, to avoid dependency loops!
#
# WARNING WARNING WARNING WARNING WARNING
# =======================================================================


import threading
from abc import ABCMeta


class SingletonABCMeta(ABCMeta, type):
    _objects = {}
    _lock_instantiation = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock_instantiation:
            if cls not in cls._objects:
                object_ = super().__call__(*args, **kwargs)
                cls._objects[cls] = object_

        return cls._objects[cls]


class SingletonMeta(type):
    """This (meta) class is a thread safe singleton implementation. It should be used as a metaclass of a new class
    (NC), which will then provide a singleton-like class (i.e. an instance of the class NC will be a singleton)

    This metaclass is used in several fedbiomed classes
    (Request, FedLogger,...)
    """

    _objects = {}
    _lock_instantiation = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """Replace default class creation for classes using this metaclass, executed before the constructor"""

        with cls._lock_instantiation:
            if cls not in cls._objects:
                object_ = super().__call__(*args, **kwargs)
                cls._objects[cls] = object_

        return cls._objects[cls]
