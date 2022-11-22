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
                object = super().__call__(*args, **kwargs)
                cls._objects[cls] = object

            else:
                #
                # some singleton need different behavior then
                # "instanciated" twice
                #
                fullclassname = cls.__module__ + "." + cls.__name__

                #
                # environment instanciation
                #
                # detect that we instanciated fedbiomed.common.environ.Environ
                # twice:
                # - once from fedbiomed.researcher.environ
                # - once from fedbiomed.node.environ
                # (order does not matter)
                #
                # if this happends, it means that it is a coding error,
                # please review the code !!!!
                #
                if fullclassname == 'fedbiomed.common.environ.Environ':

                    if 'component' in kwargs:
                        #
                        # did we call Environ() with a 'component' argument
                        # different than the first call to Environ()
                        #
                        if not kwargs['component'] == cls._objects[cls]._values['COMPONENT_TYPE'] :
                            print("CRITICAL: environment has already been instanciated as a",
                                  cls._objects[cls]._values['COMPONENT_TYPE'])
                            print("Fed-BioMed may behave weird !")
                            print("You may:")
                            print("- review/correct the code")
                            print("- or reset the notebook/notelab before executing the cell content")
                    else:
                        #
                        # we called directly fedbiomed.common.environ.Environ().values()
                        # *after* the singleton has already been correctly iniitiated
                        #
                        # this is a feature what we may need
                        # the message is just for debugging purpose
                        print("DEBUG: singleton environ called as fedbiomed.common.environ.Environ().values()")

        return cls._objects[cls]
