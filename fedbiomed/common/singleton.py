import sys
import threading

#
# WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
#
# do not import fedbiomed module here, to avoid dependancy loop !
#
# WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
#

class SingletonMeta(type):
    """
    This (meta) class is a thread safe singleton implementation.
    It should be used as a metaclass of a new class (NC), which will
    then provide a singleton-like class (i.e an instance of the
    class NC will be a singleton)

    This metaclass is used in several fedbiomed classes
    (Request, FedLogger,...)
    """

    _objects = {}
    _lock_instantiation = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Replace default class creation for classes using this metaclass,
        executed before the constructor
        """

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
                # fedbiomed.researcher.monitor.Monitor specific code
                #
                if fullclassname  == 'fedbiomed.researcher.monitor.Monitor':
                    # Change the tensorboard state with given new state if the singleton
                    # class has been already constructed
                    cls._objects[cls].reconstruct(kwargs['tensorboard'])

                #
                # environment instanciation
                #
                # detect that we instanciated fedbiomed.common.config.Config
                # twice:
                #
                # once as fedbiomed.researcher.config
                # once as fedbiomed.node.config
                #
                # this is a coding error, please review the code !!!!
                #
                if fullclassname == 'fedbiomed.common.environ.Environ':

                    if 'component' in kwargs:
                        #
                        # did we call Environ() with a 'component' argument
                        # different than the first call to Environ()
                        #
                        if not kwargs['component'] == cls._objects[cls]._storage['COMPONENT_TYPE'] :
                            print("CRITICAL: environment has already been instanciated as a",
                                  cls._objects[cls]._storage['COMPONENT_TYPE'])
                            print("Fed-Biomed may behave weird !")
                            print("You may:")
                            print("- review the code")
                            print("- or reset the notebook/notelab")
                            sys.exit(-1)


        return cls._objects[cls]
