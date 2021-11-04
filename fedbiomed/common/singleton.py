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

            """ 
            else:
                # some singleton need different behavior, which should
                # be incorporated here
                # Example: 
                fullclassname = cls.__module__ + "." + cls.__name__

                if fullclassname  == 'fedbiomed.researcher.monitor.Monitor':
                    # ....
            """
                
        return cls._objects[cls]
