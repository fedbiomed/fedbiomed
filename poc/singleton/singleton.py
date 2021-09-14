from time import sleep
from multiprocessing import Process
from threading import Lock

# singleton design pattern PoC

class SingletonMeta(type):

    _objects = {}
    _lock_instantiation = Lock()

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        print("CALL", cls)
        with cls._lock_instantiation:
            if cls not in cls._objects:
                object = super().__call__(*args, **kwargs)
                cls._objects[cls] = object
        return cls._objects[cls]


class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        print("INIT S1 ", self)

    def method(self):
        print("METHOD S1", self)

class Singleton2(metaclass=SingletonMeta):
    def __init__(self):
        print("INIT S2", self)

    def method(self):
        print("METHOD S2", self)


def fonction_sub():
  sleep(2)
  print('\nsubprocess forked before instantiation : will create an instance at first call \n')
  c = Singleton2()
  d = Singleton2()
  c.method()
  d.method()


p = Process(target=fonction_sub, name='singleton-subproc2')
p.daemon = False
p.start()

print("\none instance created \n")

a = Singleton()

b = Singleton()

a.method()

b.method()

print("\n another class using the same metaclass : also creates one instance \n ")

c = Singleton2()

d = Singleton2()

c.method()

d.method()

def fonction():
  c = Singleton2()
  d = Singleton2()
  c.method()
  d.method()

print('\nfunction : also uses created instance \n')
fonction()

print('\nsubprocess created after singleton instantation : inherits instance from creating process\n')
p = Process(target=fonction, name='singleton-subproc')
p.daemon = False
p.start()

