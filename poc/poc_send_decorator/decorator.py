#
# class decorator: add the send() method to a decorated class
#
"""
class decorator: add the send() method to a decorated class
this decorator should only ny used on a NN Model.

the send() method is called from researcher side of fedbiomed
send() manages the resereacher/node communication and the data
transfert through
"""


def transfer(initial_class):
    """
    Decorator: use on classes only
    """
    class NewCls():
        """
        Class container to wrap the old class into a decorated class
        """
        def __init__(self,*args,**kwargs):
            #
            # TODO: test that initial_class is a NN model and print a warning # pylint: disable=W0511
            self.initial_instance = initial_class(*args,**kwargs)

            # test that initial_class does not already provide a send() method
            try:
                self.initial_instance.__getattribute__("send")
            except AttributeError:
                pass
            else:
                print("Warning: class",
                      self.initial_instance.__class__,
                      "has already a send() method. Decorating it is dangerous !")

        def __getattribute__(self,s):
            """
            this is called whenever any attribute of a NewCls object is accessed.
            This function first tries to get the attribute of NewCls and run it
            (in this example, only send() is provided)

            if it fails, it then call the attibutes of the initial class
            """
            try:
                _x = super().__getattribute__(s)
            except AttributeError:
                _x = self.initial_instance.__getattribute__(s)
                return _x
            else:
                return _x

        def send(self):
            """
            do the actual work:
            - dump NN model into a file
            - transfer the file through the communication server
            - inform the nodes of the avaibility of the model
            """
            #
            # TODO: test that initial_class is a NN model and print a "not yet implemented" # pylint: disable=W0511

            print("====== entering send() for:", self)
            print("actual  class:", self.__class__)
            print("initial class:", self.initial_instance.__class__)

            print("sending data:", self.initial_instance.my_data)

    return NewCls


#
# validate the code
#
if __name__ == "__main__":
    #
    # define my own class (Test) and decorate it with the decorator
    #
    @transfer
    class Test:
        """
        Test class to validate the previous decorator
        """
        def __init__(self):
            self.my_data = "prout"

        def foofoo(self):
            """
            method specifically provided by intial class
            """
            print("executing Test.foofoo() on", self)

        def data(self):
            """
            get internal data
            """
            print("my_data contains:",self.my_data)

        def set_data(self, value):
            """
            set internal data
            """
            self.my_data = value

    # use my decorated class
    t = Test()

    # simple method
    t.foofoo()

    # access to Test internal storage
    t.data()
    t.set_data("this_is_a_new_data")
    t.data()

    # acces to decorated class method
    t.send()

    @transfer
    class Toto:
        """
        second test - this one already contains send()
        """
        def __init__(self):
            self.my_data = "toto"

        def send(self):
            """
            problem appears here !
            """
            print("this is Toto.send() from:", self)

    t = Toto()
    t.send()
