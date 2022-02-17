import unittest
import random

from fedbiomed.common.singleton import SingletonMeta


class DummySingleton(metaclass=SingletonMeta):
    """
    provide a singleton for the test
    """
    def __init__(self):
        self._id = random.random()

    def id(self):
        return self._id

    def setId(self, id):
        self._id = id

class AnotherSingleton(metaclass=SingletonMeta):
    """
    another one bites the dust
    """
    def __init__(self, id = None):

        if id is None:
            self._id = random.random()
        else:
            self._id = id

    def identity(self):
        return self._id

    def setIdentity(self, id):
        self._id = id


class TestSingleton(unittest.TestCase):
    '''
    Test the SingletonMeta mechanism
    '''
    # before all tests
    def setUp(self):
        self.d1 = DummySingleton()
        pass

    # after all tests
    def tearDown(self):
        pass


    def test_01_dummysingleton(self):
        '''
        test singleton mechanism for DummySingleton
        '''

        d2 = DummySingleton()
        d3 = DummySingleton()

        self.assertEqual( self.d1, d2)
        self.assertEqual( self.d1, d3)
        self.assertEqual( self.d1, DummySingleton())

        old = self.d1.id()
        new = 3.14

        self.d1.setId( new )
        self.assertEqual(          self.d1.id() , new)
        self.assertEqual(               d2.id() , new)
        self.assertEqual(               d3.id() , new)
        self.assertEqual( DummySingleton().id() , new)

        self.assertTrue(          self.d1 is d3)
        self.assertTrue(          self.d1 is d3)
        self.assertTrue(               d2 is d3)
        self.assertTrue( DummySingleton() is d3)

        pass

    def test_02_another_singleton(self):
        '''
        test singleton mechanism for 2 singletons
        '''

        s1 = AnotherSingleton()
        s2 = AnotherSingleton( id = 3.14 )

        self.assertTrue( s1 is s2 )

    def test_03_both_singleton(self):
        '''
        test singleton mechanism for 2 singletons
        '''

        s1 = DummySingleton()
        s2 = AnotherSingleton()

        self.assertFalse( s1 is s2 )


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
