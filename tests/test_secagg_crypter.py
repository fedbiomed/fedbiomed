import unittest

from fedbiomed.common.secagg import SecaggCrypter
from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError

class TestSecaggCrypter(unittest.TestCase):

    def setUp(self) -> None:

        self.secagg_crypter = SecaggCrypter()
        pass

    def tearDown(self) -> None:
        pass






if __name__ == "__main__":
    unittest.main()